// Q0.31 fixed-point acc_scale variant of the default Gemmini config.
//
// Replaces the float-scale mvout requantize (which loses ~7 LSBs through f32
// rounding on Saturn) with a bit-exact `(acc * mult + 2^30) >> 31` integer
// multiply. Matches the per-tensor-symmetric int8 Q0.31 requantize used by
// TFLite / muRISCV-NN and by `agents/pipeline/reference_kernels.py`.
//
// Wires up to chipyard via `chipyard.Q31GemminiRocketConfig`.

package gemmini

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile.BuildRoCC

import gemmini.Arithmetic.SIntArithmetic

object GemminiQ31Configs {
  // Q0.31 multiply with TFLite-style "+ 2^30 then arith-shift-right 31" rounding.
  // multiplicand_t is SInt(32.W) interpreted as Q0.31 (1.0 ≈ 0x7fff_ffff).
  // Caller folds output_shift into the multiplier (or applies it on CPU).
  val q31AccScale = ScaleArguments[SInt, SInt](
    (t: SInt, m: SInt) => {
      // Saturate-multiply then round and arith-shift right by 31.
      // (t.width=32) * (m.width=32) -> 64-bit signed product; +2^30 bias; >>31.
      val prod    = (t * m)                                  // SInt(64.W)
      val biased  = prod +& (1.S << 30)                      // SInt(65.W)
      val shifted = (biased >> 31).asSInt                    // SInt(34.W)
      // Saturate to int8 (acc->elem cast happens after this, but Saturn's
      // mvout pipeline expects the post-scale value already clipped).
      val maxsat  = ((1 << (t.getWidth - 1)) - 1).S
      val minsat  = (-(1 << (t.getWidth - 1))).S
      val sat     = MuxCase(shifted,
        Seq((shifted > maxsat) -> maxsat, (shifted < minsat) -> minsat))
      sat(t.getWidth - 1, 0).asSInt
    },
    latency        = 4,
    multiplicand_t = SInt(32.W),
    num_scale_units = -1,
    identity = "((acc_scale_t)1 << 30)",  // 0.5 in Q0.31; closest exactly-rep "identity"
    c_str    =
      "({ int64_t y = (((int64_t)(x) * (int64_t)(scale)) + (1LL << 30)) >> 31; " +
      "y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y); })"
  )

  val q31Config = GemminiArrayConfig[SInt, Float, SInt](
    inputType  = SInt(8.W),
    weightType = SInt(8.W),
    accType    = SInt(32.W),

    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(20.W),

    tileRows    = 1,
    tileColumns = 1,
    meshRows    = 16,
    meshColumns = 16,

    dataflow = Dataflow.BOTH,

    sp_capacity  = CapacityInKilobytes(256),
    acc_capacity = CapacityInKilobytes(64),

    sp_banks         = 4,
    acc_banks        = 2,
    sp_singleported  = true,
    acc_singleported = false,

    has_training_convs        = true,
    has_max_pool              = true,
    has_nonlinear_activations = true,

    reservation_station_entries_ld = 8,
    reservation_station_entries_st = 4,
    reservation_station_entries_ex = 16,

    ld_queue_length = 8,
    st_queue_length = 2,
    ex_queue_length = 8,

    max_in_flight_mem_reqs = 16,

    dma_maxbytes = 64,
    dma_buswidth = 128,

    tlb_size = 4,

    // mvin keeps the original float-scale path (typically ACC_SCALE_IDENTITY at
    // mvin time anyway — the bit-exactness we care about is at mvout).
    mvin_scale_args     = GemminiConfigs.defaultConfig.mvin_scale_args,
    mvin_scale_acc_args = None,
    mvin_scale_shared   = false,

    acc_scale_args = Some(q31AccScale),

    num_counter = 8,

    acc_read_full_width = true,
    acc_read_small_width = true,

    ex_read_from_spad = true,
    ex_read_from_acc  = true,
    ex_write_to_spad  = true,
    ex_write_to_acc   = true,

    headerFileName = "gemmini_params_q31.h",
  )
}

// Drop-in mixin: substitutes the default int gemmini's float acc-scale with Q0.31.
class Q31GemminiConfig[T <: Data : Arithmetic, U <: Data, V <: Data](
  gemminiConfig: GemminiArrayConfig[T, U, V] = GemminiQ31Configs.q31Config
) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val gemmini = LazyModule(new Gemmini(gemminiConfig))
      gemmini
    }
  )
})

object GemminiQ31WsConfigs {
  // Same as q31Config but Weight-Stationary-only.  Dropping Dataflow.BOTH
  // removes the OS/WS dataflow mux + half of the c1/c2 register usage
  // inside every PE, and lets the PE control logic collapse to a single
  // path.  Functional equivalence: agents pipeline + xpurt already emit
  // WS-only matmuls (the int8 systolic configs we care about never use
  // OS).  Drops a few hundred LUTs per PE * 256 PEs.
  val q31WsConfig = GemminiQ31Configs.q31Config.copy(
    dataflow = Dataflow.WS,
  )

  // 32x32 mesh on top of q31Ws.  1024 PEs == 1024 DSP48E2 (still 53% of
  // KU040's 1920).  Doubles scratchpad and accumulator rows, keeps
  // total kbits constant (row gets wider, fewer rows).  dma_buswidth
  // is bumped to 256b so a 32x8b mesh column can drain in a single
  // cycle.
  val q31Ws32x32Config = q31WsConfig.copy(
    meshRows     = 32,
    meshColumns  = 32,
    dma_buswidth = 256,
  )

  // Same as 32x32 but doubles acc_capacity from 64 KB to 128 KB.  At the
  // narrower 16x16 mesh, MacroCompiler split the accumulator into
  // 64 tiles of 512 deep x 8 wide (4 Kbit each) which BRAM-pack cleanly.
  // At 32x32 with acc_capacity = 64 KB, each tile collapses to 256 x 8
  // (2 Kbit) which falls under Vivado's BRAM-efficiency threshold and
  // gets demoted to LUTRAM.  Doubling acc_capacity restores 512 deep x
  // 8 wide splits -> BRAM-friendly again.  Burns ~128 extra RAMB18 tiles
  // (fine: KU040 has 1200) to recover ~10K LUTs.
  val q31Ws32x32AccConfig = q31Ws32x32Config.copy(
    acc_capacity = CapacityInKilobytes(128),
  )
}

class Q31WsGemminiConfig extends Q31GemminiConfig(GemminiQ31WsConfigs.q31WsConfig)
class Q31Ws32x32GemminiConfig extends Q31GemminiConfig(GemminiQ31WsConfigs.q31Ws32x32Config)
class Q31Ws32x32AccGemminiConfig extends Q31GemminiConfig(GemminiQ31WsConfigs.q31Ws32x32AccConfig)

object GemminiQ31BothConfigs {
  // 32x32 Q31 with BOTH dataflow retained.  Same mesh-scale + dma_buswidth
  // + acc_capacity bumps as q31Ws32x32AccConfig, but keeps the OS+WS
  // dataflow MUX so the PE has both c1/c2 accumulator paths.  Useful
  // baseline against Q31Ws32x32Acc to isolate the LUT cost of the BOTH
  // dataflow MUX network at 32x32 mesh width.
  val q31_32x32_bothConfig = GemminiQ31Configs.q31Config.copy(
    meshRows     = 32,
    meshColumns  = 32,
    dma_buswidth = 256,
    acc_capacity = CapacityInKilobytes(128),
  )
}

class Q3132x32GemminiConfig extends Q31GemminiConfig(GemminiQ31BothConfigs.q31_32x32_bothConfig)
