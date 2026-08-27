// Standalone gemmini_params.h generator.
//
// Gemmini normally emits gemmini_params.h as a side effect of Chisel elaboration
// (Controller.scala:29 writes config.generateHeader() when the Gemmini LazyModule is
// constructed).  That couples getting a header to running a full elaboration, which for
// a dual-core Saturn+Gemmini RoSE design is minutes of FIRRTL we do not need.
//
// generateHeader() is a *pure* function of GemminiArrayConfig, so this object calls it
// directly: no Parameters, no diplomacy, no FIRRTL.  It runs in seconds and works for
// ANY gemmini shape -- pick a base config and override mesh / capacities / bus width /
// dataflow on the command line.
//
// This is what makes the spike (libgemmini) rebuild flow dimension-agnostic: the header
// that libgemmini compiles against is generated from the same GemminiArrayConfig the RTL
// elaborates from, so the spike model and the hardware cannot silently disagree.
//
// Usage (via soc/sim/build_libgemmini.sh, or directly):
//   sbt "gemmini/runMain gemmini.GemminiHeaderGen --base q31ws --out /tmp/gemmini_params.h"
//   sbt "gemmini/runMain gemmini.GemminiHeaderGen --base q31 --mesh 32 --acc-capacity 128 \
//        --dma-buswidth 256 --out /tmp/gemmini_params.h"
//
// Bases: q31, q31ws, q31ws_32x32, q31ws_32x32_acc, q31_32x32_both, default, lean
//   (`default`/`lean` are the stock f32-acc_scale configs, for A/B against Q0.31.)

package gemmini

import chisel3._
import gemmini.Arithmetic.SIntArithmetic

object GemminiHeaderGen {

  /** Apply shape overrides to any GemminiArrayConfig and render its header.
    * Generic + context-bounded so it works for every base config regardless of T/U/V. */
  private def render[T <: Data : Arithmetic, U <: Data, V <: Data](
      base:     GemminiArrayConfig[T, U, V],
      mesh:     Option[Int],
      spCapKB:  Option[Int],
      accCapKB: Option[Int],
      dmaBus:   Option[Int],
      dataflow: Option[Dataflow.Value]
  ): (String, String) = {
    var c = base
    mesh.foreach     { m => c = c.copy(meshRows = m, meshColumns = m) }
    spCapKB.foreach  { k => c = c.copy(sp_capacity  = CapacityInKilobytes(k)) }
    accCapKB.foreach { k => c = c.copy(acc_capacity = CapacityInKilobytes(k)) }
    dmaBus.foreach   { b => c = c.copy(dma_buswidth = b) }
    dataflow.foreach { d => c = c.copy(dataflow = d) }
    val summary =
      s"mesh=${c.meshRows}x${c.meshColumns} dataflow=${c.dataflow} " +
      s"sp_capacity=${c.sp_capacity} acc_capacity=${c.acc_capacity} " +
      s"dma_buswidth=${c.dma_buswidth} acc_scale=${c.acc_scale_args.map(_.multiplicand_t.toString).getOrElse("none")}"
    (c.generateHeader(), summary)
  }

  def main(args: Array[String]): Unit = {
    var base     = "q31"
    var out      = "gemmini_params.h"
    var mesh:     Option[Int] = None
    var spCap:    Option[Int] = None
    var accCap:   Option[Int] = None
    var dmaBus:   Option[Int] = None
    var dataflow: Option[Dataflow.Value] = None

    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--base"          => base = args(i + 1); i += 2
        case "--out"           => out  = args(i + 1); i += 2
        case "--mesh"          => mesh = Some(args(i + 1).toInt); i += 2
        case "--sp-capacity"   => spCap = Some(args(i + 1).toInt); i += 2
        case "--acc-capacity"  => accCap = Some(args(i + 1).toInt); i += 2
        case "--dma-buswidth"  => dmaBus = Some(args(i + 1).toInt); i += 2
        case "--dataflow"      =>
          dataflow = Some(args(i + 1).toUpperCase match {
            case "WS"   => Dataflow.WS
            case "OS"   => Dataflow.OS
            case "BOTH" => Dataflow.BOTH
            case other  => sys.error(s"GemminiHeaderGen: bad --dataflow '$other' (WS|OS|BOTH)")
          })
          i += 2
        case other => sys.error(s"GemminiHeaderGen: unknown arg '$other'")
      }
    }

    val (header, summary) = base match {
      case "q31"             => render(GemminiQ31Configs.q31Config,               mesh, spCap, accCap, dmaBus, dataflow)
      case "q31ws"           => render(GemminiQ31WsConfigs.q31WsConfig,           mesh, spCap, accCap, dmaBus, dataflow)
      case "q31ws_32x32"     => render(GemminiQ31WsConfigs.q31Ws32x32Config,      mesh, spCap, accCap, dmaBus, dataflow)
      case "q31ws_32x32_acc" => render(GemminiQ31WsConfigs.q31Ws32x32AccConfig,   mesh, spCap, accCap, dmaBus, dataflow)
      case "q31_32x32_both"  => render(GemminiQ31BothConfigs.q31_32x32_bothConfig,mesh, spCap, accCap, dmaBus, dataflow)
      case "default"         => render(GemminiConfigs.defaultConfig,              mesh, spCap, accCap, dmaBus, dataflow)
      case "lean"            => render(GemminiConfigs.leanConfig,                 mesh, spCap, accCap, dmaBus, dataflow)
      case other             => sys.error(s"GemminiHeaderGen: unknown --base '$other'")
    }

    val p = java.nio.file.Paths.get(out)
    Option(p.getParent).foreach(java.nio.file.Files.createDirectories(_))
    java.nio.file.Files.write(p, header.getBytes(java.nio.charset.StandardCharsets.UTF_8))
    println(s"[gemmini-header-gen] base=$base $summary")
    println(s"[gemmini-header-gen] wrote $out")
  }
}
