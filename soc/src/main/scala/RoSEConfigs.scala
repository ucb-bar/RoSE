//These are the examples configs for the RoSE-supported SoC
// *** Not to be mistaken with the configurations for the RoSE adapters ***
package chipyard.config

import org.chipsalliance.cde.config.{Config}
import rose._
import saturn.common.{VectorParams}
import freechips.rocketchip.prci.{AsynchronousCrossing}
import freechips.rocketchip.subsystem.{InCluster}
// import stereoacc._
import javax.xml.crypto.Data

import firechip.bridgeinterfaces.{DstParams_Container, DstParams, CompleteDataflowConfig, Dataflow}

class AbstractRoseConfig extends Config(
  new chipyard.iobinders.WithRoseIOPunchthrough ++
  new chipyard.config.AbstractConfig)

class RoseTLRocketConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0"),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++         // single rocket-core
  new chipyard.config.AbstractRoseConfig)

// Saturn RVV vector-core variant of RoseTLRocketConfig.
// Adds a Saturn vector unit (V + Zfh scalar-fp16 + Zvfh vector-fp16) so the
// curated RVV vision kernels (conv2d_s8_pc / linear_f16 / lstm_f16, which use
// vfmacc.vv f16, vfredusum f16, zvfh) run on real HW, matching the co-sim ISA
// rv64gcv_zicntr_zihpm_zfh_zvfh. WithRocketVectorUnit sets vfh=true (Zvfh) and
// minFLen=16 (Zfh) on the Rocket tile. VLEN=128/DLEN=64 mirrors the co-sim
// Saturn rocket config (MINV128D64RocketCosimConfig); refParams gives SIMD
// FP16 FMUs for real-HW throughput. Everything else is identical to
// RoseTLRocketConfig (RoSE adapter dst_ports, single big core, AbstractRoseConfig).
class RoseTLRocketSaturnConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0"),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++
  new saturn.rocket.WithRocketVectorUnit(128, 64, VectorParams.refParams) ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++         // single rocket-core + Saturn VPU
  new chipyard.config.AbstractRoseConfig)

// Dual-core Saturn-vector + int8 Gemmini variant for future multi-accelerator
// experiments. TWO Rocket big cores, each carrying BOTH a Saturn RVV vector unit
// (V + Zfh + Zvfh, same VLEN128/DLEN64 as RoseTLRocketSaturnConfig -- runs the RVV
// fp16 vision kernels) AND an int8 Gemmini systolic RoCC (gemmini.GemminiConfigs.
// defaultConfig: SInt8 in / SInt32 acc / Q0.31 per-oc requant -- this is the
// "gemmini_q31" systolic backend ModelBlaster lowers to). Accelerator mixins sit
// to the LEFT of WithNBigCores(2) so the two created tiles pick up BuildRoCC (Gemmini)
// + the vector-unit tile params. Same RoSE adapter dst_ports (DMA0 @ 0x88000000 +
// two reqrsp) as the single-core configs, so the guest device-tree/bridge contract is
// unchanged. A big design (2x Rocket + Saturn VPU + Gemmini) -- built at 30 MHz for
// timing headroom on the U250.
class RoseTLDualRocketSaturnGemminiConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0"),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++
  new gemmini.DefaultGemminiConfig(gemmini.GemminiConfigs.defaultConfig) ++  // int8 Gemmini RoCC (gemmini_q31)
  new saturn.rocket.WithRocketVectorUnit(128, 64, VectorParams.refParams) ++ // Saturn RVV vector unit
  new freechips.rocketchip.rocket.WithNBigCores(2) ++                        // dual rocket big-core
  new chipyard.config.AbstractRoseConfig)

// class RocketStereoAccRoCCConfig extends Config(
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new stereoacc.WithDefaultStereoAccConfig() ++
//   new chipyard.config.AbstractRoseConfig
// )

// class RoseTLRocketStereoAccRoccConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="reqrsp", name="reqrsp0"),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new stereoacc.WithDefaultStereoAccConfig() ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLRocketStereoAccRoccOptConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="reqrsp", name="reqrsp0"),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new stereoacc.WithDefaultStereoAccConfig(use_optimization = true) ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLBOOMDualDMAConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="DMA", DMA_address = 0x89000000L, name="DMA1"),
//     DstParams(port_type="reqrsp", name="reqrsp0"),
//   ))) ++     
//   new boom.common.WithNLargeBooms(1) ++ 
//   new chipyard.config.AbstractRoseConfig
// )

// class RoseTLRocketDualDMAConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="DMA", DMA_address = 0x89000000L, name="DMA1"),
//     DstParams(port_type="reqrsp", name="reqrsp0"),
//   ))) ++     
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig
// )

// class RoseTLRocketStereoAccConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="reqrsp", name="reqrsp0", 
//       df_params = 
//         Seq(CompleteDataflowConfig(StereoAccParams()))
//       ),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLRocketStereoAccDMAConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="DMA", DMA_address = 0x89000000L, name="DMA1",
//       df_params = 
//         Seq(CompleteDataflowConfig(StereoAccParams()))
//       ),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLRocketStereoAccDMADDIO64kBConfig extends Config(
//   new freechips.rocketchip.subsystem.WithInclusiveCache(nWays=2, capacityKB=64) ++
//   new freechips.rocketchip.subsystem.WithNBanks(2) ++
//   new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="DMA", DMA_address = 0x89000000L, name="DMA1",
//       df_params = 
//         Seq(CompleteDataflowConfig(StereoAccParams()))
//       ),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLRocketStereoAccRoCCDDIO64kBConfig extends Config(
//   new freechips.rocketchip.subsystem.WithInclusiveCache(nWays=2, capacityKB=64) ++
//   new freechips.rocketchip.subsystem.WithNBanks(2) ++
//   new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="DMA", DMA_address = 0x89000000L, name="DMA1"),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new stereoacc.WithDefaultStereoAccConfig() ++
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLRocketStereoAccRoCCOptDDIO64kBConfig extends Config(
//   new freechips.rocketchip.subsystem.WithInclusiveCache(nWays=2, capacityKB=64) ++
//   new freechips.rocketchip.subsystem.WithNBanks(2) ++
//   new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="DMA", DMA_address = 0x89000000L, name="DMA1"),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new stereoacc.WithDefaultStereoAccConfig(use_optimization = true) ++
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig)
  

// class RoseTLRocketEdgeDetAccConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="reqrsp", name="reqrsp0", 
//       df_params = 
//         Seq(CompleteDataflowConfig(EdgeDetAccParams()))
//       ),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++        
//   new freechips.rocketchip.subsystem.WithNBigCores(1) ++
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLBOOMConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="reqrsp", name="reqrsp0"),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++                         
//   new boom.common.WithNLargeBooms(1) ++   
//   new chipyard.config.AbstractRoseConfig)

// class RoseTLBOOMGemminiConfig extends Config(
//   new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
//     DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
//     DstParams(port_type="reqrsp", name="reqrsp0"),
//     DstParams(port_type="reqrsp", name="reqrsp1"),
//   ))) ++                          
//   new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
//   new boom.common.WithNLargeBooms(1) ++   
//   new chipyard.config.AbstractRoseConfig)

