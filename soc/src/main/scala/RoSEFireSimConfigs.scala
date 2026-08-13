package firechip.chip

import java.io.File

import chisel3._
import chisel3.util.{log2Up}
import org.chipsalliance.cde.config.{Parameters, Config}
import freechips.rocketchip.groundtest.TraceGenParams
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.DCacheParams
import freechips.rocketchip.subsystem._
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.devices.debug.{DebugModuleParams, DebugModuleKey}
// import freechips.rocketchip.diplomacy.{LazyModule}
// import freechips.rocketchip.prci.{AsynchronousCrossing}
import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTParams}

// import firesim.bridges._
// import firesim.configs._

import firechip.bridgestubs._
import firesim.lib.bridges._

class RoseTLRocketMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  // new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketConfig)

// Saturn RVV vector-core variant: identical MMIO-only RoSE FireSim bridge stack
// (WithRoseBridge + MMIO-only bridges + FireSim tweaks) applied on top of the
// Saturn-vector Rocket base (V + Zfh + Zvfh). Target for the U250 bitstream that
// runs the curated RVV fp16 vision kernels on real HW.
class RoseTLRocketSaturnMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketSaturnConfig)

// DMA-datapath variant of RoseTLRocketSaturnMMIOOnlyConfig: identical Saturn+RoSE
// target, but with the host->FPGA bulk DMA datapath turned ON deterministically.
// rose.WithRoseDmaRx sets dmaRx=true on the RoSE adapter params, which ride in the
// bridge's serialized constructor arg and so reach the GoldenGate host-side bridge
// elaboration WITHOUT depending on the shell environment (an exported ROSE_DMA_RX
// does NOT survive FireSim's SSH build dispatch -- that is why this uses a param,
// not the env var). The distinct class NAME also gives FireSim its own deploy
// triplet / build dir, so a DMA bitstream never collides with (or reuses stale
// collateral from) the MMIO Saturn build. Register-map offsets are byte-identical
// between the DMA and MMIO modes.
class RoseTLRocketSaturnDMAMMIOOnlyConfig extends Config(
  new rose.WithRoseDmaRx ++
  new RoseTLRocketSaturnMMIOOnlyConfig)

// Dual-core Saturn-vector + int8 Gemmini FireSim wrapper: the same MMIO-only RoSE
// bridge stack (WithRoseBridge + MMIO-only bridges + FireSim tweaks) on top of the
// dual-core Saturn+Gemmini base (RoseTLDualRocketSaturnGemminiConfig). Target for the
// multi-accelerator U250 bitstream (2x Rocket, each with Saturn RVV + int8 Gemmini RoCC).
class RoseTLDualRocketSaturnGemminiMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLDualRocketSaturnGemminiConfig)

// class RoseTLRocketStereoAccMMIOOnlyDMAConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccDMAConfig) 

// class RoseTLRocketStereoAccDMADDIO64kBMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithTracerVBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccDMADDIO64kBConfig) 

// class RoseTLRocketStereoAccRoCCDDIO64kBMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithTracerVBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccRoCCDDIO64kBConfig) 

// class RoseTLRocketStereoAccRoCCOptDDIO64kBMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithTracerVBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccRoCCOptDDIO64kBConfig) 

// class RoseTLRocketStereoAccMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccConfig) 

// class RoseTLRocketStereoAccRoccMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccRoccConfig) 

// class RoseTLRocketStereoAccRoccOptMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketStereoAccRoccOptConfig) 

// class RoseTLRocketDualDMAFireSimConfig extends Config(
//   new WithRoseBridge ++
//   new WithTracerVBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketDualDMAConfig) 

// class RoseTLRocketDualDMAMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketDualDMAConfig) 

// class RoseTLBOOMDualDMAMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLBOOMDualDMAConfig) 

// class RoseTLRocketEdgeDetAccMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLRocketEdgeDetAccConfig)  

// class RoseTLRocketStereoAccNICConfig extends Config(
//   new WithRoSENICFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new WithNIC ++
//   new chipyard.config.RoseTLRocketStereoAccConfig) 

// class RocketNICConfig extends Config(
//   new WithDefaultFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new WithNIC ++
//   new chipyard.RocketConfig
// )

// class RoseTLBOOMMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLBOOMConfig) 

// class RoseTLBOOMGemminiMMIOOnlyConfig extends Config(
//   new WithRoseBridge ++
//   new WithDefaultMMIOOnlyFireSimBridges ++
//   new WithDefaultMemModel ++
//   new WithFireSimConfigTweaks ++
//   new chipyard.config.RoseTLBOOMGemminiConfig) 