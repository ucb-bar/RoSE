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

// Saturn+RoSE + TACIT instruction-trace streaming bridge. BOTH the RoSE co-sim bridge
// AND the ported TACIT bridge are present: WithTacitBridge (HarnessBinder, streams the
// raw-byte trace off-FPGA via StreamToHostCPU -> host tacit.cc) pairs with the
// WithTraceSinkRawBytePunchthrough IOBinder (punches each tile's tacit_byte egress out to
// the harness) on top of the RoSE MMIO-only bridge stack. Ported from
// riscv-tacit/chipyard@fix-issue-unit (bridge) + tacit@648943d (sink), adapted to our
// 1-lane (single-byte) tacit encoder.
class RoseTLRocketSaturnTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketSaturnTacitConfig)

// Dual-core Saturn-vector + int8 Gemmini FireSim wrapper: the same MMIO-only RoSE
// bridge stack (WithRoseBridge + MMIO-only bridges + FireSim tweaks) on top of the
// dual-core Saturn+Gemmini base (RoseTLDualRocketSaturnGemminiConfig). Target for the
// multi-accelerator U250 bitstream (2x Rocket, each with Saturn RVV + int8 Gemmini RoCC).
class RoseTLDualRocketSaturnGemminiMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLDualRocketSaturnGemminiConfig)

// Q0.31 FireSim wrappers: same MMIO-only RoSE bridge stack as above, over the Q0.31
// integer-mvout-requantize Gemmini bases.  These are the targets to build when you want
// Gemmini results that are bit-comparable against ModelBlaster's scalar Q0.31 golden
// (the fp32-acc_scale wrapper above loses ~7 LSBs on mvout).  Bridge/DT contract is
// identical to the fp32 variant, so guest software needs no change beyond selecting a
// Q31 gemmini_params.h.
class RoseTLDualRocketSaturnGemminiQ31MMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLDualRocketSaturnGemminiQ31Config)

// WS-only Q0.31 variant -- smaller PE footprint, same numerics.
class RoseTLDualRocketSaturnGemminiQ31WsMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLDualRocketSaturnGemminiQ31WsConfig)

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

// ===========================================================================
// F2 build matrix FireSim wrappers: {dual,quad} x {small,large} x {RoSE,no-RoSE}
//
// All eight enable TACIT instruction tracing over the FireSim streaming bridge:
// WithTraceSinkRawBytePunchthrough exposes one TraceSinkRawBytePort per tile
// (system.tacit_bytes.zipWithIndex in IOBinders.scala), and WithTacitBridge binds
// a TacitBridge to each -- so a quad-core build streams four trace channels.
//
// WithDefaultMMIOOnlyFireSimBridges (rather than WithDefaultFireSimBridges) drops
// the NIC and TracerV bridges while keeping TSI/DMI/UART/BlockDev/FASED. TracerV
// is exactly what TACIT replaces, and we have no NIC in these targets. The stock
// f1 RoSE recipe already builds an MMIOOnly config on AWS, so this composes on F2.
// ===========================================================================

// ---- RoSE variants --------------------------------------------------------
class RoseTLDualSmallTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLDualSmallTacitConfig)

class RoseTLQuadSmallTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLQuadSmallTacitConfig)

class RoseTLDualLargeTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLDualLargeTacitConfig)

class RoseTLQuadLargeTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLQuadLargeTacitConfig)

// ---- no-RoSE variants (no WithRoseBridge) ---------------------------------
class SatGemDualSmallTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.SatGemDualSmallTacitConfig)

class SatGemQuadSmallTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.SatGemQuadSmallTacitConfig)

class SatGemDualLargeTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.SatGemDualLargeTacitConfig)

class SatGemQuadLargeTacitMMIOOnlyConfig extends Config(
  new WithTacitBridge ++
  new chipyard.iobinders.WithTraceSinkRawBytePunchthrough ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.SatGemQuadLargeTacitConfig)
