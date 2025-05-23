package firesim.firesim

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
import freechips.rocketchip.diplomacy.{LazyModule, AsynchronousCrossing}
import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTParams}

import firesim.bridges._
import firesim.configs._

class RoseTLRocketMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketConfig) 

class RoseTLRocketStereoAccMMIOOnlyDMAConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccDMAConfig) 

class RoseTLRocketStereoAccDMADDIO64kBMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithTracerVBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccDMADDIO64kBConfig) 

class RoseTLRocketStereoAccRoCCDDIO64kBMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithTracerVBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccRoCCDDIO64kBConfig) 

class RoseTLRocketStereoAccRoCCOptDDIO64kBMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithTracerVBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccRoCCOptDDIO64kBConfig) 

class RoseTLRocketStereoAccMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccConfig) 

class RoseTLRocketStereoAccRoccMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccRoccConfig) 

class RoseTLRocketStereoAccRoccOptMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketStereoAccRoccOptConfig) 

class RoseTLRocketDualDMAFireSimConfig extends Config(
  new WithRoseBridge ++
  new WithTracerVBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketDualDMAConfig) 

class RoseTLRocketDualDMAMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketDualDMAConfig) 

class RoseTLBOOMDualDMAMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLBOOMDualDMAConfig) 

class RoseTLRocketEdgeDetAccMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLRocketEdgeDetAccConfig)  

class RoseTLRocketStereoAccNICConfig extends Config(
  new WithRoSENICFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new WithNIC ++
  new chipyard.config.RoseTLRocketStereoAccConfig) 

class RocketNICConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new WithNIC ++
  new chipyard.RocketConfig
)

class RoseTLBOOMMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLBOOMConfig) 

class RoseTLBOOMGemminiMMIOOnlyConfig extends Config(
  new WithRoseBridge ++
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.RoseTLBOOMGemminiConfig) 