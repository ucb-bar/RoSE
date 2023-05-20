package firesim.firesim

import java.io.File

import chisel3._
import chisel3.util.{log2Up}
import freechips.rocketchip.config.{Parameters, Config}
import freechips.rocketchip.groundtest.TraceGenParams
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.DCacheParams
import freechips.rocketchip.subsystem._
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.devices.debug.{DebugModuleParams, DebugModuleKey}
import freechips.rocketchip.diplomacy.{LazyModule, AsynchronousCrossing}
import testchipip.{BlockDeviceKey, BlockDeviceConfig, TracePortKey, TracePortParams}
import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTParams}
import scala.math.{min, max}

import chipyard.clocking.{ChipyardPRCIControlKey}
import icenet._

import firesim.bridges._
import firesim.configs._

class AirSimIOTLFPGemminiRocketMMIOOnlyConfig extends Config(
  new WithDefaultMMIOOnlyFireSimBridges ++
  new WithDefaultMemModel ++
  new WithFireSimConfigTweaks ++
  new chipyard.config.AirSimIOTLFPGemminiRocketConfig)
