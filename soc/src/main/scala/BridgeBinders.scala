//See LICENSE for license details.

package firesim.firesim

import chisel3._
import chisel3.experimental.{DataMirror, Direction}
import chisel3.util.experimental.BoringUtils

import org.chipsalliance.cde.config.{Field, Config, Parameters}
import freechips.rocketchip.diplomacy.{LazyModule}
import freechips.rocketchip.devices.debug.{Debug, HasPeripheryDebug, ExportDebug, DMI}
import freechips.rocketchip.amba.axi4.{AXI4Bundle}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.prci.{ClockBundle, ClockBundleParameters}
import freechips.rocketchip.util.{ResetCatchAndSync}
import sifive.blocks.devices.uart._

import testchipip.serdes.{ExternalSyncSerialIO}
import testchipip.tsi.{SerialRAM}
import icenet.{CanHavePeripheryIceNIC, SimNetwork, NicLoopback, NICKey, NICIOvonly}

import junctions.{NastiKey, NastiParameters}
import midas.models.{FASEDBridge, AXI4EdgeSummary, CompleteConfig}
import firesim.bridges._
import firesim.configs.MemModelKey
import tracegen.{TraceGenSystemModuleImp}
import cva6.CVA6Tile

import barstools.iocell.chisel._
import chipyard.iobinders._
import chipyard._
import chipyard.harness._

import rose.RosePortIO

object MainMemoryConsts {
  val regionNamePrefix = "MainMemory"
  def globalName(chipId: Int) = s"${regionNamePrefix}_$chipId"
}

trait Unsupported {
  require(false, "We do not support this IOCell type")
}

class FireSimAnalogIOCell extends RawModule with AnalogIOCell with Unsupported {
  val io = IO(new AnalogIOCellBundle)
}
class FireSimDigitalGPIOCell extends RawModule with DigitalGPIOCell with Unsupported {
  val io = IO(new DigitalGPIOCellBundle)
}
class FireSimDigitalInIOCell extends RawModule with DigitalInIOCell {
  val io = IO(new DigitalInIOCellBundle)
  io.i := io.pad
}
class FireSimDigitalOutIOCell extends RawModule with DigitalOutIOCell {
  val io = IO(new DigitalOutIOCellBundle)
  io.pad := io.o
}

case class FireSimIOCellParams() extends IOCellTypeParams {
  def analog() = Module(new FireSimAnalogIOCell)
  def gpio()   = Module(new FireSimDigitalGPIOCell)
  def input()  = Module(new FireSimDigitalInIOCell)
  def output() = Module(new FireSimDigitalOutIOCell)
}

class WithFireSimIOCellModels extends Config((site, here, up) => {
  case IOCellKey => FireSimIOCellParams()
})

class WithTSIBridgeAndHarnessRAMOverSerialTL extends HarnessBinder({
  case (th: FireSim, port: SerialTLPort, chipId: Int) => {
    port.io match {
      case io: ExternalSyncSerialIO => {
        io.clock_in := th.harnessBinderClock
        val ram = Module(LazyModule(new SerialRAM(port.serdesser, port.params)(port.serdesser.p)).module)
        ram.io.ser.in <> io.out
        io.in <> ram.io.ser.out

        // This assumes that:
        // If ExtMem for the target is defined, then FASED bridge will be attached
        // If FASED bridge is attached, loadmem widget is present
        val hasMainMemory = th.chipParameters(chipId)(ExtMem).isDefined
        val mainMemoryName = Option.when(hasMainMemory)(MainMemoryConsts.globalName(chipId))
        TSIBridge(th.harnessBinderClock, ram.io.tsi.get, mainMemoryName, th.harnessBinderReset.asBool)(th.p)
      }
    }
  }
})

class WithNICBridge extends HarnessBinder({
  case (th: FireSim, port: NICPort, chipId: Int) => {
    NICBridge(port.io.clock, port.io.bits)(th.p)
  }
})

class WithUARTBridge extends HarnessBinder({
  case (th: FireSim, port: UARTPort, chipId: Int) =>
    val uartSyncClock = th.harnessClockInstantiator.requestClockMHz("uart_clock", port.freqMHz)
    UARTBridge(uartSyncClock, port.io, th.harnessBinderReset.asBool, port.freqMHz)(th.p)
})

class WithBlockDeviceBridge extends HarnessBinder({
  case (th: FireSim, port: BlockDevicePort, chipId: Int) => {
    BlockDevBridge(port.io.clock, port.io.bits, th.harnessBinderReset.asBool)
  }
})


class WithFASEDBridge extends HarnessBinder({
  case (th: FireSim, port: AXI4MemPort, chipId: Int) => {
    val nastiKey = NastiParameters(port.io.bits.r.bits.data.getWidth,
                                   port.io.bits.ar.bits.addr.getWidth,
                                   port.io.bits.ar.bits.id.getWidth)
    FASEDBridge(port.io.clock, port.io.bits, th.harnessBinderReset.asBool,
      CompleteConfig(th.p(firesim.configs.MemModelKey),
        nastiKey,
        Some(AXI4EdgeSummary(port.edge)),
        Some(MainMemoryConsts.globalName(chipId))))(th.p)
  }
})

class WithTracerVBridge extends HarnessBinder({
  case (th: FireSim, port: TracePort, chipId: Int) => {
    port.io.traces.map(tileTrace => TracerVBridge(tileTrace)(th.p))
  }
})

class WithCospikeBridge extends HarnessBinder({
  case (th: FireSim, port: TracePort, chipId: Int) => {
    port.io.traces.zipWithIndex.map(t => CospikeBridge(t._1, t._2, port.cosimCfg))
  }
})

class WithSuccessBridge extends HarnessBinder({
  case (th: FireSim, port: SuccessPort, chipId: Int) => {
    GroundTestBridge(th.harnessBinderClock, port.io)(th.p)
  }
})

class WithRoseBridge extends HarnessBinder({
  case (th: FireSim, port: RoseAdapterPort, chipId: Int) => {
    RoseBridge(port.io.clock, port.io.bits, th.harnessBinderReset.asBool)(th.p) 
  }
})

// Shorthand to register all of the provided bridges above
class WithDefaultFireSimBridges extends Config(
  new WithTSIBridgeAndHarnessRAMOverSerialTL ++
  new WithNICBridge ++
  new WithUARTBridge ++
  new WithBlockDeviceBridge ++
  new WithFASEDBridge ++
  new WithFireSimMultiCycleRegfile ++
  new WithFireSimFAME5 ++
  new WithTracerVBridge ++
  new WithFireSimIOCellModels
)

// Shorthand to register all of the provided mmio-only bridges above
class WithDefaultMMIOOnlyFireSimBridges extends Config(
  new WithTSIBridgeAndHarnessRAMOverSerialTL ++
  new WithUARTBridge ++
  new WithBlockDeviceBridge ++
  new WithFASEDBridge ++
  new WithFireSimMultiCycleRegfile ++
  new WithFireSimFAME5 ++
  new WithRoseBridge ++ 
  new WithFireSimIOCellModels
)