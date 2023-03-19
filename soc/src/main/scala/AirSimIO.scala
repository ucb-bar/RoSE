package chipyard.example

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.BaseSubsystem
import freechips.rocketchip.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf


// DOC include start: AirSimIO params
case class AirSimIOParams(
  address: BigInt = 0x2000,
  width: Int = 32,
  useAXI4: Boolean = false)
// DOC include end: AirSimIO params

// DOC include start: AirSimIO key
case object AirSimIOKey extends Field[Option[AirSimIOParams]](None)
// DOC include end: AirSimIO key

class AirSimIOIO(val w: Int) extends Bundle {
  val clock = Input(Clock())
  val reset = Input(Bool())

  // Data being written from AirSim to the SoC (SoC Side)
  val rx_deq_ready = Input(Bool())
  val rx_deq_valid = Output(Bool())
  val rx_deq_bits = Output(UInt(w.W))

  // Data being written from AirSim to the SoC (Bridge Side)
  val rx_enq_ready = Output(Bool())
  val rx_enq_valid = Input(Bool())
  val rx_enq_bits = Input(UInt(w.W))

  // Data being written from the SoC to AirSim (SoC Side)
  val tx_enq_ready = Output(Bool())
  val tx_enq_valid = Input(Bool())
  val tx_enq_bits = Input(UInt(w.W))

  // Data being written from the SoC to AirSim (Bridge Side)
  val tx_deq_ready = Input(Bool())
  val tx_deq_valid = Output(Bool())
  val tx_deq_bits = Output(UInt(w.W))
}

class AirSimPortIO extends Bundle {
  val port_rx_enq_ready = Output(Bool())
  val port_rx_enq_valid = Input(Bool())
  val port_rx_enq_bits = Input(UInt(32.W))

  val port_tx_deq_ready = Input(Bool())
  val port_tx_deq_valid = Output(Bool())
  val port_tx_deq_bits = Output(UInt(32.W))
}

trait AirSimIOTopIO extends Bundle {
  val top_rx_enq_ready = Output(Bool())
  val top_rx_enq_valid = Input(Bool())
  val top_rx_enq_bits = Input(UInt(32.W))

  val top_tx_deq_ready = Input(Bool())
  val top_tx_deq_valid = Output(Bool())
  val top_tx_deq_bits = Output(UInt(32.W))
}


trait HasAirSimIOIO extends BaseModule {
  val w: Int
  val io = IO(new AirSimIOIO(w))
}

trait HasAirSimPortIO extends BaseModule {
  val port = IO(new AirSimPortIO())
}

// DOC include start: AirSimIO chisel
class AirSimIOMMIOChiselModule(val w: Int) extends Module
  with HasAirSimIOIO
{

  val txfifo = Module(new Queue(UInt(w.W), 64))
  val rxfifo = Module(new Queue(UInt(w.W), 64))

  // Enqueueing logic
  io.rx_enq_ready     := rxfifo.io.enq.ready
  rxfifo.io.enq.valid := io.rx_enq_valid
  rxfifo.io.enq.bits  := io.rx_enq_bits

  io.tx_enq_ready     := txfifo.io.enq.ready
  txfifo.io.enq.valid := io.tx_enq_valid
  txfifo.io.enq.bits  := io.tx_enq_bits

  // Dequeing logic
  rxfifo.io.deq.ready := io.rx_deq_ready
  io.rx_deq_valid     := rxfifo.io.deq.valid
  io.rx_deq_bits      := rxfifo.io.deq.bits

  txfifo.io.deq.ready := io.tx_deq_ready
  io.tx_deq_valid     := txfifo.io.deq.valid
  io.tx_deq_bits      := txfifo.io.deq.bits
}
// DOC include end: AirSimIO chisel

// DOC include start: AirSimIO instance regmap

trait AirSimIOModule extends HasRegMap {
  val io: AirSimIOTopIO

  implicit val p: Parameters
  def params: AirSimIOParams
  val clock: Clock
  val reset: Reset

  val tx_data = Wire(new DecoupledIO(UInt(params.width.W)))
  val rx_data = Wire(new DecoupledIO(UInt(params.width.W)))
  val status = Wire(UInt(2.W))

  val impl = Module(new AirSimIOMMIOChiselModule(params.width))

  impl.io.clock := clock
  impl.io.reset := reset.asBool

  impl.io.tx_enq_bits  := tx_data.bits
  impl.io.tx_enq_valid := tx_data.valid
  tx_data.ready        := impl.io.tx_enq_ready

  rx_data.bits         := impl.io.rx_deq_bits
  rx_data.valid        := impl.io.rx_deq_valid
  impl.io.rx_deq_ready := rx_data.ready

  status := Cat(impl.io.tx_enq_ready, impl.io.rx_deq_valid)

  // Connect to top IO
  io.top_rx_enq_ready  := impl.io.rx_enq_ready
  impl.io.rx_enq_bits  := io.top_rx_enq_bits
  impl.io.rx_enq_valid := io.top_rx_enq_valid

  io.top_tx_deq_valid  := impl.io.tx_deq_valid
  io.top_tx_deq_bits   := impl.io.tx_deq_bits
  impl.io.tx_deq_ready := io.top_tx_deq_ready 

  regmap(
    0x00 -> Seq(
      RegField.r(2, status)), // a read-only register capturing current status
    0x08 -> Seq(
      RegField.w(params.width, tx_data)), // write-only, y.valid is set on write
    0x0C -> Seq(
      RegField.r(params.width, rx_data))) // read-only, airsimio.ready is set on read
}
// DOC include end: AirSimIO instance regmap

// DOC include start: AirSimIO router
class AirSimIOTL(params: AirSimIOParams, beatBytes: Int)(implicit p: Parameters)
  extends TLRegisterRouter(
    params.address, "airsimio", Seq("ucbbar,airsimio"),
    beatBytes = beatBytes)(
      new TLRegBundle(params, _) with AirSimIOTopIO)(
      new TLRegModule(params, _, _) with AirSimIOModule)

class AirSimIOAXI4(params: AirSimIOParams, beatBytes: Int)(implicit p: Parameters)
  extends AXI4RegisterRouter(
    params.address,
    beatBytes=beatBytes)(
      new AXI4RegBundle(params, _) with AirSimIOTopIO)(
      new AXI4RegModule(params, _, _) with AirSimIOModule)
// DOC include end: AirSimIO router

// DOC include start: AirSimIO lazy trait
trait CanHavePeripheryAirSimIO { this: BaseSubsystem =>
  private val portName = "airsimio"

  val airsimio = p(AirSimIOKey).map { params =>
    val airsimmod = LazyModule(new AirSimIOTL(params, pbus.beatBytes)(p))
    pbus.toVariableWidthSlave(Some(portName)) { airsimmod.node }

    val outer_io = InModuleBody {
      val outer_io = IO(new ClockedIO(new AirSimPortIO)).suggestName(portName)


      //val outer_io = IO(new AirSimPortIO).suggestName(portName)
      outer_io.clock := airsimmod.module.clock

      outer_io.bits.port_rx_enq_ready           := airsimmod.module.io.top_rx_enq_ready
      airsimmod.module.io.top_rx_enq_bits       := outer_io.bits.port_rx_enq_bits
      airsimmod.module.io.top_rx_enq_valid      := outer_io.bits.port_rx_enq_valid

      airsimmod.module.io.top_tx_deq_ready      := outer_io.bits.port_tx_deq_ready
      outer_io.bits.port_tx_deq_bits            := airsimmod.module.io.top_tx_deq_bits
      outer_io.bits.port_tx_deq_valid           := airsimmod.module.io.top_tx_deq_valid

      outer_io
    }
    outer_io
  }
}
// DOC include end: AirSimIO lazy trait

// DOC include start: AirSimIO imp trait
trait CanHavePeripheryAirSimIOModuleImp extends LazyModuleImp {
  val outer: CanHavePeripheryAirSimIO
}
// DOC include end: AirSimIO imp trait


// DOC include start: AirSimIO config fragment
class WithAirSimIO(useAXI4: Boolean) extends Config((site, here, up) => {
  case AirSimIOKey => Some(AirSimIOParams(useAXI4 = useAXI4))
})
// DOC include end: AirSimIO config fragment
