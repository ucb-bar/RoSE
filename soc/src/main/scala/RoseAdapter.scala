//    ____       U  ___ u   ____     U _____ u 
// U |  _"\ u     \/"_ \/  / __"| u  \| ___"|/ 
//  \| |_) |/     | | | | <\___ \/    |  _|"   
//   |  _ <   .-,_| |_| |  u___) |    | |___   
//   |_| \_\   \_)-\___/   |____/>>   |_____|  
//   //   \\_       \\      )(  (__)  <<   >>  
//  (__)  (__)     (__)    (__)      (__) (__) 
// --- --- Get some RoSE adapter rolling --- ---

package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes}
import freechips.rocketchip.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf
import testchipip.TLHelper

// Parameter used accross the project
case class RoseAdapterParams(
  // This is the base address of the adapter TL Registers
  address: BigInt = 0x2000,
  // This is the width of the queues
  width: Int = 32,
  // This decides whether to use AXI4 or TileLink
  useAXI4: Boolean = false,
  // This is the base address of the DMA Engine Memory Location
  base: BigInt = 0x88000000L)
case object RoseAdapterKey extends Field[Option[RoseAdapterParams]](None)

// Core IO of the adapter
class RoseAdapterIO(val w: Int) extends Bundle {
  val clock = Input(Clock())
  val reset = Input(Bool())
  // Bridge -> SoC
  val rx = new Bundle{
    val enq = Flipped(Decoupled(UInt(w.W)))
    val deq = Decoupled(UInt(w.W))
  }
  // SoC -> Bridge
  val tx = new Bundle{
    val enq = Flipped(Decoupled(UInt(w.W)))
    val deq = Decoupled(UInt(w.W))
  }
  // This to dequeue the cam FIFO
  val cam = Decoupled(UInt(w.W))
  // Indicating which of the two buffers the image lies in
  val cam_buffer = Input(UInt(1.W))
}

// PortIO is used for 
class RosePortIO extends Bundle {
  val enq = Flipped(Decoupled(UInt(32.W)))
  val deq = Decoupled(UInt(32.W))
}

// TopIO is used for RoseAdapterTL communicating to the cam DMA engine, and the TL registers
trait RoseAdapterTopIO extends Bundle {
    val enq = Flipped(Decoupled(UInt(32.W)))
    val deq = Decoupled(UInt(32.W))
    val cam = Decoupled(UInt(32.W))
    val cam_buffer = Input(UInt(1.W))
    val counter_max = Output(UInt(32.W))
}

trait HasRoseAdapterIO extends BaseModule {
  val w: Int
  val io = IO(new RoseAdapterIO(w))
}

trait HasRosePortIO extends BaseModule {
  val port = IO(new RosePortIO())
}

class RoseAdapterMMIOChiselModule(val w: Int) extends Module
  with HasRoseAdapterIO
{
  val txfifo = Module(new Queue(UInt(w.W), 64))
  val rxfifo = Module(new Queue(UInt(w.W), 64))

  val camfifo = Module(new Queue(UInt(w.W), 32))
  val otherfifo = Module(new Queue(UInt(w.W), 32))

  val arbiter = Module(new RoseAdapterArbiter(w))

  camfifo.io.enq <> arbiter.io.cam

  otherfifo.io.enq <> arbiter.io.other
  rxfifo.io.deq <> arbiter.io.rx

  io.rx.enq <> rxfifo.io.enq
  io.rx.deq <> otherfifo.io.deq 

  io.tx.enq <> txfifo.io.enq
  io.tx.deq <> txfifo.io.deq
  io.cam <> camfifo.io.deq
}

case class CamDMAEngineConfig(base: BigInt)
case object CamDMAEngineKey extends Field[Option[CamDMAEngineConfig]](None)

class CamDMAEngine(implicit p: Parameters) extends LazyModule {
  //TODO: use my own TLhelper
  val node = TLHelper.makeClientNode(name = "cam-dma", sourceId = IdRange(0,1))
  lazy val module = new CamDMAEngineModuleImp(this)
}

class CamDMAEngineModuleImp(outer: CamDMAEngine) extends LazyModuleImp(outer){
  val config = p(RoseAdapterKey).get

  val io = IO(new Bundle{
    val fifo = Flipped(Decoupled(UInt(config.width.W)))
    val cam_buffer = Output(UInt(1.W))
    val counter_max = Input(UInt(config.width.W))
  })

  val (mem, edge) = outer.node.out(0)
  val addrBits = edge.bundle.addressBits
  val blockBytes = p(CacheBlockBytes)

  //TODO: allowing strided writes living in memory
  require(config.width <= blockBytes)

  val mIdle :: mWrite :: mResp :: Nil = Enum(3)
  val mstate = RegInit(mIdle)
  
  val addr = Reg(UInt(addrBits.W))
  val buffer_next = Wire(UInt(config.width.W))
  buffer_next := io.fifo.bits
  val buffer = RegEnable(next = buffer_next, enable = io.fifo.fire)
  //TODO: set this to something reasonable
  // val counter_max = RegInit(32.U(config.width.W))
  //TODO: add a size here
  val counter_enabled = Wire(Bool())
  // It does not like recursive definitions?
  val counter_next = Wire(UInt(config.width.W))
  val counter : UInt = RegEnable(next = counter_next, init = 0.U(config.width.W), enable = counter_enabled)
  counter_next := Mux((counter < io.counter_max + io.counter_max - 4.U), counter + 4.U, 0.U)

  io.cam_buffer := counter >= io.counter_max

  io.fifo.ready := mstate === mIdle
  mem.a.valid := mstate === mWrite
  mem.d.ready := mstate === mResp
  dontTouch(mem.d.valid)

  // putting the buffer data on the TL mem lane
  mem.a.bits := edge.Put(
  fromSource = 0.U,
  toAddress = addr,
  lgSize = log2Ceil(config.width).U - 3.U,
  data = buffer)._2

  //FIXME: Hardwired because i am crazy
  addr := config.base.U + counter

  counter_enabled := mem.a.fire && mstate === mWrite

  switch(mstate){
    is (mIdle){
      //grab data into the fifo
      mstate := Mux(io.fifo.fire, mWrite, mIdle)
    }
    is (mWrite){
      // edge.done refers to fully transmit a message possibly with multiple beats, while mem.d.fire refers to response to a single beat
      // edge.done means we need to load another block from FIFO to buffer
      mstate := Mux(mem.a.fire, mResp, mWrite)
    }
    is (mResp){
      mstate := Mux(mem.d.fire, mIdle, mResp)
    }
  }
}

//arbiter, connected between the camfifo, otherfifo, and rxfifo
class RoseAdapterArbiter(val w: Int) extends Module {

  val io = IO(new Bundle{
    //valid and bits are output, ready is input in DecoupledIO
    val cam = Decoupled(UInt(w.W))
    val other = Decoupled(UInt(w.W))
    //flipped for rx
    val rx = Flipped(Decoupled(UInt(w.W)))
  })

  //                        - cam.fifo
  // rx.fifo-----arbiter----
  //                        - other.fifo
  //the buffer of rx deque data
  val buffer_next = Wire(UInt(w.W))
  buffer_next := io.rx.bits
  //goal: io.rx.fire should be synced with io.target.fire
  val buffer = RegEnable(next = buffer_next, enable = io.rx.fire)
  //the counter of how many cycles of loads, decided by the second queue val
  val counter = Reg(UInt(w.W))
  val sIdle :: sFlag :: sHeader :: sCounter :: sLoad :: Nil = Enum(5)
  val state = RegInit(sIdle)

  //flag determining where the data goes
  val fOther :: fCam :: Nil = Enum(2)
  val flag = RegInit(false.B)

  // general enque logic, applicable to both cam&others
  val tx_val = Wire(Bool())
  io.cam.valid := Mux(flag, tx_val, 0.U)
  io.other.valid := Mux(~flag, tx_val, 0.U)

  io.cam.bits := buffer
  io.other.bits := buffer

  tx_val := 0.U
  io.rx.ready := Mux(flag, io.cam.ready, io.other.ready)

  switch(state) {
    // heat up the buffer with the header
    is(sIdle) {
      // what if the target fifo is not ready?
      // notice that flag is not set here yet
      when(io.cam.ready & io.other.ready) {
        io.rx.ready := 1.U
        state := Mux(io.rx.fire, sFlag, sIdle)
      } .otherwise {
        state := sIdle
      }
    }
    // header is already in the buffer, now set flag
    is (sFlag) {
      // make sure the buffer does not change here
      io.rx.ready := 0.U
      flag := (buffer === 0x11.U)
      state := sHeader
    }
    // send the header packet to the right fifo according to the flag
    is(sHeader) {
      // if it is a camera header, throw it away, else transmit it
      tx_val := Mux(flag, false.B, io.rx.valid)
      state := Mux(io.rx.fire, sCounter, sHeader)
    }
    // set the value of counter to the desired cycles
    // the buffer now holds the counter value. notice that counter will be set at the next cycle. 
    is(sCounter) {
      // TODO: verify this
      counter := buffer >> 2
      // counter := Mux(buffer === 0.U, 0.U, 1.U << ((Log2(buffer) - (log2Ceil(w) - log2Ceil(8)).U) - 1.U))
      tx_val := Mux(flag, false.B, io.rx.valid)
      // don't bother to step into sLoad and do nothing and exit
      when (buffer === 0.U){
        io.rx.ready := 0.U
        state := Mux(Mux(flag, true.B, io.other.fire), sIdle, sCounter)
      } .otherwise {
        state := Mux(io.rx.fire, sLoad, sCounter)
      }
    }
    is(sLoad){
      // TODO: verify this to be 1.U or 0.U
      when(counter > 1.U){
        tx_val := io.rx.valid
        counter := Mux(io.rx.fire, counter - 1.U, counter)
        state := sLoad
        //when counter is strictly 1, there's one last thing to send. Go back to Idle after this
      } .otherwise {
        io.rx.ready := 0.U
        tx_val := true.B
        state := Mux(Mux(flag, io.cam.fire, io.other.fire), sIdle, sLoad)
      }
    }
  }
}

//topmost wrapper, connects to the bridge and the SoC
trait RoseAdapterModule extends HasRegMap {
  // TODO: wrap this with IO(new Bundle) struct?
  val io: RoseAdapterTopIO

  implicit val p: Parameters
  def params: RoseAdapterParams
  val clock: Clock
  val reset: Reset

  // TL Register Wires
  val tx_data = Wire(new DecoupledIO(UInt(params.width.W)))
  val rx_data = Wire(new DecoupledIO(UInt(params.width.W)))
  val status = Wire(UInt(3.W))

  // FIXME: get the max value
  val written_counter_max = RegInit(99999.U(32.W))

  // Instantiate the internal RTL Module
  val impl = Module(new RoseAdapterMMIOChiselModule(params.width))
  //TODO: data to be connected 'Clock' must be hardware, not a bare Chisel type. Perhaps you forgot to wrap it in Wire(_) or IO(_)?
  impl.io.clock := clock
  impl.io.reset := reset.asBool

  impl.io.cam <> io.cam

  tx_data <> impl.io.tx.enq
  rx_data <> impl.io.rx.deq

  // Accessed in reverse order in C code
  status := Cat(impl.io.cam_buffer, impl.io.tx.enq.ready, impl.io.rx.deq.valid)

  // Connect to top IO
  io.enq <> impl.io.rx.enq
  io.deq <> impl.io.tx.deq
  io.cam_buffer <> impl.io.cam_buffer
  io.counter_max <> written_counter_max

  regmap(
    0x00 -> Seq(
      RegField.r(3, status)), // a read-only register capturing current status
    0x04 -> Seq(
      RegField.w(params.width, written_counter_max)), // write-only, max_counter is set on write
    0x08 -> Seq(
      RegField.w(params.width, tx_data)), // write-only, y.valid is set on write
    0x0C -> Seq(
      RegField.r(params.width, rx_data))) // read-only, RoseAdapter.ready is set on read
}

class RoseAdapterTL(params: RoseAdapterParams, beatBytes: Int)(implicit p: Parameters)
  extends TLRegisterRouter(
    params.address, "RoseAdapter", Seq("ucbbar,RoseAdapter"),
    beatBytes = beatBytes)(
      new TLRegBundle(params, _) with RoseAdapterTopIO)(
      new TLRegModule(params, _, _) with RoseAdapterModule)

class RoseAdapterAXI4(params: RoseAdapterParams, beatBytes: Int)(implicit p: Parameters)
  extends AXI4RegisterRouter(
    params.address,
    beatBytes=beatBytes)(
      new AXI4RegBundle(params, _) with RoseAdapterTopIO)(
      new AXI4RegModule(params, _, _) with RoseAdapterModule)

trait CanHavePeripheryRoseAdapter { this: BaseSubsystem =>

  private val portName = "RoseAdapter"

  val roseAdapter = p(RoseAdapterKey).map { 

    params =>

    val roseAdapterTL = LazyModule(new RoseAdapterTL(params, pbus.beatBytes)(p))
    pbus.toVariableWidthSlave(Some(portName)) { roseAdapterTL.node }

    val camDMAEngine = LazyModule(new CamDMAEngine()(p))
    // TODO: hard coded AF, bad practice, don't listen to me
    fbus.fromPort(Some("cam-dma"))() := TLWidthWidget(4) := camDMAEngine.node

    val outer_io = InModuleBody {
      val outer_io = IO(new ClockedIO(new RosePortIO)).suggestName(portName)

      outer_io.clock := roseAdapterTL.module.clock
      outer_io.bits.enq <> roseAdapterTL.module.io.enq
      outer_io.bits.deq <> roseAdapterTL.module.io.deq

      camDMAEngine.module.io.cam_buffer <> roseAdapterTL.module.io.cam_buffer
      camDMAEngine.module.io.counter_max <> roseAdapterTL.module.io.counter_max
      camDMAEngine.module.io.fifo <> roseAdapterTL.module.io.cam
      outer_io
    }
    outer_io
  }
}

trait CanHavePeripheryRoseAdapterModuleImp extends LazyModuleImp {
  val outer: CanHavePeripheryRoseAdapter
}

class WithRoseAdapter(useAXI4: Boolean, base: BigInt, width: Int) extends Config((site, here, up) => {
  case RoseAdapterKey => Some(RoseAdapterParams(useAXI4 = useAXI4, width = width, base = base))
})
