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

class RoseAdapterMMIOChiselModule(params: RoseAdapterParams) extends Module
{
  val io = IO(new RoseAdapterIO(params))
  val txfifo = Module(new Queue(UInt(params.width.W), 64))

  for (i <- 0 until params.dst_ports.size) {
    if (params.dst_ports(i).port_type != "DMA") {
      io.rx(i) <> txfifo.io.enq
    }
  } 

  io.tx.enq <> txfifo.io.enq
  io.tx.deq <> txfifo.io.deq

}

class CamDMAEngine(param: DstParams)(implicit p: Parameters) extends LazyModule {
  //TODO: use my own TLhelper
  val port_param = param
  val node = TLHelper.makeClientNode(name = "cam-dma", sourceId = IdRange(0,1))
  lazy val module = new CamDMAEngineModuleImp(this)
}

class CamDMAEngineModuleImp(outer: CamDMAEngine) extends LazyModuleImp(outer){
  val config = p(RoseAdapterKey).get

  val io = IO(new Bundle{
    val rx = Flipped(Decoupled(UInt(config.width.W)))
    val cam_buffer = Output(UInt(1.W))
    val counter_max = Input(UInt(config.width.W))
  })

  // TODO: find a suitable depth
  val fifo = Module(new Queue(UInt(config.width.W), 32))
  fifo.io.enq <> io.rx

  val (mem, edge) = outer.node.out(0)
  val addrBits = edge.bundle.addressBits
  val blockBytes = p(CacheBlockBytes)

  //TODO: allowing strided writes living in memory
  require(config.width <= blockBytes)

  val mIdle :: mWrite :: mResp :: Nil = Enum(3)
  val mstate = RegInit(mIdle)

  val addr = Reg(UInt(addrBits.W))
  val buffer_next = Wire(UInt(config.width.W))
  buffer_next := fifo.io.deq.bits
  val buffer = RegEnable(next = buffer_next, enable = fifo.io.deq.fire)
  //TODO: set this to something reasonable
  // val counter_max = RegInit(32.U(config.width.W))
  //TODO: add a size here
  val counter_enabled = Wire(Bool())
  // It does not like recursive definitions?
  val counter_next = Wire(UInt(config.width.W))
  val counter : UInt = RegEnable(next = counter_next, init = 0.U(config.width.W), enable = counter_enabled)
  counter_next := Mux((counter < io.counter_max + io.counter_max - 4.U), counter + 4.U, 0.U)

  io.cam_buffer := counter >= io.counter_max
  fifo.io.deq.ready := mstate === mIdle
  mem.a.valid := mstate === mWrite
  mem.d.ready := mstate === mResp
  dontTouch(mem.d.valid)

  // putting the buffer data on the TL mem lane
  mem.a.bits := edge.Put(
  fromSource = 0.U,
  toAddress = addr,
  lgSize = log2Ceil(config.width).U - 3.U,
  data = buffer)._2

  addr := outer.port_param.DMA_address.U + counter

  counter_enabled := mem.a.fire && mstate === mWrite

  switch(mstate){
    is (mIdle){
      //grab data into the fifo
      mstate := Mux(fifo.io.deq.fire, mWrite, mIdle)
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
  val rx_data = Vec(params.dst_ports.count(_.port_type != "DMA"), Wire(new DecoupledIO(UInt(params.width.W))))
  val status = Wire(UInt((2 + params.dst_ports.size).W))

  // FIXME: get the max value
  val written_counter_max = Vec(params.dst_ports.count(_.port_type == "DMA"),RegInit(99999.U(32.W)))

  // Instantiate the internal RTL Module
  val impl = Module(new RoseAdapterMMIOChiselModule(params))
  
  impl.io.clock := clock
  impl.io.reset := reset.asBool

  tx_data <> impl.io.tx.enq

  // Create a sequence of all DMA cam buffers
  val cam_buffers:Seq[UInt] = Seq()
  for (i <- 0 until params.dst_ports.count(_.port_type == "DMA")) {
    cam_buffers :+ impl.io.cam_buffer(i)
  }

  // create a sequence of all rx valid signals
  val rx_valids:Seq[Bool] = Seq()
  for (i <- 0 until params.dst_ports.count(_.port_type != "DMA")) {
    rx_valids :+ impl.io.rx(i).valid
  }

  // Concat all cam buffers
  val cam_buffers_cat = Cat(cam_buffers)
  // Concat all rx valid signals
  val rx_valids_cat = Cat(rx_valids)
  // Concat all rx valid signals and cam buffers, and tx ready
  status := Cat(cam_buffers_cat, rx_valids_cat, impl.io.tx.enq.ready)
  //
  // Connect to top IO
  for (i <- 0 until params.dst_ports.count(_.port_type != "DMA")) {
    io.rx(i) <> impl.io.rx(i)
    rx_data(i) <> impl.io.rx(i).deq
  }
  io.tx <> impl.io.tx.deq

  for (i <- 0 until params.dst_ports.count(_.port_type == "DMA")) {
    io.cam_buffer(i) <> impl.io.cam_buffer(i)
    io.counter_max(i) <> written_counter_max(i)
  }

  val rx_datas =     
  for (i <- 0 until params.dst_ports.count(_.port_type != "DMA")) yield {
      0x0C + i*4 -> Seq(
        RegField.r(params.width, rx_data(i))) // read-only, RoseAdapter.ready is set on read
  }

  val written_counters = 
  for (i <- 0 until params.dst_ports.count(_.port_type == "DMA")) yield {
      0x0C + (params.dst_ports.count(_.port_type != "DMA") + i)*4 -> Seq(
        RegField.r(params.width, written_counter_max(i))) // read-only, RoseAdapter.ready is set on read
  }

  regmap(
    (Seq(
    0x00 -> Seq(
      // only support leq 30 ports
      RegField.r(2 + params.dst_ports.size, status)),

    0x08 -> Seq(
      RegField.w(params.width, tx_data))) ++ // write-only, y.valid is set on write

    // create all rx_data output MMIO regs
    // create all cam_buffer output MMIO regs
    rx_datas ++ written_counters): _*
  )
}

class RoseAdapterTL(params: RoseAdapterParams, beatBytes: Int)(implicit p: Parameters)
  extends TLRegisterRouter(
    params.address, "RoseAdapter", Seq("ucbbar,RoseAdapter"),
    beatBytes = beatBytes)(
      new TLRegBundle(params, _) with RoseAdapterTopIO)(
      new TLRegModule(params, _, _) with RoseAdapterModule)

trait CanHavePeripheryRoseAdapter { this: BaseSubsystem =>

  private val portName = "RoseAdapter"

  val roseAdapter = p(RoseAdapterKey).map { 

    params =>
    // generate the lazymodule with regmap
    val roseAdapterTL = LazyModule(new RoseAdapterTL(params, pbus.beatBytes)(p))
    pbus.toVariableWidthSlave(Some(portName)) { roseAdapterTL.node }

    // save all the DMA Engines for Inmodulebody use
    val DMA_lazymods = Seq[CamDMAEngine]()
    val idx_map = Seq[Int]()
    // generate all the DMA engines
    var DMA_count = 0
    var other_count = 0
    params.dst_ports.foreach(
      i => i.port_type match {
        case "DMA" => {
          val camDMAEngine = LazyModule(new CamDMAEngine(i)(p))
          fbus.fromPort(Some(f"cam-dma-$DMA_count"))() := TLWidthWidget(4) := camDMAEngine.node
          DMA_lazymods :+ camDMAEngine
          idx_map :+ DMA_count
          DMA_count += 1
        }
        case _ => {
          other_count += 1
          idx_map :+ other_count
        }
      }
    )

    val outer_io = InModuleBody {
      val outer_io = IO(new ClockedIO(new RosePortIO(params))).suggestName(portName)

      outer_io.clock := roseAdapterTL.module.clock

      for (i <- 0 until params.dst_ports.length) {
        params.dst_ports(i).port_type match {
          case "DMA" => {
            outer_io.bits.rx(i) <> DMA_lazymods(idx_map(i)).module.io.rx
          }
          case _ => {
            outer_io.bits.rx(i) <> roseAdapterTL.module.io.rx(idx_map(i))
          }
        }
      }

      // connect the DMA engines
      for (i <- 0 until DMA_lazymods.length) {
        DMA_lazymods(i).module.io.cam_buffer <> roseAdapterTL.module.io.cam_buffer(i)
        DMA_lazymods(i).module.io.counter_max <> roseAdapterTL.module.io.counter_max(i)
      }
      outer_io
    }
    outer_io
  }
}

trait CanHavePeripheryRoseAdapterModuleImp extends LazyModuleImp {
  val outer: CanHavePeripheryRoseAdapter
}
