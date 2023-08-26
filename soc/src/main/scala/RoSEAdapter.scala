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
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf
// import testchipip.TLHelper

// A utility read-only register look up table that maps the id to the corresponding dst_port index
class rolut(params: RoseAdapterParams) extends Module {
  val io = IO(new Bundle{
    val key = Input(UInt(params.width.W))
    val value = Output(UInt(params.width.W))
    val keep_header = Output(Bool())
  })
  dontTouch(io)
  io.value := 0.U
  io.keep_header := false.B
  for (i <- 0 until params.dst_ports.seq.length) {
    for (j <- 0 until params.dst_ports.seq(i).IDs.length) {
      when (io.key === params.dst_ports.seq(i).IDs(j).U) {
        io.value := i.U
        if (params.dst_ports.seq(i).port_type == "reqrsp") {
          io.keep_header := true.B
        } else {
          io.keep_header := false.B
        } 
      }
    }
  }
}

class RoseAdapterMMIOChiselModule(params: RoseAdapterParams) extends Module
{
  val io = IO(new RoseAdapterIO(params))
  dontTouch(io)
  val txfifo = Module(new Queue(UInt(params.width.W), 64))

  io.tx.enq <> txfifo.io.enq
  io.tx.deq <> txfifo.io.deq

  for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) {
    io.rx(i).ready := true.B
  }
}

class CamDMAEngine(param: DstParams)(implicit p: Parameters) extends LazyModule {
  val port_param = param
  val node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "init-zero", sourceId = IdRange(0, 1))))))
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
class RoseAdapterArbiter(params: RoseAdapterParams) extends Module {
  val w = params.width
  // val io = IO(new Bundle{
  //   //valid and bits are output, ready is input in DecoupledIO
  //   val cam = Decoupled(UInt(w.W))
  //   val other = Decoupled(UInt(w.W))
  //   //flipped for rx
  //   val rx = Flipped(Decoupled(UInt(w.W)))
  // })
  val io = IO(new RoseAdapterArbiterIO(params))
  //                        - cam.fifo
  // rx.fifo-----arbiter--- - something.fifo
  //                        - other.fifo
  //the buffer of rx deque data
  val buffer_next = Wire(UInt(w.W))
  buffer_next := io.tx.bits
  //goal: io.rx.fire should be synced with io.target.fire
  val rolut = Module(new rolut(params))
  val buffer = RegEnable(next = buffer_next, enable = io.tx.fire)
  val budget = RegEnable(next = io.budget.bits, enable = io.budget.fire)
  val keep_header = RegEnable(next=rolut.io.keep_header, enable = io.tx.fire)
  //the counter of how many cycles of loads, decided by the second queue val
  val counter = Reg(UInt(w.W))
  val sIdle :: sFlag :: sHeader :: sCounter :: sLoad :: Nil = Enum(5)
  val state = RegInit(sIdle)

  val cycle_reset_next = Wire(Bool())
  val cycle_reset_en = Wire(Bool())
  cycle_reset_next := Mux(state === sIdle, false.B, io.cycleBudget === 0.U)
  cycle_reset_en := state === sIdle || io.cycleBudget === 0.U
  val step_passed = RegEnable(next = cycle_reset_next, enable = cycle_reset_en)

  // create a map with id as key and the corresponding dst_port index as value
  // val id_map = scala.collection.mutable.Map[Chisel.UInt, Int]()
  // for (i <- 0 until params.dst_ports.seq.length) {
  //   for (j <- 0 until params.dst_ports.seq(i).IDs.length) {
  //     id_map += (params.dst_ports.seq(i).IDs(j).U -> i)
  //   }
  // }

  // create a sequence of vectors, with each vector representing one params.dst_ports.seq(i).IDs(j).U
  // var id_collection = Seq[Chisel.Vec]()
  // for (i <- 0 until params.dst_ports.seq.length) {
  //   // create an iterator of chisel UInts from dst_ports.seq
  //   val id_iter:Seq[Chisel.UInt] = for (j <- params.dst_ports.seq(i).IDs) yield j.U 
  //   // convert the iterator to a vector
  //   val id_vec = VecInit(id_iter)
  //   id_collection = id_collection :+ id_vec
  // }

  // general enque logic, applicable to both cam&others
  val tx_val = Wire(Bool())
  val idx = RegInit(0.U(32.W))
  // if idx equals i, get tx_val, otherwise get 0
  params.dst_ports.seq.zipWithIndex.foreach {
    case (dstParam, i) =>
      when (i.U === idx) {
        io.rx(i).valid := tx_val
      } .otherwise {
        io.rx(i).valid := false.B
      }
      io.rx(i).bits := buffer
  }

  // io.cam.valid := Mux(flag, tx_val, 0.U)
  // io.other.valid := Mux(~flag, tx_val, 0.U)

  tx_val := 0.U
  io.tx.ready := io.rx(idx).ready
  io.budget.ready := state === sFlag
  rolut.io.key := buffer
  switch(state) {
    // heat up the buffer with the header
    is(sIdle) {
      // FIXME: would disregard the reday signal here - make sure to sink all requests!
      io.tx.ready := 1.U
      state := Mux(io.tx.fire, sFlag, sIdle)
    }
    // header is already in the buffer, now set flag
    is (sFlag) {
      // make sure the buffer does not change here
      io.tx.ready := 0.U
      // look up the id_map to get the index
      idx := rolut.io.value
      // get idx out of idx_wrapper
      // idx = idx_wrapper.getOrElse(0) // FIXME: would default to 0 
      // flag := (buffer === 0x11.U)
      state := sHeader
    }
    // send the header packet to the right fifo according to the flag
    is(sHeader) {
      // if it is a camera header, throw it away, else transmit it
      tx_val := Mux(keep_header, io.tx.valid, false.B)
      val can_advance = step_passed || budget < io.cycleBudget
      io.tx.ready := can_advance && io.rx(idx).ready
      state := Mux(can_advance, Mux(io.tx.fire, sCounter, sHeader), sHeader)
    }
    // set the value of counter to the desired cycles
    // the buffer now holds the counter value. notice that counter will be set at the next cycle. 
    is(sCounter) {
      counter := buffer >> 2
      // counter := Mux(buffer === 0.U, 0.U, 1.U << ((Log2(buffer) - (log2Ceil(w) - log2Ceil(8)).U) - 1.U))
      tx_val := Mux(keep_header, io.tx.valid, false.B)
      // don't bother to step into sLoad and do nothing and exit
      when (buffer === 0.U){
        io.tx.ready := 0.U
        state := Mux(io.rx(idx).fire, sIdle, sCounter)
      } .otherwise {
        state := Mux(io.tx.fire, sLoad, sCounter)
      }
    }
    is(sLoad){
      when(counter > 1.U){
        tx_val := io.tx.valid
        counter := Mux(io.tx.fire, counter - 1.U, counter)
        state := sLoad
        //when counter is strictly 1, there's one last thing to send. Go back to Idle after this
      } .otherwise {
        io.tx.ready := 0.U
        tx_val := true.B
        state := Mux(io.rx(idx).fire, sIdle, sLoad)
      }
    }
  }
}

//topmost wrapper, connects to the bridge and the SoC
trait RoseAdapterModule extends HasRegMap {
  // TODO: wrap this with IO(new Bundle) struct?
  implicit val p: Parameters
  def params: RoseAdapterParams
  
  val clock: Clock
  val reset: Reset
  val io: RoseAdapterTopIO

  dontTouch(io)
  // TL Register Wires
  val tx_data = Wire(Decoupled(UInt(params.width.W)))
  val rx_data = Wire(Vec(params.dst_ports.seq.count(_.port_type != "DMA"), UInt(params.width.W)))
  val status = Wire(UInt((1 + params.dst_ports.seq.size).W))

  // FIXME: get the max value
  val written_counter_max = RegInit(VecInit(Seq.fill(params.dst_ports.seq.count(_.port_type == "DMA"))(99999.U(32.W))))

  // Instantiate the internal RTL Module
  val impl = Module(new RoseAdapterMMIOChiselModule(params))
  
  impl.io.clock := clock
  impl.io.reset := reset.asBool

  tx_data <> impl.io.tx.enq
  
  // Create a sequence of all DMA cam buffers
  var cam_buffers:Seq[UInt] = Seq()
  for (i <- 0 until params.dst_ports.seq.count(_.port_type == "DMA")) {
    cam_buffers = cam_buffers :+ impl.io.cam_buffer(i)
  }

  var idx_map = Seq[Int]()
  var reversed_idx_map = Seq[Int]()
  // generate all the DMA engines
  var DMA_count = 0
  var other_count = 0
  params.dst_ports.seq.zipWithIndex.foreach(
    {case (i, n) => i.port_type match {
        case "DMA" => {
          idx_map = idx_map :+ DMA_count
          DMA_count += 1
        }
        case _ => {
          idx_map = idx_map :+ other_count
          other_count += 1
          reversed_idx_map = reversed_idx_map :+ n
        }
      }
    }
  )
  // create a sequence of all rx valid signals
  var rx_valids:Seq[Bool] = Seq()
  for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) {
    rx_valids = rx_valids :+ impl.io.rx(i).valid
    rx_data(i) := impl.io.rx(i).bits
    io.rx(reversed_idx_map(i)) <> impl.io.rx(i)
  }

  // // Concat all cam buffers
  // val cam_buffers_cat = Cat(cam_buffers)
  // // Concat all rx valid signals
  // val rx_valids_cat = Cat(rx_valids)
  // Concat all rx valid signals and cam buffers, and tx ready
  val status_seq = cam_buffers ++ rx_valids ++ Seq(impl.io.tx.enq.ready)
  status := Cat(status_seq)
  // Connect to top IO
  // io.rx <> impl.io.rx
  io.tx <> impl.io.tx.deq

  for (i <- 0 until params.dst_ports.seq.count(_.port_type == "DMA")) {
    io.cam_buffer(i) <> impl.io.cam_buffer(i)
    io.counter_max(i) <> written_counter_max(i)
  }

  val rx_datas =     
  (for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) yield {
      0x0C + i*4 -> Seq(
        RegField.r(params.width, rx_data(i))) // read-only, RoseAdapter.ready is set on read
  }).toSeq


  val written_counters = 
  (for (i <- 0 until params.dst_ports.seq.count(_.port_type == "DMA")) yield {
      0x0C + (params.dst_ports.seq.count(_.port_type != "DMA") + i)*4 -> Seq(
        RegField.w(params.width, written_counter_max(i))) // read-only, RoseAdapter.ready is set on read
  }).toSeq

  // val statusreg = 0x00 -> Seq(
  //   // only support leq 30 ports
  //   RegField.r(1 + params.dst_ports.seq.size, status))

  // val tx_datareg = 0x08 -> Seq(
  //   RegField.w(params.width, tx_data)) // write-only, y.valid is set on write

  // regmap(statusreg, tx_datareg)
  // regmap((rx_datas): _*)
  // regmap((written_counters): _*)
  dontTouch(status)
  regmap(
    (Seq(
    0x00 -> Seq(
      // only support leq 30 ports
      RegField.r(1 + params.dst_ports.seq.size, status)),

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
    pbus.coupleTo(portName) { roseAdapterTL.node := TLFragmenter(pbus.beatBytes, pbus.blockBytes) := _ }

    // save all the DMA Engines for Inmodulebody use
    var DMA_lazymods = Seq[CamDMAEngine]()
    var idx_map = Seq[Int]()
    // generate all the DMA engines
    var DMA_count = 0
    var other_count = 0
    params.dst_ports.seq.foreach(
      i => i.port_type match {
        case "DMA" => {
          val camDMAEngine = LazyModule(new CamDMAEngine(i)(p))
          fbus.coupleFrom(f"cam-dma-$DMA_count") { _ := TLWidthWidget(4) := camDMAEngine.node}
          DMA_lazymods = DMA_lazymods :+ camDMAEngine
          idx_map = idx_map :+ DMA_count
          DMA_count += 1
        }
        case _ => {
          idx_map = idx_map :+ other_count
          other_count += 1
        }
      }
    )

    val outer_io = InModuleBody {
      val outer_io = IO(new ClockedIO(new RosePortIO(params))).suggestName(portName)
      dontTouch(outer_io)
      outer_io.clock := roseAdapterTL.module.clock
      outer_io.bits.tx <> roseAdapterTL.module.io.tx
      for (i <- 0 until params.dst_ports.seq.length) {
        outer_io.bits.rx(i) <> roseAdapterTL.module.io.rx(i)
        params.dst_ports.seq(i).port_type match {
          case "DMA" => {
            outer_io.bits.rx(i) <> DMA_lazymods(idx_map(i)).module.io.rx
          }
          case _ => {
            // outer_io.bits.rx(i) <> roseAdapterTL.module.io.rx(idx_map(i))
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
