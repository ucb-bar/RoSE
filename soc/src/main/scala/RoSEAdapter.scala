package rose

import chisel3._
import chisel3.util._
// import testchipip._
import testchipip.util.{ClockedIO}
import chisel3.experimental.{IntParam, BaseModule}
import freechips.rocketchip.prci._
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes, FBUS, PBUS}
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf
import freechips.rocketchip.interrupts._
// import testchipip.TLHelper

import firechip.bridgeinterfaces.{RosePortIO, RoseAdapterKey, RoseAdapterParams, DstParams_Container, DstParams, CompleteDataflowConfig}

// buffers non-DMA data and tx data
class RoseAdapterMMIOChiselModule(params: RoseAdapterParams) extends Module
{
  val io = IO(new RoseAdapterIO(params))
  val txfifo = Module(new Queue(UInt(params.width.W), 64))

  io.tx.enq <> txfifo.io.enq
  io.tx.deq <> txfifo.io.deq

  for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) {
    val rx_buffer_fifo = Module(new Queue(UInt(params.width.W), 8)) 
    rx_buffer_fifo.io.enq <> io.rx.enq(i)
    rx_buffer_fifo.io.deq <> io.rx.deq(i)
  }
}

//topmost wrapper, connects to the bridge and the SoC
class RoseAdapterTL(params: RoseAdapterParams, beatBytes: Int)(implicit p: Parameters)
  extends ClockSinkDomain(ClockSinkParameters())(p){
  val device = new SimpleDevice("RoseAdapter", Seq("ucbbar,RoseAdapter")) 
  val node = TLRegisterNode(
    address = Seq(AddressSet(params.address, 0xFFF)),
    device = device,
    beatBytes = beatBytes)
  
  def nInterrupts = params.dst_ports.seq.count(_.port_type == "DMA")
  println(s"RoseAdapterTL nInterrupts detected: $nInterrupts")

  val intnode = IntSourceNode(IntSourcePortSimple(num = nInterrupts, resources = Seq(Resource(device, "int"))))

  // Externally, this helper should be used to connect the interrupts to a bus
  val intXing: IntOutwardClockCrossingHelper = this.crossOut(intnode)

  // Internally, this wire should be used to drive interrupt values
  override lazy val module = new RoseAdapterImpl
  class RoseAdapterImpl extends Impl {
    val io = IO(new RoseAdapterTopIO(params))

    withClockAndReset(clock, reset) {

      val impl = Module(new RoseAdapterMMIOChiselModule(params))

      // initialize to all ones
      val pending = RegInit(0.U(nInterrupts.W))

      val interrupts = Wire(Vec(nInterrupts, Bool()))
      interrupts := pending.asBools

      intnode.out(0)._1 := interrupts

      // MMIO Exposure to the SoC
      val tx_data = Wire(Decoupled(UInt(params.width.W)))
      val rx_data = Wire(Vec(params.dst_ports.seq.count(_.port_type != "DMA"), Decoupled(UInt(params.width.W))))
      val status = Wire(UInt((1 + params.dst_ports.seq.size).W))

      val written_counter_max = RegInit(VecInit(Seq.fill(params.dst_ports.seq.count(_.port_type == "DMA"))(0xFFFFFFFFL.U(32.W))))
      tx_data <> impl.io.tx.enq
      io.tx <> impl.io.tx.deq

      val cam_buffers = params.dst_ports.seq.filter(_.port_type == "DMA").zipWithIndex.map{ case (_, i) => io.cam_buffer(i) }
      val trigger = params.dst_ports.seq.filter(_.port_type == "DMA").zipWithIndex.map{ case (_, i) => io.interrupt_trigger(i) }
      val rx_valids = params.dst_ports.seq.filter(_.port_type != "DMA").zipWithIndex.map{ case (_, i) => impl.io.rx.deq(i).valid }

      for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) {
        rx_data(i) <> impl.io.rx.deq(i)
        io.rx(i) <> impl.io.rx.enq(i)
      }     

      status := Cat(cam_buffers ++ rx_valids ++ Seq(impl.io.tx.enq.ready))
      
      for (i <- 0 until params.dst_ports.seq.count(_.port_type == "DMA")) {
        io.cam_buffer(i) <> impl.io.cam_buffer(i)
        io.counter_max(i) <> written_counter_max(i)
        io.curr_counter(i) <> impl.io.curr_counter(i)
      }

      val triggers = Cat(trigger)

      val rx_datas =     
      (for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) yield {
          0x0C + i*4 -> Seq(
            RegField.r(params.width, rx_data(i), RegFieldDesc(s"rx_data_$i", "the decoupled interface of rx_queue"))) // read-only, RoseAdapter.ready is set on read
      }).toSeq

      val written_counters = 
      (for (i <- 0 until params.dst_ports.seq.count(_.port_type == "DMA")) yield {
          0x0C + (params.dst_ports.seq.count(_.port_type != "DMA") + i)*4 -> Seq(
            RegField.w(params.width, written_counter_max(i), RegFieldDesc(s"dma_${i}_written_counter", "the control register telling DMA when to wrap back"))) // write-only, y.valid is set on write
      }).toSeq

      val curr_counters = 
      (for (i <- 0 until params.dst_ports.seq.count(_.port_type == "DMA")) yield {
          0x0C + (params.dst_ports.seq.size + i)*4 -> Seq(
            RegField.r(params.width, impl.io.curr_counter(i), RegFieldDesc(s"dma_${i}_curr_counter", "the status register telling host where DMA is currently at"))) // read-only, RoseAdapter.ready is set on read
      }).toSeq

      val interrupts_regs = Seq(
      0x0C + (params.dst_ports.seq.size + params.dst_ports.seq.count(_.port_type == "DMA") + 1) * 4  -> Seq(
        RegField.w1ToClear(nInterrupts, pending, triggers, Some(RegFieldDesc(s"int", "the pending interrupts", reset=Some(0x0), volatile=true)))
      )) // write-1-to-clear, y.valid is set on write
      
      node.regmap(
        (Seq(
          0x00 -> Seq(
            RegField.r(1 + params.dst_ports.seq.size, status, RegFieldDesc("status", "the status register of the RoseAdapter"))),

          0x08 -> Seq(
            RegField.w(params.width, tx_data, RegFieldDesc("tx_data", "the decoupled inteface of tx_queue ")))) ++ // write-only, y.valid is set on write

          rx_datas ++ written_counters ++ curr_counters ++ interrupts_regs
        ): _*
      )
    }
  }
}

trait CanHavePeripheryRoseAdapter { this: BaseSubsystem =>
  val roseio = p(RoseAdapterKey) match { 
    case Some(params) => {
      // generate the lazymodule with regmap
      val fbus = locateTLBusWrapper(FBUS)
      val pbus = locateTLBusWrapper(PBUS)
      val roseAdapterTL = LazyModule(new RoseAdapterTL(params, pbus.beatBytes)(p))
      roseAdapterTL.clockNode := pbus.fixedClockNode
      pbus.coupleTo("RoseAdapter") { roseAdapterTL.node := TLFragmenter(pbus.beatBytes, pbus.blockBytes) := _ }

      // save all the DMA Engines for Inmodulebody use
      var DMA_lazymods = Seq[RoseDMA]()
      // generate all the DMA engines
      params.dst_ports.seq.foreach(
        i => i.port_type match {
          case "DMA" => {
            val roseDMA = LazyModule(new RoseDMA(i)(p))
            roseDMA.clockNode := fbus.fixedClockNode
            fbus.coupleFrom(f"cam-dma-$i") { _ := TLWidthWidget(4) := roseDMA.node}
            DMA_lazymods = DMA_lazymods :+ roseDMA
          }
          case _ => None
        }
      )
      
      val idx_map = params.genidxmap

      val rose_outer_io = InModuleBody {
        val outer_io = IO(new ClockedIO(new RosePortIO(params))).suggestName("RoseAdapter")
        dontTouch(outer_io)
        outer_io.clock := roseAdapterTL.module.clock
        outer_io.bits.tx <> roseAdapterTL.module.io.tx
        for (i <- 0 until params.dst_ports.seq.length) {
          params.dst_ports.seq(i).port_type match {
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
          DMA_lazymods(i).module.io.curr_counter <> roseAdapterTL.module.io.curr_counter(i)
          DMA_lazymods(i).module.io.interrupt_trigger <> roseAdapterTL.module.io.interrupt_trigger(i)
        }
        outer_io
      }

      ibus.fromSync := roseAdapterTL.intnode
      
      Some(rose_outer_io)
    }
    case None => None
  }
}