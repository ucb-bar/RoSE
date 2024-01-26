package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.prci._
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes}
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf
// import testchipip.TLHelper

// buffers non-DMA data and tx data
class RoseAdapterMMIOChiselModule(params: RoseAdapterParams) extends Module
{
  val io = IO(new RoseAdapterIO(params))
  dontTouch(io)
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
  extends ClockSinkDomain(ClockSinkParameters())(p) {
  val device = new SimpleDevice("RoseAdapter", Seq("ucbbar,RoseAdapter")) 
  val node = TLRegisterNode(
    address = Seq(AddressSet(params.address, 0xFFF)),
    device = device,
    beatBytes = beatBytes)
  
  override lazy val module = new RoseAdapterImpl
  class RoseAdapterImpl extends Impl {
    val io = IO(new RoseAdapterTopIO(params))

    withClockAndReset(clock, reset) {
      val impl = Module(new RoseAdapterMMIOChiselModule(params))
      
      // MMIO Exposure to the SoC
      val tx_data = Wire(Decoupled(UInt(params.width.W)))
      val rx_data = Wire(Vec(params.dst_ports.seq.count(_.port_type != "DMA"), Decoupled(UInt(params.width.W))))
      val status = Wire(UInt((1 + params.dst_ports.seq.size).W))

      val written_counter_max = RegInit(VecInit(Seq.fill(params.dst_ports.seq.count(_.port_type == "DMA"))(0xFFFFFFFFl.U(32.W))))
      tx_data <> impl.io.tx.enq
      io.tx <> impl.io.tx.deq

      val cam_buffers = params.dst_ports.seq.filter(_.port_type == "DMA").zipWithIndex.map{ case (_, i) => impl.io.cam_buffer(i) }
      val rx_valids = params.dst_ports.seq.filter(_.port_type != "DMA").zipWithIndex.map{ case (_, i) => impl.io.rx.deq(i).valid }
      
      // This is a mapping from dst_port index to non-DMA index
      var reversed_idx_map = Seq[Int]()
      params.dst_ports.seq.zipWithIndex.foreach(
        {case (port, n) => port.port_type match {
            case _ => reversed_idx_map = reversed_idx_map :+ n
          }
        }
      )

      for (i <- 0 until params.dst_ports.seq.count(_.port_type != "DMA")) {
        rx_data(i) <> impl.io.rx.deq(i)
        io.rx(reversed_idx_map(i)) <> impl.rx.enq(i)
      }     

      status := Cat(cam_buffers ++ rx_valids ++ Seq(impl.io.tx.enq.ready))
      
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

      node.regmap(
        (Seq(
          0x00 -> Seq(
            RegField.r(1 + params.dst_ports.seq.size, status)),

          0x08 -> Seq(
            RegField.w(params.width, tx_data))) ++ // write-only, y.valid is set on write

          rx_datas ++ written_counters
        ): _*
      )
    }
  }
}

trait CanHavePeripheryRoseAdapter { this: BaseSubsystem =>
  private val portName = "RoseAdapter"

  val roseAdapter = p(RoseAdapterKey).map { 
    params =>
    // generate the lazymodule with regmap
    val roseAdapterTL = LazyModule(new RoseAdapterTL(params, pbus.beatBytes)(p))
    roseAdapterTL.clockNode := pbus.fixedClockNode
    pbus.coupleTo(portName) { roseAdapterTL.node := TLFragmenter(pbus.beatBytes, pbus.blockBytes) := _ }

    // save all the DMA Engines for Inmodulebody use
    var DMA_lazymods = Seq[RoseDMA]()
    var idx_map = Seq[Int]()
    // generate all the DMA engines
    var DMA_count = 0
    var other_count = 0
    params.dst_ports.seq.foreach(
      i => i.port_type match {
        case "DMA" => {
          val roseDMA = LazyModule(new RoseDMA(i)(p))
          fbus.coupleFrom(f"cam-dma-$DMA_count") { _ := TLWidthWidget(4) := roseDMA.node}
          DMA_lazymods = DMA_lazymods :+ roseDMA
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
