// See LICENSE for license details

package firechip.goldengateimplementations

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters

import midas.widgets._
import firesim.lib.bridgeutils._

import firechip.bridgeinterfaces._

class SerialWidthAggregator(wideW: Int) extends Module {
  private val narrowW = 8
  private val numLanes = TraceEgressRawByteConstants.numLanes
  require(wideW > narrowW)
  require(wideW % narrowW == 0)
  val io = IO(new Bundle {
    val narrow = Flipped(new TraceEgressRawByteInterface)
    val wide   = Decoupled(UInt(wideW.W))
  })

  val beats = wideW / narrowW
  val max_entries = beats*2
  val ptrWidth = log2Ceil(max_entries)
  val counterWidth = log2Ceil(max_entries + 1)

  def wrapAdd(ptr: UInt, inc: UInt): UInt = {
    val sum = ptr +& inc
    Mux(sum >= max_entries.U, (sum - max_entries.U)(ptrWidth - 1, 0), sum(ptrWidth - 1, 0))
  }

  val staged_data = Reg(Vec(max_entries, UInt(narrowW.W))) // double buffered
  val enq_ptr = RegInit(0.U(ptrWidth.W))
  val deq_ptr = RegInit(0.U(1.W))
  val entries = RegInit(0.U(counterWidth.W))

  val enq_count = Mux(io.narrow.fire, PopCount(io.narrow.mask), 0.U)
  val do_enq = enq_count =/= 0.U

  val do_deq = io.wide.fire

  val maskPop = PopCount(io.narrow.mask)
  val expectedMask = Wire(UInt(numLanes.W))
  expectedMask := Mux(maskPop === 0.U, 0.U, (UIntToOH(maskPop) - 1.U)(numLanes - 1, 0))
  assert(!io.narrow.valid || io.narrow.mask.asUInt === expectedMask,
    "SerialWidthAggregator requires contiguous narrow.mask starting at lane 0")

  val wide0_data = Cat(staged_data.slice(0, beats).reverse)
  val wide1_data = Cat(staged_data.slice(beats, max_entries).reverse)
  io.wide.valid := entries >= beats.U // at least one wide data is available
  io.wide.bits := Mux(deq_ptr(0), wide1_data, wide0_data)

  val entries_next = entries +& enq_count -& do_deq.asUInt * beats.U
  assert(entries <= max_entries.U, "SerialWidthAggregator entries exceeded capacity")
  assert(entries_next <= max_entries.U, "SerialWidthAggregator next entries exceeded capacity")
  assert(!do_deq || entries >= beats.U, "SerialWidthAggregator underflow on dequeue")
  entries := entries_next

  when (do_enq) {
    for (i <- 0 until numLanes) {
      when (io.narrow.mask(i)) {
        staged_data(wrapAdd(enq_ptr, i.U)) := io.narrow.bits(i)
      }
    }
    enq_ptr := wrapAdd(enq_ptr, enq_count)
  }

  when (do_deq) {
    deq_ptr := deq_ptr + 1.U // allow overflow
  }

  io.narrow.ready := entries < (max_entries-numLanes).U
}

class TacitBridgeModule(key: TraceRawByteKey)(implicit p: Parameters) 
  extends BridgeModule[HostPortIO[TraceRawByteBridgeTargetIO]]()(p) 
    with StreamToHostCPU {

  val toHostCPUQueueDepth = 6144
  lazy val module = new BridgeModuleImp(this) {
    val io = IO(new WidgetIO())

    val hPort = IO(HostPort(new TraceRawByteBridgeTargetIO))

    val aggregator = Module(new SerialWidthAggregator(BridgeStreamConstants.streamWidthBits))
    val txfifo = Module(new Queue(UInt(BridgeStreamConstants.streamWidthBits.W), 3072))

    val target = hPort.hBits.byte

    val fire = hPort.toHost.hValid &&
               hPort.fromHost.hReady &&
               txfifo.io.enq.ready

    val targetReset = fire & hPort.hBits.reset
    txfifo.reset := reset.asBool || targetReset

    hPort.toHost.hReady := fire
    hPort.fromHost.hValid := fire

    aggregator.io.narrow.bits := target.out.bits
    aggregator.io.narrow.mask := target.out.mask
    aggregator.io.narrow.valid := target.out.valid && fire
    target.out.ready := aggregator.io.narrow.ready && fire

    txfifo.io.enq <> aggregator.io.wide
    streamEnq <> txfifo.io.deq

    genCRFile()
    
    override def genHeader(base: BigInt, memoryRegions: Map[String, BigInt], sb: StringBuilder): Unit = {
      genConstructor(
          base, 
          sb, 
          "tacit_t", 
          "tacit", Seq(
            UInt32(toHostStreamIdx),
            UInt32(toHostCPUQueueDepth),
          ),
          hasStreams = true)
    }
  }
}