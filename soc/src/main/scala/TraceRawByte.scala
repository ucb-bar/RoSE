package firechip.bridgeinterfaces

import chisel3._
import chisel3.util._

// Ported from riscv-tacit/chipyard @ fix-issue-unit. ADAPTED for our tacit lineage
// (fe61365 "modular"): our tacit encoder emits a SINGLE byte per beat
// (rocket-chip trace/TraceSink.scala: trace_in = Flipped(Decoupled(UInt(8.W)))),
// whereas the fork's pinned tacit (648943d) emitted a 4-lane TraceEgressInterface.
// So numLanes = 1 here; the SerialWidthAggregator in the bridge module handles the
// (now trivial) 1-byte-per-beat contiguous case unchanged.
object TraceEgressRawByteConstants {
  val numLanes = 1
}

class TraceEgressRawByteInterface extends Bundle {
  val bits = Output(Vec(TraceEgressRawByteConstants.numLanes, UInt(8.W)))
  val mask = Output(Vec(TraceEgressRawByteConstants.numLanes, Bool()))
  val valid = Output(Bool())
  val ready = Input(Bool())
  def fire = valid && ready
}

class TraceRawBytePortIO extends Bundle {
  val out = new TraceEgressRawByteInterface
}

class TraceRawByteBridgeTargetIO extends Bundle {
  val byte = Flipped(new TraceRawBytePortIO)
  val reset = Input(Bool())
  val clock = Input(Clock())
}

case class TraceRawByteKey() // intentionally empty
