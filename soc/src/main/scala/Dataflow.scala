package rose

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Parameters, Field}

abstract class Dataflow()(implicit val p: Parameters) extends Module {
  val io = IO(new Bundle {
    val enq = Decoupled(UInt(32.W))
    val deq = Decoupled(UInt(32.W))
    // for testing
    val finished = Output(Bool())
  })
}

// Dataflow construction
case object BuildDataflow extends Field[Seq[Parameters => Dataflow]](Nil)