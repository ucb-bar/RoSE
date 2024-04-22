package rose

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Parameters, Field}

abstract class Dataflow(param: DataflowParameter) extends Module {
  final val io = IO(new Bundle {
    val enq = Decoupled(UInt(param.channleWidth.W))
    val deq = Decoupled(UInt(param.channleWidth.W))
    // for testing
    if (param.hasFinished){
      val finished = Output(Bool())
    }
  })
}

