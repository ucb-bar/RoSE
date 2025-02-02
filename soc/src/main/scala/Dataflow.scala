package rose

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Parameters, Field}
import firrtl.annotations.HasSerializationHints
import freechips.rocketchip.rocket.PRV.U

import firechip.bridgeinterfaces.{BaseDataflowParameter, CompleteDataflowConfig, Dataflow}

// abstract class Dataflow(cfg: CompleteDataflowConfig) extends Module {
//   val io = IO(new QueueIO(UInt(cfg.userProvided.getChannelWidth.W), 32, false))
//   io.count := DontCare
// }

// abstract class BaseDataflowParameter(  
//   channelWidth: Int = 32,
// ) {
//   def elaborate: Dataflow = ??? // abstract unimplemented method
//   final def getChannelWidth: Int = channelWidth
// }

// case class CompleteDataflowConfig(userProvided: BaseDataflowParameter) extends HasSerializationHints {
//   def typeHints: Seq[Class[_]] = Seq(userProvided.getClass)
// }