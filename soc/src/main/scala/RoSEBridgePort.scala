//See LICENSE for license details
package firechip.bridgeinterfaces

import chisel3._
import chisel3.util._
// import org.chipsalliance.cde.config.Parameters
// import rose.{RosePortIO, RoseAdapterParams, RoseAdapterKey, RoseAdapterArbiterIO, ConfigRoutingIOBundle, Dataflow}
import firrtl.annotations.HasSerializationHints
import java.io.{File, FileWriter}

////////////////////////////////
import org.chipsalliance.cde.config.{Field}

class RosePortIO(params: RoseAdapterParams) extends Bundle {
  // SoC receive from bridge, a vector of flipped decoupled IOs, degraded from enq
  val rx = Vec(params.dst_ports.seq.size, Flipped(Decoupled(UInt(32.W))))
  // SoC send to bridge, simple for now, degraded from deq
  val tx = Decoupled(UInt(32.W))
}

case object RoseAdapterKey extends Field[Option[RoseAdapterParams]](None)

case class DstParams_Container (seq: Seq[DstParams]) 

case class DstParams (
  val port_type: String = "reqrsp", // supported are DMA and reqrsp
  val DMA_address: BigInt = 0, // this attribute is only used if port_type is DMA
  val name: String = "anonymous", // optional name for the port
  val df_params: Seq[CompleteDataflowConfig] = Seq(), // optional dataflow parameters
  val interrupt: Boolean = true // optional interrupt, only used if port_type is DMA
) extends HasSerializationHints{
  def typeHints: Seq[Class[_]] = Seq(df_params.getClass()) ++ df_params.flatMap(_.typeHints)
}

abstract class BaseDataflowParameter(  
  channelWidth: Int = 32,
) {
  def elaborate: Dataflow = ??? // abstract unimplemented method
  final def getChannelWidth: Int = channelWidth
}

case class CompleteDataflowConfig(userProvided: BaseDataflowParameter) extends HasSerializationHints {
  def typeHints: Seq[Class[_]] = Seq(userProvided.getClass)
}

abstract class Dataflow(cfg: CompleteDataflowConfig) extends Module {
  val io = IO(new QueueIO(UInt(cfg.userProvided.getChannelWidth.W), 32, false))
  io.count := DontCare
}


// Parameter used accross the project
case class RoseAdapterParams(
  // This is the base address of the adapter TL Registers
  address: BigInt,
  // This is the width of the queues
  width: Int,
  // Sequence of Destination ports
  dst_ports: DstParams_Container) extends HasSerializationHints{
  require(dst_ports.seq.size < 30)
  def typeHints: Seq[Class[_]] = Seq(dst_ports.getClass()) ++ dst_ports.seq.flatMap(_.typeHints)
  def genidxmap: Seq[Int] = {
    var idx_map = Seq[Int]()
    var other_idx: Int = 0
    var dma_idx: Int = 0
    dst_ports.seq.zipWithIndex.foreach(
      {case (port, n) => port.port_type match {
          case "DMA" => {
            idx_map = idx_map :+ dma_idx
            dma_idx = dma_idx + 1
          }
          case _ => {
            idx_map = idx_map :+ other_idx
            other_idx = other_idx + 1
          }
        }
      }
    )
    idx_map
  }
} 

class ConfigRoutingIOBundle(params: RoseAdapterParams) extends Bundle {
  val header = UInt(32.W)
  val channel = UInt(log2Ceil(params.dst_ports.seq.size).W)
}

class RoseAdapterArbiterIO(params: RoseAdapterParams) extends Bundle {
    val rx = Vec(params.dst_ports.seq.size, Decoupled(UInt(32.W)))
    val tx = Flipped(Decoupled(UInt(32.W)))

    val bigstep = Flipped(Decoupled(UInt(32.W)))
    val budget = Flipped(Decoupled(UInt(32.W)))
    // advancing counter
    val cycleBudget = Input(UInt(32.W))
    val currstep = Input(UInt(32.W))
    // fixed step size
    val cycleStep = Input(UInt(32.W))

    val config_routing = Flipped(Decoupled(new ConfigRoutingIOBundle(params)))


    // Debug singals
    val debug = new Bundle {
      val counter_state_sheader = Output(UInt(32.W))
      val counter_budget_fired = Output(UInt(32.W))
      val counter_tx_fired = Output(UInt(32.W))
      val counter_rx_0_fired = Output(UInt(32.W))
      val counter_rx_1_fired = Output(UInt(32.W))
    }
}

////////////////////////////////


class RoseBridgeTargetIO(params: RoseAdapterParams) extends Bundle {
  val clock = Input(Clock())
  val airsimio = Flipped(new RosePortIO(params))
  val reset = Input(Bool())
  // Note this reset is optional and used only to reset target-state modelled
  // in the bridge This reset just like any other Bool included in your target
  // interface, simply appears as another Bool in the input token.
}

case class RoseKey(roseparams: RoseAdapterParams) extends HasSerializationHints {
  def typeHints = Seq(classOf[RoseAdapterParams]) ++ roseparams.typeHints
}

