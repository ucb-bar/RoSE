package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IntParam, BaseModule}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes}
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import firrtl.annotations.{HasSerializationHints}

import firechip.bridgeinterfaces.{RoseAdapterKey, RoseAdapterParams, DstParams_Container, DstParams, CompleteDataflowConfig}

case object RoseAdapterKey extends Field[Option[RoseAdapterParams]](None)

class WithRoseAdapter(address: BigInt = 0x2000, width: Int = 32, dst_ports: DstParams_Container) extends Config((site, here, up) => {
  case RoseAdapterKey => Some(RoseAdapterParams(address, width, dst_ports))
})

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

// case class DstParams_Container (seq: Seq[DstParams]) 

// case class DstParams (
//   val port_type: String = "reqrsp", // supported are DMA and reqrsp
//   val DMA_address: BigInt = 0, // this attribute is only used if port_type is DMA
//   val name: String = "anonymous", // optional name for the port
//   val df_params: Seq[CompleteDataflowConfig] = Seq(), // optional dataflow parameters
//   val interrupt: Boolean = true // optional interrupt, only used if port_type is DMA
// ) extends HasSerializationHints{
//   def typeHints: Seq[Class[_]] = Seq(df_params.getClass()) ++ df_params.flatMap(_.typeHints)
// }