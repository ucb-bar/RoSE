package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes}
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import firrtl.annotations.{HasSerializationHints}

case object RoseAdapterKey extends Field[Option[RoseAdapterParams]](None)

class WithRoseAdapter(
  address: BigInt = 0x2000,
  width: Int = 32,
  dst_ports: CParam_Container, 
) extends Config((site, here, up) => {
  case RoseAdapterKey => Some(RoseAdapterParams(
    address = address,
    width = width,
    dst_ports = dst_ports
  ))
})

// Parameter used accross the project
case class RoseAdapterParams(
  // This is the base address of the adapter TL Registers
  address: BigInt = 0x2000,
  // This is the width of the queues
  width: Int = 32,
  // Sequence of Destination ports
  dst_ports: CParam_Container){
  
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

case class CParam_Container (seq: Seq[CParam])

case class CParam (
  val port_type: String = "reqrsp", // supported are DMA and reqrsp
  val DMA_address: BigInt = 0x88000000L, // this attribute is only used if port_type is DMA
  val name: String = "anonymous", // optional name for the port
  val df_keys: Option[Config] = None // optional configuration for the dataflow accelerators
) extends HasSerializationHints {
  def typeHints: Seq[Class[_]] = Seq(classOf[Config])
}

// *** some example CParam ***
class SingleChannelReqRspParam extends CParam_Container(Seq(
  CParam(port_type = "reqrsp", name = "reqrsp0"),
))

class SingleChannelDMAParam extends CParam_Container(Seq(
  CParam(port_type = "DMA", DMA_address = 0x88000000L, name = "DMA0")
))

class DualChannelReqRspParam extends CParam_Container(Seq(
  CParam(port_type = "reqrsp", name = "reqrsp0"),
  CParam(port_type = "reqrsp", name = "reqrsp1")
))

class TripleChannelMixedParam extends CParam_Container(Seq(
  CParam(port_type = "DMA", DMA_address = 0x88000000L, name = "DMA0"),
  CParam(port_type = "reqrsp", name = "reqrsp0"),
  CParam(port_type = "reqrsp", name = "reqrsp1")
))