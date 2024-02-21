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

class WithRoseAdapter() extends Config((site, here, up) => {
  case RoseAdapterKey => Some(RoseAdapterParams())
})

// Parameter used accross the project
case class RoseAdapterParams(
  // This is the base address of the adapter TL Registers
  address: BigInt = 0x2000,
  // This is the width of the queues
  width: Int = 32,
  // Sequence of Destination ports
  dst_ports: DstParams_Container = DstParams_Container(Seq(
    DstParams(port_type = "DMA", DMA_address = 0x88000000L),
    DstParams(port_type = "reqrsp"),
    DstParams(port_type = "reqrsp")
  ))) extends HasSerializationHints {
  require(dst_ports.seq.size < 30)
  def typeHints: Seq[Class[_]] = Seq(classOf[DstParams_Container])
  
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

case class DstParams_Container (seq: Seq[DstParams]) extends HasSerializationHints {
  def typeHints: Seq[Class[_]] = Seq(classOf[DstParams])
}

case class DstParams (
  val port_type: String = "reqrsp", // supported are DMA and reqrsp
  val DMA_address: BigInt = 0x88000000L, // this attribute is only used if port_type is DMA
  val name: String = "anonymous" // optional name for the port
)
