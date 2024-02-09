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
  ))
  require(dst_ports.seq.size < 30)
) extends HasSerializationHints {
  def typeHints: Seq[Class[_]] = Seq(classOf[DstParams_Container])
}

case class DstParams_Container (seq: Seq[DstParams]) extends HasSerializationHints {
  def typeHints: Seq[Class[_]] = Seq(classOf[DstParams])
}

case class DstParams (
  val port_type: String = "reqrsp", // supported are DMA and reqrsp
  val DMA_address: BigInt = 0x88000000L, // this attribute is only used if port_type is DMA
)
