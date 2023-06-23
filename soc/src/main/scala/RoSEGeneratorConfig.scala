// This is actually configurable the rose adapter

package rose


import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes}
import freechips.rocketchip.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._

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
  dst_ports: Seq[DstParams] = Seq(
    DstParams(port_type = "reqrsp", IDs = Seq(0x00), latency = 0, bandwidth = 32),
    DstParams(port_type = "DMA", IDs = Seq(0x01), DMA_address = 0x88000000L, latency = 0, bandwidth = 32)
  )
  // require none of the dst params ID overlap
  // require(RoseAdapterParams().dst_ports.map(_.IDs).flatten.distinct.size == RoseAdapterParams().dst_ports.map(_.IDs).flatten.size)
  // require less than 30 dst ports
  // require(RoseAdapterParams().dst_ports.size < 30)
)



case class DstParams (
  val port_type: String = "reqrsp", // supported are stream, decoupled, interrupt, and DMA
  val IDs: Seq[Byte] = Seq(0x00), // sequence of ID bytes, must be non-overlapping
  val DMA_address: BigInt = 0x88000000L, // this is only used if port_type is DMA
  val latency: Int = 0,
  val bandwidth: Int = 32,
  //re-iteration of width for convenience, do not modify
  val width: Int = 32
)
