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

import firechip.bridgeinterfaces.{RoseAdapterParams, DstParams_Container, DstParams, CompleteDataflowConfig}

// RoseAdapterKey is the CDE config key for the Rose adapter. It lives in the `rose`
// generator (which has CDE via rocketchip), NOT in firechip.bridgeinterfaces, which
// must stay pure-Chisel for cross-compilation to FireSim's GoldenGate compiler.
// RoseAdapterParams (the pure-Chisel param case class) is canonical in bridgeinterfaces.
case object RoseAdapterKey extends Field[Option[RoseAdapterParams]](None)

class WithRoseAdapter(address: BigInt = 0x2000, width: Int = 32, dst_ports: DstParams_Container) extends Config((site, here, up) => {
  case RoseAdapterKey => Some(RoseAdapterParams(address, width, dst_ports))
})

// Turns ON the host->FPGA bulk DMA datapath (ROSE_DMA_RX) deterministically, by
// flipping the dmaRx flag on whatever RoseAdapterParams the base config already
// set. Because dmaRx lives in RoseAdapterParams -- the bridge's serialized
// constructor arg -- this reaches the GoldenGate host-side bridge elaboration
// independent of the shell environment (the ROSE_DMA_RX env var does NOT survive
// FireSim's SSH-dispatched build). Compose it to the LEFT of a base RoSE config,
// e.g. `new WithRoseDmaRx ++ new RoseTLRocketSaturnMMIOOnlyConfig`.
class WithRoseDmaRx extends Config((site, here, up) => {
  case RoseAdapterKey => up(RoseAdapterKey).map(_.copy(dmaRx = true))
})

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