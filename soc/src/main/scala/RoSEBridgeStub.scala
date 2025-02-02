//See LICENSE for license details
package firechip.bridgestubs

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
// import rose.{RosePortIO, RoseAdapterParams, RoseAdapterKey, RoseAdapterArbiterIO, ConfigRoutingIOBundle, Dataflow}
import firrtl.annotations.HasSerializationHints
import java.io.{File, FileWriter}
import freechips.rocketchip.util.{AsyncQueue, AsyncQueueParams}

import firesim.lib.bridgeutils._

import firechip.bridgeinterfaces._

class RoseBridge()(implicit p: Parameters) extends BlackBox with Bridge[HostPortIO[RoseBridgeTargetIO]] {
  val moduleName = "firechip.goldengateimplementations.RoSEBridgeModule"

  println(s"Got an implicit paramter of {${p(RoseAdapterKey).get}}")
  val io = IO(new RoseBridgeTargetIO(p(RoseAdapterKey).get))

  val bridgeIO = HostPort(io)

  val constructorArg = Some(RoseKey(p(RoseAdapterKey).get))
  println(s"Got a constructor arg of {${constructorArg.get}}")

  // Finally, and this is critical, emit the Bridge Annotations -- without
  // this, this BlackBox would appear like any other BlackBox to Golden Gate
  generateAnnotations()
}

object RoseBridge {
  def apply(clock: Clock, airsimio: RosePortIO, reset: Bool)(implicit p: Parameters): RoseBridge = {
    val rosebridge = Module(new RoseBridge())
    rosebridge.io.airsimio <> airsimio
    rosebridge.io.clock := clock
    rosebridge.io.reset := reset
    rosebridge
  }
}
