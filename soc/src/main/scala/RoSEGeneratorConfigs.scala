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

class WithRoseAdapter(base: BigInt, width: Int) extends Config((site, here, up) => {
  case RoseAdapterKey => Some(RoseAdapterParams(width = width, base = base))
})
