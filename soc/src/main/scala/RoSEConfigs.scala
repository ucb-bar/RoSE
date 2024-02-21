//These are the examples configs for the RoSE-supported SoC
// *** Not to be mistaken with the configurations for the RoSE adapters ***
package chipyard.config

import org.chipsalliance.cde.config.{Config}
import rose._

class AbstractRoseConfig extends Config(
  new chipyard.iobinders.WithRoseIOPunchthrough ++
  new chipyard.config.AbstractConfig)

class RoseTLRocketConfig extends Config(
  new rose.WithRoseAdapter() ++          // Use GCD Chisel, connect Tilelink                      // use FP32Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractRoseConfig)

class RoseTLBOOMConfig extends Config(
  new rose.WithRoseAdapter() ++          // Use GCD Chisel, connect Tilelink                      // use FP32Gemmini systolic array GEMM accelerator
  new boom.common.WithNLargeBooms(1) ++   
  new chipyard.config.AbstractRoseConfig)

class RoseTLBOOMGemminiConfig extends Config(
  new rose.WithRoseAdapter() ++          // Use GCD Chisel, connect Tilelink                      // use FP32Gemmini systolic array GEMM accelerator
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new boom.common.WithNLargeBooms(1) ++   
  new chipyard.config.AbstractRoseConfig)

