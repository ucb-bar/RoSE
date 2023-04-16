//These are the examples configs for the RoSE-supported SoC
// *** Not to be mistaken with the configurations for the RoSE adapters ***
package chipyard

import freechips.rocketchip.config.{Config}

class AbstractRoseConfig extends Config(
  new chipyard.iobinders.WithRoseIOPunchthrough ++
  new chipyard.config.AbstractConfig)

class AirSimIOTLFPGemminiLargeBoomConfig extends Config(
  new chipyard.example.WithAirSimIO(useAXI4=false) ++          // Usesim/src/main/cc/firesim/firesim_top.cc GCD Chisel, connect Tilelink
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new boom.common.WithNLargeBooms(1) ++                          // large boom config
  new chipyard.config.AbstractConfig)

class AirSimIOTLFPGemminiLargeDualBoomConfig extends Config(
  new chipyard.example.WithAirSimIO(useAXI4=false) ++          // Use GCD Chisel, connect Tilelink
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new boom.common.WithNLargeBooms(2) ++                          // large boom config
  new chipyard.config.AbstractConfig)

class AirSimIOTLFPGemminiRocketConfig extends Config(
  new chipyard.example.WithAirSimIO(useAXI4=false) ++          // Use GCD Chisel, connect Tilelink
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig)

class RoseTLRocketConfig extends Config(
  new rose.WithRoseAdapter(base=0x88000000L, width=32) ++          // Use GCD Chisel, connect Tilelink                      // use FP32Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.AbstractRoseConfig)


