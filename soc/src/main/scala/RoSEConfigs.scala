package chipyard

import freechips.rocketchip.config.{Config}

// ---------------------
// BOOM Configs
// ---------------------
class AirSimIOTLFPGemminiLargeBoomConfig extends Config(
  new chipyard.example.WithAirSimIO(useAXI4=false) ++          // Usesim/src/main/cc/firesim/firesim_top.cc GCD Chisel, connect Tilelink
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new boom.common.WithNLargeBooms(1) ++                          // large boom config
  new chipyard.config.AbstractConfig)

class AirSimIOTLFPGemminiRocketConfig extends Config(
  new chipyard.example.WithAirSimIO(useAXI4=false) ++          // Use GCD Chisel, connect Tilelink
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig)