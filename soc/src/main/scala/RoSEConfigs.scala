//These are the examples configs for the RoSE-supported SoC
// *** Not to be mistaken with the configurations for the RoSE adapters ***
package chipyard.config

import org.chipsalliance.cde.config.{Config}
import rose._
import stereoacc._
import javax.xml.crypto.Data

class AbstractRoseConfig extends Config(
  new chipyard.iobinders.WithRoseIOPunchthrough ++
  new chipyard.config.AbstractConfig)

class RoseTLRocketConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0"),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++        
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractRoseConfig)

class RoseTLRocketStereoAccConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0", 
      df_params = 
        Seq(CompleteDataflowConfig(StereoAccParams()))
      ),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++        
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractRoseConfig)

class RoseTLRocketEdgeDetAccConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0", 
      df_params = 
        Seq(CompleteDataflowConfig(EdgeDetAccParams()))
      ),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++        
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractRoseConfig)

class RoseTLBOOMConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0"),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++                         
  new boom.common.WithNLargeBooms(1) ++   
  new chipyard.config.AbstractRoseConfig)

class RoseTLBOOMGemminiConfig extends Config(
  new rose.WithRoseAdapter(dst_ports = new DstParams_Container(Seq(
    DstParams(port_type="DMA", DMA_address = 0x88000000L, name="DMA0"),
    DstParams(port_type="reqrsp", name="reqrsp0"),
    DstParams(port_type="reqrsp", name="reqrsp1"),
  ))) ++                          
  new gemmini.GemminiFP32DefaultConfig ++                         // use FP32Gemmini systolic array GEMM accelerator
  new boom.common.WithNLargeBooms(1) ++   
  new chipyard.config.AbstractRoseConfig)

