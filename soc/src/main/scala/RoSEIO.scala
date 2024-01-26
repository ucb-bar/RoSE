package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.IO
import org.chipsalliance.cde.config.{Parameters, Field, Config}

// PortIO is used for bridge <--> SoC communication
class RosePortIO(params: RoseAdapterParams) extends Bundle {
  // SoC receive from bridge, a vector of flipped decoupled IOs, degraded from enq
  val rx = Vec(params.dst_ports.seq.size, Flipped(Decoupled(UInt(32.W))))
  // SoC send to bridge, simple for now, degraded from deq
  val tx = Decoupled(UInt(32.W))
}

class ConfigRoutingIO() extends Bundle {
  val config_routing_header = Input(UInt(32.W))
  val config_routing_valid = Input(Bool())
  val config_routing_channel = Input(UInt(32.W))
}

class RoseAdapterArbiterIO(params: RoseAdapterParams) extends Bundle {
    val rx = Vec(params.dst_ports.seq.size, Decoupled(UInt(32.W)))
    val tx = Flipped(Decoupled(UInt(32.W)))

    val budget = Flipped(Decoupled(UInt(32.W)))
    // advancing counter
    val cycleBudget = Input(UInt(32.W))
    // fixed step size
    val cycleStep = Input(UInt(32.W))

    val config_routing = new ConfigRoutingIO()
}

// Core IO of the adapter
class RoseAdapterIO(params: RoseAdapterParams) extends Bundle {
  val clock = Input(Clock())
  val reset = Input(Bool())
  // Bridge -> SoC
  val rx = new Bundle {
    val enq = Vec(params.dst_ports.seq.count(_.port_type != "DMA"), Flipped(Decoupled(UInt(32.W))))
    val deq = Vec(params.dst_ports.seq.count(_.port_type != "DMA"), Decoupled(UInt(32.W)))
  }
  // SoC -> Bridge
  val tx = new Bundle{
    val enq = Flipped(Decoupled(UInt(params.width.W)))
    val deq = Decoupled(UInt(params.width.W))
  }
  // Indicating which of the two buffers the image lies in
  val cam_buffer = Vec(params.dst_ports.seq.count(_.port_type == "DMA"),Input(UInt(1.W)))
}

// TopIO is used for Regmap communicating to the cam DMA engine & to the post-bridge, and the TL registers
// TopIO RoseadApterModule is the wrapper for the actuall MMIOChiselModule
class RoseAdapterTopIO(params: RoseAdapterParams) extends Bundle {
    // SoC receive from bridge, a vector of flipped decoupled IOs
    val rx = Vec(params.dst_ports.seq.size, Flipped(Decoupled(UInt(32.W))))
    // SoC send to bridge, simple for now
    val tx = Decoupled(UInt(32.W))
    val cam_buffer = Vec(params.dst_ports.seq.count(_.port_type == "DMA"), Input(UInt(1.W)))
    val counter_max = Vec(params.dst_ports.seq.count(_.port_type == "DMA"), Output(UInt(32.W)))
}