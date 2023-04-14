//    ____       U  ___ u   ____     U _____ u 
// U |  _"\ u     \/"_ \/  / __"| u  \| ___"|/ 
//  \| |_) |/     | | | | <\___ \/    |  _|"   
//   |  _ <   .-,_| |_| |  u___) |    | |___   
//   |_| \_\   \_)-\___/   |____/>>   |_____|  
//   //   \\_       \\      )(  (__)  <<   >>  
//  (__)  (__)     (__)    (__)      (__) (__) 
// --- --- Get some RoSE IOs rolling --- --- ---

package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.config.{Parameters, Field, Config}

// PortIO is used for top level ports enq and deq
class RosePortIO extends Bundle {
  val enq = Flipped(Decoupled(UInt(32.W)))
  val deq = Decoupled(UInt(32.W))
}

// Core IO of the adapter
class RoseAdapterIO(val w: Int) extends Bundle {
  val clock = Input(Clock())
  val reset = Input(Bool())
  // Bridge -> SoC
  val rx = new Bundle{
    val enq = Flipped(Decoupled(UInt(w.W)))
    val deq = Decoupled(UInt(w.W))
  }
  // SoC -> Bridge
  val tx = new Bundle{
    val enq = Flipped(Decoupled(UInt(w.W)))
    val deq = Decoupled(UInt(w.W))
  }
  // This to dequeue the cam FIFO
  val cam = Decoupled(UInt(w.W))
  // Indicating which of the two buffers the image lies in
  val cam_buffer = Input(UInt(1.W))
}

// TopIO is used for RoseAdapterTL communicating to the cam DMA engine, and the TL registers
trait RoseAdapterTopIO extends Bundle {
    val enq = Flipped(Decoupled(UInt(32.W)))
    val deq = Decoupled(UInt(32.W))
    val cam = Decoupled(UInt(32.W))
    val cam_buffer = Input(UInt(1.W))
    val counter_max = Output(UInt(32.W))
}

trait HasRoseAdapterIO extends BaseModule {
  val w: Int
  val io = IO(new RoseAdapterIO(w))
}

trait HasRosePortIO extends BaseModule {
  val port = IO(new RosePortIO())
}
