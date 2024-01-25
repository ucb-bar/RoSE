package rose

import chisel3._
import chisel3.util._
import testchipip._
import chisel3.experimental.{IO, IntParam, BaseModule}
import freechips.rocketchip.subsystem.{BaseSubsystem, CacheBlockBytes}
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf

class RoseDMA(param: DstParams)(implicit p: Parameters) extends LazyModule {
  val port_param = param
  val node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "rose-dma", sourceId = IdRange(0, 1))))))
  lazy val module = new RoseDMAModuleImp(this)
}

class RoseDMAModuleImp(outer: RoseDMA) extends LazyModuleImp(outer){
  val config = p(RoseAdapterKey).get

  val io = IO(new Bundle{
    val rx = Flipped(Decoupled(UInt(config.width.W)))
    val cam_buffer = Output(UInt(1.W))
    val counter_max = Input(UInt(config.width.W))
  })

  val fifo = Module(new Queue(UInt(config.width.W), 32))
  fifo.io.enq <> io.rx

  val (mem, edge) = outer.node.out(0)
  val addrBits = edge.bundle.addressBits
  val blockBytes = p(CacheBlockBytes)

  require(config.width <= blockBytes)

  val mIdle :: mWrite :: mResp :: Nil = Enum(3)
  val mstate = RegInit(mIdle)

  val addr = Reg(UInt(addrBits.W))
  val buffer_next = Wire(UInt(config.width.W))
  buffer_next := fifo.io.deq.bits
  val buffer = RegEnable(next = buffer_next, enable = fifo.io.deq.fire)
  val counter_enabled = Wire(Bool())
  // It does not like recursive definitions?
  val counter_next = Wire(UInt(config.width.W))
  val counter : UInt = RegEnable(next = counter_next, init = 0.U(config.width.W), enable = counter_enabled)
  counter_next := Mux((counter < io.counter_max + io.counter_max - 4.U), counter + 4.U, 0.U)

  io.cam_buffer := counter >= io.counter_max
  fifo.io.deq.ready := mstate === mIdle
  mem.a.valid := mstate === mWrite
  mem.d.ready := mstate === mResp
  dontTouch(mem.d.valid)

  // putting the buffer data on the TL mem lane
  mem.a.bits := edge.Put(
  fromSource = 0.U,
  toAddress = addr,
  lgSize = log2Ceil(config.width).U - 3.U,
  data = buffer)._2

  addr := outer.port_param.DMA_address.U + counter
  counter_enabled := mem.a.fire && mstate === mWrite

  switch(mstate){
    is (mIdle){
      //grab data into the fifo
      mstate := Mux(fifo.io.deq.fire, mWrite, mIdle)
    }
    is (mWrite){
      // edge.done refers to fully transmit a message possibly with multiple beats, while mem.d.fire refers to response to a single beat
      // edge.done means we need to load another block from FIFO to buffer
      mstate := Mux(mem.a.fire, mResp, mWrite)
    }
    is (mResp){
      mstate := Mux(mem.d.fire, mIdle, mResp)
    }
  }
}