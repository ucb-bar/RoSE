//See LICENSE for license details
package firesim.bridges

import midas.widgets._

import chisel3._
import chisel3.util._
import chisel3.experimental.{DataMirror, Direction, IO}
import freechips.rocketchip.config.Parameters
import freechips.rocketchip.subsystem.PeripheryBusKey
import sifive.blocks.devices.uart.{UARTPortIO, UARTParams}

import rose.{RosePortIO, RoseAdapterParams, RoseAdapterKey, RoseAdapterArbiter}

class rxcontroller(params: RoseAdapterParams) extends Module{
  val io = IO(new Bundle{
    val rx = Flipped(Decoupled(params.width.W))
    val tx = Decoupled(params.width.W)
    val fire = Input(Bool())
  })

  val sRxIdle :: sRxRecv :: sRxSend :: sRxDelay1 :: sRxDelay2 :: Nil = Enum(5)
  val rxState = RegInit(sRxIdle)
  val rxData = Reg(UInt(32.W))

  switch(rxState) {
    is(sRxIdle) {
      rxData := 0.U
      when(io.tx.ready) {
        rxState := sRxRecv
      }.otherwise {
        rxState := sRxIdle
      }
    }
    is(sRxRecv) {
      when(io.rx.valid) {
        rxState := sRxSend
        rxData := io.rx.bits
      }.otherwise {
        rxState := sRxRecv
        rxData := 0.U
      }
    }
    is(sRxSend) {
      when(fire) {
        rxState := sRxDelay1
        rxData := 0.U
      }.otherwise {
        rxState := sRxSend
        rxData := rxData
      }
    }
    is(sRxDelay1) {
      rxData := 0.U
      when(fire) {
        rxState := sRxDelay2
      } otherwise {
        rxState := sRxDelay1
      }
    }
    is(sRxDelay2) {
      rxData := 0.U
      when(fire) {
        rxState := sRxIdle
      } otherwise {
        rxState := sRxDelay2
      }
    }
  }
}

// DOC include start: AirSim Bridge Target-Side Interface
class RoseBridgeTargetIO(params: RoseAdapterParams) extends Bundle {
  val clock = Input(Clock())
  val airsimio = Flipped(new RosePortIO(params))
  val reset = Input(Bool())
  // Note this reset is optional and used only to reset target-state modelled
  // in the bridge This reset just like any other Bool included in your target
  // interface, simply appears as another Bool in the input token.
}
// DOC include end: AirSim Bridge Target-Side Interface

// DOC include start: AirSIm Bridge Constructor Arg
// Out bridge module constructor argument. This captures all of the extra
// metadata we'd like to pass to the host-side BridgeModule. Note, we need to
// use a single case class to do so, even if it is simply to wrap a primitive
// type, as is the case for AirSim (int)
case class RoseKey(cycles: Int)
// DOC include end: AirSim Bridge Constructor Arg

// DOC include start: AirSim Bridge Target-Side Module
class RoseBridge()(implicit p: Parameters) extends BlackBox
    with Bridge[HostPortIO[RoseBridgeTargetIO], RoseBridgeModule] {
  // Since we're extending BlackBox this is the port will connect to in our target's RTL
  
  val io = IO(new RoseBridgeTargetIO(p(RoseAdapterKey).get))
  // Implement the bridgeIO member of Bridge using HostPort. This indicates that
  // we want to divide io, into a bidirectional token stream with the input
  // token corresponding to all of the inputs of this BlackBox, and the output token consisting of 
  // all of the outputs from the BlackBox
  val bridgeIO = HostPort(io)

  // And then implement the constructorArg member
  val constructorArg = Some(RoseKey(1000000))

  // Finally, and this is critical, emit the Bridge Annotations -- without
  // this, this BlackBox would appear like any other BlackBox to Golden Gate
  generateAnnotations()
}
// DOC include end: AirSim Bridge Target-Side Module

// DOC include start: AirSim Bridge Companion Object
object RoseBridge {
  def apply(clock: Clock, airsimio: RosePortIO)(implicit p: Parameters): RoseBridge = {
 // def apply(airsimio: AirSimPortIO)(implicit p: Parameters): AirSimBridge = {
    // TODO
    //val clock: Clock = Wire(Clock())
    //clock := false.B.asClock
    val rosebridge = Module(new RoseBridge())
    rosebridge.io.airsimio <> airsimio
    rosebridge.io.clock := clock
    // println("ddddddddddddddddddddddddddddddd: AirSim")
    // println("ep")
    // println(ep)
    // println("ep.io.airsimio")
    // println(ep.io.airsimio)
    // println("ep.io.clock")
    // println(ep.io.clock)
    // println("ddddddddddddddddddddddddddddddd")
    rosebridge
  }
}
// DOC include end: AirSim Bridge Companion Object

// DOC include start: AirSim Bridge Header
// Our AirSimBridgeModule definition, note:
// 1) it takes one parameter, key, of type AirSimKey --> the same case class we captured from the target-side
// 2) It accepts one implicit parameter of type Parameters
// 3) It extends BridgeModule passing the type of the HostInterface
//
// While the Scala type system will check if you parameterized BridgeModule
// correctly, the types of the constructor arugument (in this case AirSimKey),
// don't match, you'll only find out later when Golden Gate attempts to generate your module.
class RoseBridgeModule(key: RoseKey)(implicit p: Parameters) extends BridgeModule[HostPortIO[RoseBridgeTargetIO]]()(p) {
  lazy val module = new BridgeModuleImp(this) {
    val cycles = key.cycles
    // This creates the interfaces for all of the host-side transport
    // AXI4-lite for the simulation control bus, =
    // AXI4 for DMA
    val io = IO(new WidgetIO())

    // This creates the host-side interface of your TargetIO
    val hPort = IO(HostPort(new RoseBridgeTargetIO(p(RoseAdapterKey).get)))

    // Generate some FIFOs to capture tokens...
    val txfifo = Module(new Queue(UInt(32.W), 32))
    val rxfifo = Module(new Queue(UInt(32.W), 32))

    // COSIM-CODE
    // Generate a FIFO to capture time step allocations
    val rx_ctrl_fifo = Module(new Queue(UInt(8.W), 16))

    // Create counters to track number of cycles elapsed
    // Initialize number of cycles 
    val cycleBudget = RegInit(0.U(32.W))

    // Initialize amount to increment cycle budget by
    val cycleStep   = RegInit(1000.U(32.W))

    // can add to budget every cycle
    rx_ctrl_fifo.io.deq.ready := true.B;
    // COSIM-CODE

    val target = hPort.hBits.airsimio
    // In general, your BridgeModule will not need to do work every host-cycle. In simple Bridges,
    // we can do everything in a single host-cycle -- fire captures all of the
    // conditions under which we can consume and input token and produce a new
    // output token
    val fire = hPort.toHost.hValid &&    // We have a valid input token: toHost ~= leaving the transformed RTL
               hPort.fromHost.hReady &&  // We have space to enqueue a new output token
               txfifo.io.enq.ready  &&   // We have space to capture new TX data
               (cycleBudget > 0.U(32.W)) // still have cycles left in the budget

    val targetReset = fire & hPort.hBits.reset
    rxfifo.reset := reset.asBool || targetReset
    txfifo.reset := reset.asBool || targetReset


    // COSIM-CODE
    // Add to the cycles the tool is permitted to run forward
    when(rx_ctrl_fifo.io.deq.valid) {
        cycleBudget := cycleBudget + cycleStep
    }  .elsewhen(fire) {
        cycleBudget := cycleBudget - 1.U(32.W)
    } .otherwise {
        cycleBudget := cycleBudget
    }

    rx_ctrl_fifo.reset := reset.asBool || targetReset

    // Count total elapsed cycles
    val (cycleCount, testWrap) = Counter(fire, 65535 * 256)
    // COSIM-CODE

    // instantiate the rose arbiter
    val rosearb = Module(new RoseAdapterArbiter(p(RoseAdapterKey).get))
    rosearb.io.tx = rxfifo.io.deq
    // for each dst_port, generate a shallow queue and connect it to the arbiter
    for (i <- 0 until p(RoseAdapterKey).get.dst_ports.size) {
      val dst_port = p(RoseAdapterKey).get.dst_ports(i)
      val q = Module(new Queue(UInt(dst_port.width.W), 32))
      // generate a rxcontroller for each dst_port
      val rxctrl = Module(new rxcontroller(p(RoseAdapterKey).get))
      q.io.enq <> rosearb.io.rx(i)
      q.io.deq <> rxctrl.io.rx 
      rxctrl.io.tx <> target.rx(i)
    }
    // Drive fifo signals from AirSimIO
    txfifo.io.enq.valid := target.tx.valid && fire
    txfifo.io.enq.bits  := target.tx.bits
    target.tx.ready := txfifo.io.enq.ready
    //TODO: vectorize the for loop to reduce or it for this use
    rxfifo.io.deq.ready := (rxState === sRxRecv)
    // // Drive AirSimIO signals from fifo
    // target.port_rx_enq_valid := rxfifo.io.deq.valid
    // target.port_tx_deq_ready := txfifo.io.enq.ready
    // target.port_rx_enq_bits  := rxfifo.io.deq.bits

    hPort.toHost.hReady := fire
    hPort.fromHost.hValid := fire

    // DOC include start: AirSim Bridge Footer
    // Exposed the head of the queue and the valid bit as a read-only registers
    // with name "out_bits" and out_valid respectively
    genROReg(txfifo.io.deq.bits, "out_bits")
    genROReg(txfifo.io.deq.valid, "out_valid")

    // Generate a writeable register, "out_ready", that when written to dequeues
    // a single element in the tx_fifo. Pulsify derives the register back to false
    // after pulseLength cycles to prevent multiple dequeues
    Pulsify(genWORegInit(txfifo.io.deq.ready, "out_ready", false.B), pulseLength = 1)

    // Generate regisers for the rx-side of the AirSim; this is eseentially the reverse of the above
    genWOReg(rxfifo.io.enq.bits, "in_bits")
    Pulsify(genWORegInit(rxfifo.io.enq.valid, "in_valid", false.B), pulseLength = 1)
    genROReg(rxfifo.io.enq.ready, "in_ready")

    // COSIM-CODE
    // Generate registers for reading in time step limits
    genWOReg(rx_ctrl_fifo.io.enq.bits, "in_ctrl_bits")
    Pulsify(genWORegInit(rx_ctrl_fifo.io.enq.valid, "in_ctrl_valid", false.B), pulseLength = 1)
    genROReg(rx_ctrl_fifo.io.enq.ready, "in_ctrl_ready")

    // Generate registers for reading total cycles that have passed
    genROReg(cycleCount, "cycle_count")
    genROReg(cycleBudget, "cycle_budget")

    // Generate registers for writing the step amount
    genWOReg(cycleStep, "cycle_step")
    // COSIM-CODE

    // This method invocation is required to wire up all of the MMIO registers to
    // the simulation control bus (AXI4-lite)
    genCRFile()
    // DOC include end: AirSim Bridge Footer
  }
}
