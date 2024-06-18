//See LICENSE for license details
package firesim.bridges

import midas.widgets._
import midas.models.AbstractClockGate

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import rose.{RosePortIO, RoseAdapterParams, RoseAdapterKey, RoseAdapterArbiterIO, ConfigRoutingIOBundle, Dataflow}
import firrtl.annotations.HasSerializationHints
import java.io.{File, FileWriter}
import freechips.rocketchip.util.{AsyncQueue, AsyncQueueParams}

// A utility register-based lookup table that maps the id to the corresponding dst_port index
class RoseArbTable(params: RoseAdapterParams) extends Module {
  val io = IO(new Bundle{
    // Look up ports
    val key = Input(UInt(params.width.W))
    val key_valid = Input(Bool())
    val value = Output(UInt(params.width.W))
    val keep_header = Output(Bool())
    // Configuration ports
    val config_routing = Flipped(Decoupled(new ConfigRoutingIOBundle(params)))
  })

  // spawn a vector of registers, storing the configured routing values
  // TODO: replace with a memory
  val routing_table = RegInit(VecInit(Seq.fill(0x80)(0.U(log2Ceil(params.dst_ports.seq.size).W))))
  // Only keep headers if the port is a reqrsp port and has no dataflow
  val keeping_table = VecInit(params.dst_ports.seq.map(port => (port.port_type == "reqrsp" && port.df_params.size == 0).B))

  when (io.config_routing.valid) {
    routing_table(io.config_routing.bits.header) := io.config_routing.bits.channel
  }

  // look up the routing table
  io.value := Mux(io.key_valid, routing_table(io.key), 0.U)
  io.keep_header := keeping_table(io.value)
  io.config_routing.ready := true.B
}

class RoseAdapterArbiter(params: RoseAdapterParams) extends Module{
  val w = params.width
  val io = IO(new RoseAdapterArbiterIO(params))

  val sIdle :: sHeader :: sLoad :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val counter = Reg(UInt(w.W))

  val arb_table = Module(new RoseArbTable(params))
  io.config_routing <> arb_table.io.config_routing
  arb_table.io.key := io.tx.bits
  // storing the query result for future use
  val latched_keep_header = RegEnable(arb_table.io.keep_header, io.tx.fire && (state === sIdle))
  val latched_idx = RegEnable(arb_table.io.value, io.tx.fire && (state === sIdle))

  val rx_val = Wire(Bool())
  for (i <- 0 until params.dst_ports.seq.size) {
    when (Mux(state === sIdle, i.U === arb_table.io.value, i.U === latched_idx)) {
      io.rx(i).valid := rx_val
    } .otherwise {
      io.rx(i).valid := false.B
    }
    io.rx(i).bits := io.tx.bits 
  }

  io.budget.ready := false.B
  io.bigstep.ready := false.B
  arb_table.io.key_valid := (state === sIdle)
  rx_val := false.B
  io.tx.ready := false.B

  switch(state) {
    is(sIdle) {
      def can_advance = io.budget.valid && io.tx.valid && io.bigstep.valid && (
        ((io.budget.bits < io.cycleBudget) || (io.currstep > io.bigstep.bits))
           && (io.cycleBudget =/= io.cycleStep))
      io.tx.ready := io.rx(arb_table.io.value).ready && can_advance 
      io.budget.ready := io.rx(arb_table.io.value).ready && can_advance
      io.bigstep.ready := io.rx(arb_table.io.value).ready && can_advance
      rx_val := Mux(io.tx.fire, arb_table.io.keep_header, false.B)
      state := Mux(io.tx.fire, sHeader, sIdle)
    } 
    is(sHeader) {
      rx_val := Mux(latched_keep_header, io.tx.valid, false.B)
      counter := io.tx.bits >> 2
      io.tx.ready := io.rx(latched_idx).ready 
      state := Mux(io.tx.fire, Mux(io.tx.bits === 0.U, sIdle, sLoad), sHeader)
    }
    is(sLoad) {
      rx_val := io.tx.valid
      counter := Mux(io.tx.fire, counter - 1.U, counter)
      state := Mux(counter === 0.U, sIdle, sLoad)
      io.tx.ready := io.rx(latched_idx).ready && (counter =/= 0.U) 
    }
  }

  // number of times the state goes to sHeader
  val counter_state_sheader = RegInit(0.U(32.W))
  when(state === sIdle && io.tx.fire) {
    counter_state_sheader := counter_state_sheader + 1.U
  }
  // number of times the budget is fired
  val counter_budget_fired = RegInit(0.U(32.W))
  when(io.budget.fire) {
    counter_budget_fired := counter_budget_fired + 1.U
  }
  // number of times the tx is fired
  val counter_tx_fired = RegInit(0.U(32.W))
  when(io.tx.fire) {
    counter_tx_fired := counter_tx_fired + 1.U
  }
  // number of times the rx_0 is fired
  val counter_rx_0_fired = RegInit(0.U(32.W))
  when(io.rx(0).fire) {
    counter_rx_0_fired := counter_rx_0_fired + 1.U
  }

  val counter_rx_1_fired = RegInit(0.U(32.W))
  when(io.rx(1).fire) {
    counter_rx_1_fired := counter_rx_1_fired + 1.U
  }
  
  io.debug.counter_state_sheader := counter_state_sheader
  io.debug.counter_budget_fired := counter_budget_fired
  io.debug.counter_tx_fired := counter_tx_fired
  io.debug.counter_rx_0_fired := counter_rx_0_fired
  io.debug.counter_rx_1_fired := counter_rx_1_fired
}

class BandWidthWriter(params: RoseAdapterParams) extends Module{
  val io = IO(new Bundle{
    val config_bits = Input(UInt(32.W))
    val config_valid = Input(Bool())
    val config_destination = Input(UInt(32.W))
    val output_bits = Vec(params.dst_ports.seq.size, Output(UInt(32.W)))
    val output_valid = Vec(params.dst_ports.seq.size, Output(Bool()))
  })

  for (i <- 0 until params.dst_ports.seq.length) {
    io.output_bits(i) := io.config_bits
    io.output_valid(i) := io.config_valid && (io.config_destination === i.U)
  }
}

class rxcontroller(width: Int) extends Module{
  val io = IO(new Bundle{
    val rx = Flipped(Decoupled(UInt(width.W)))
    val tx = Decoupled(UInt(width.W))
    val fire = Input(Bool())
    // val counter_reset = Input(Bool())
    val bww_bits = Input(UInt(32.W))
    val bww_valid = Input(Bool())
  })

  val sRxIdle :: sRxRecv :: sRxSend :: sRxDelay1 :: sRxDelay2 :: Nil = Enum(5)
  val rxState = RegInit(sRxIdle)
  val rxData = Reg(UInt(32.W))

  val bandwidth_threshold = RegEnable(io.bww_bits, 0.U(32.W), io.bww_valid)

  val counter_next = Wire(UInt(width.W))
  val counter: UInt = RegEnable(counter_next, 0.U(width.W), io.fire)
  counter_next := Mux(io.tx.fire, 0.U, Mux((counter < bandwidth_threshold), counter + 1.U, counter))
  val depleted = Wire(Bool())
  depleted := counter === bandwidth_threshold

  io.tx.bits := rxData
  io.tx.valid := (rxState === sRxSend)
  io.rx.ready := (rxState === sRxRecv) && depleted

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
      when(io.rx.fire) {
        rxState := sRxSend
        rxData := io.rx.bits
      }.otherwise {
        rxState := sRxRecv
        rxData := 0.U
      }
    }
    is(sRxSend) {
      when(io.fire) {
        rxState := sRxDelay1
        rxData := 0.U
      }.otherwise {
        rxState := sRxSend
        rxData := rxData
      }
    }
    is(sRxDelay1) {
      rxData := 0.U
      when(io.fire) {
        rxState := sRxDelay2
      } otherwise {
        rxState := sRxDelay1
      }
    }
    is(sRxDelay2) {
      rxData := 0.U
      when(io.fire) {
        rxState := sRxIdle
      } otherwise {
        rxState := sRxDelay2
      }
    }
  }
}

class RoseBridgeTargetIO(params: RoseAdapterParams) extends Bundle {
  val clock = Input(Clock())
  val airsimio = Flipped(new RosePortIO(params))
  val reset = Input(Bool())
  // Note this reset is optional and used only to reset target-state modelled
  // in the bridge This reset just like any other Bool included in your target
  // interface, simply appears as another Bool in the input token.
}

case class RoseKey(roseparams: RoseAdapterParams) extends HasSerializationHints {
  def typeHints = Seq(classOf[RoseAdapterParams]) ++ roseparams.typeHints
}

class RoseBridge()(implicit p: Parameters) extends BlackBox with Bridge[HostPortIO[RoseBridgeTargetIO], RoseBridgeModule] {

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

class RoseBridgeModule(key: RoseKey)(implicit p: Parameters) extends BridgeModule[HostPortIO[RoseBridgeTargetIO]]()(p) {
  lazy val module = new BridgeModuleImp(this) {
    val params = key.roseparams
    val io = IO(new WidgetIO())

    // This creates the host-side interface of your TargetIO
    val hPort = IO(HostPort(new RoseBridgeTargetIO(params)))

    // Generate some FIFOs to capture tokens...
    val txfifo = Module(new Queue(UInt(32.W), 32))
    //val rxfifo = Module(new Queue(UInt(32.W), 128))

    // Generate a FIFO to capture time step allocations
    val rx_ctrl_fifo = Module(new Queue(UInt(8.W), 16))

    val cycleBudget = RegInit(0.U(32.W))
    val cycleStep   = RegInit(0.U(32.W))

    rx_ctrl_fifo.io.deq.ready := true.B;

    val target = hPort.hBits.airsimio
    // In general, your BridgeModule will not need to do work every host-cycle. In simple Bridges,
    // we can do everything in a single host-cycle -- fire captures all of the
    // conditions under which we can consume and input token and produce a new
    // output token
    val fire = hPort.toHost.hValid &&    // We have a valid input token: toHost ~= leaving the transformed RTL
               hPort.fromHost.hReady &&  // We have space to enqueue a new output token
              //  txfifo.io.enq.ready  &&   // We have space to capture new TX data
               (cycleBudget < cycleStep) // still have cycles left in the budget
    val targetReset = fire & hPort.hBits.reset
    
    val acg = Module(new AbstractClockGate)
    acg.I := clock
    acg.CE := fire

    val rxfifo = Module(new AsyncQueue(UInt(32.W), AsyncQueueParams(depth = 256))) 
    val rx_budget_fifo = Module(new AsyncQueue(UInt(32.W), AsyncQueueParams(depth = 64)))
    val rx_bigstep_fifo = Module(new AsyncQueue(UInt(32.W), AsyncQueueParams(depth = 64)))
    val config_async_queue = Module(new AsyncQueue(new ConfigRoutingIOBundle(params), AsyncQueueParams(depth = 64)))

    def bind_clock_domain(module: AsyncQueue[UInt]) = {
      module.io.enq_clock := clock 
      module.io.enq_reset := reset.asBool || targetReset
      module.io.deq_clock := acg.O
      module.io.deq_reset := reset.asBool || targetReset
    }

    bind_clock_domain(rxfifo)
    bind_clock_domain(rx_budget_fifo)
    bind_clock_domain(rx_bigstep_fifo)
    
    config_async_queue.io.enq_clock := clock
    config_async_queue.io.enq_reset := reset.asBool || targetReset
    config_async_queue.io.deq_clock := acg.O
    config_async_queue.io.deq_reset := reset.asBool || targetReset

    txfifo.reset := reset.asBool || targetReset
    // Add to the cycles the tool is permitted to run forward
    when(rx_ctrl_fifo.io.deq.valid) {
        cycleBudget := 0.U(32.W) 
    }  .elsewhen(fire) {
        cycleBudget := cycleBudget + 1.U(32.W)
    } .otherwise {
        cycleBudget := cycleBudget
    }
    rx_ctrl_fifo.reset := reset.asBool || targetReset

    val cycle_count_up_wrap = (cycleBudget + 1.U(32.W) === cycleStep) && fire
    val (stepCount, _) = Counter(cycle_count_up_wrap, 65535 * 256)

    // instantiate the rose arbiter
    val rosearb = Module(new RoseAdapterArbiter(params))
    rosearb.reset := reset.asBool || targetReset
    rosearb.clock := acg.O

    rosearb.io.cycleBudget := cycleBudget
    rosearb.io.tx <> rxfifo.io.deq
    rosearb.io.budget <> rx_budget_fifo.io.deq
    rosearb.io.bigstep <> rx_bigstep_fifo.io.deq
    rosearb.io.cycleStep := cycleStep
    rosearb.io.currstep := stepCount
    
    val bww = Module(new BandWidthWriter(params))
    rosearb.clock := acg.O
    rosearb.io.config_routing <> config_async_queue.io.deq
    
    for (i <- 0 until params.dst_ports.seq.size) {
      // for each dst_port, generate a shallow queue and connect it to the arbiter
      val q = Module(new Queue(UInt(params.width.W), 32))
      q.reset := reset.asBool || targetReset
      q.clock := acg.O

      val end_async_q = Module(new AsyncQueue(UInt(32.W), AsyncQueueParams(depth = 64)))
      end_async_q.io.enq_clock := acg.O 
      end_async_q.io.enq_reset := reset.asBool || targetReset
      end_async_q.io.deq_clock := clock
      end_async_q.io.deq_reset := reset.asBool || targetReset

      // generate a rxcontroller for each dst_port
      val rxctrl = Module(new rxcontroller(params.width))
      rxctrl.io.rx <> end_async_q.io.deq
      rxctrl.reset := reset.asBool || targetReset

      val channel_param = params.dst_ports.seq(i)
      val dataflows: Seq[Dataflow] = channel_param.df_params.map(_.userProvided.elaborate)
      
      if (dataflows.nonEmpty) {
        // connect the dataflows one after the other
        dataflows.map(_.io).foldLeft(q.io) { case (prev, next) =>
          next.enq <> prev.deq // chaining
          next
        }
        dataflows.last.io.deq <> end_async_q.io.enq // tail case
        dataflows.foreach { df =>
          df.clock := acg.O // clock gating
          df.reset := reset.asBool || targetReset
        }
      } else {
        q.io.deq <> end_async_q.io.enq
      }
      q.io.enq <> rosearb.io.rx(i)

      rxctrl.io.fire := fire
      rxctrl.io.tx <> target.rx(i)
      rxctrl.io.bww_bits := bww.io.output_bits(i)
      rxctrl.io.bww_valid := bww.io.output_valid(i)
    }
    
    txfifo.io.enq.valid := target.tx.valid && fire
    txfifo.io.enq.bits  := target.tx.bits
    target.tx.ready := txfifo.io.enq.ready

    hPort.toHost.hReady := fire
    hPort.fromHost.hValid := fire

    // DEBUG: 
      // 1. a counter when something gets fired int target
      // 2. a counter whenever state goes to sHeader
      // 3. a counter on the tx and budget interface

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
    genWOReg(rx_bigstep_fifo.io.enq.bits, "in_bigstep_bits")
    Pulsify(genWORegInit(rx_bigstep_fifo.io.enq.valid, "in_bigstep_valid", false.B), pulseLength = 1)
    genROReg(rx_bigstep_fifo.io.enq.ready, "in_bigstep_ready")      
    // Generate regisers for the rx-side of the AirSim; this is eseentially the same as the above
    genWOReg(rx_budget_fifo.io.enq.bits, "in_budget_bits")
    Pulsify(genWORegInit(rx_budget_fifo.io.enq.valid, "in_budget_valid", false.B), pulseLength = 1)
    genROReg(rx_budget_fifo.io.enq.ready, "in_budget_ready")
    // Generate registers for reading in time step limits
    genWOReg(rx_ctrl_fifo.io.enq.bits, "in_ctrl_bits")
    Pulsify(genWORegInit(rx_ctrl_fifo.io.enq.valid, "in_ctrl_valid", false.B), pulseLength = 1)
    genROReg(rx_ctrl_fifo.io.enq.ready, "in_ctrl_ready")

    // Generate registers for reading total cycles that have passed
    genROReg(stepCount, "cycle_count")
    genROReg(cycleBudget, "cycle_budget")

    // Generate registers for writing the step amount
    genWOReg(cycleStep, "cycle_step")
    
    // Generate registers for configuring bandwidth
    genWOReg(bww.io.config_bits, "bww_config_bits")
    Pulsify(genWORegInit(bww.io.config_valid, "bww_config_valid", false.B), pulseLength = 1)
    genWOReg(bww.io.config_destination, "bww_config_destination")

    // Generate registers for configuring routing table
    genWOReg(config_async_queue.io.enq.bits.header, "config_routing_header")
    genWOReg(config_async_queue.io.enq.bits.channel, "config_routing_channel")
    Pulsify(genWORegInit(config_async_queue.io.enq.valid, "config_routing_valid", false.B), pulseLength = 1)
    genROReg(config_async_queue.io.enq.ready, "config_routing_ready")
    // Debug Registers
    genROReg(rosearb.io.debug.counter_state_sheader, "arb_counter_state_sheader")
    genROReg(rosearb.io.debug.counter_budget_fired, "arb_counter_budget_fired")
    genROReg(rosearb.io.debug.counter_tx_fired, "arb_counter_tx_fired")
    genROReg(rosearb.io.debug.counter_rx_0_fired, "arb_counter_rx_0_fired")
    genROReg(rosearb.io.debug.counter_rx_1_fired, "arb_counter_rx_1_fired")
    
    // This method invocation is required to wire up the bridge to the simulated software
    override def genHeader(base: BigInt, memoryRegions: Map[String, BigInt], sb: StringBuilder): Unit = {
      genConstructor(base, sb, "airsim_t", "airsim")
    }

    // Emits a C header for this bridge construction
    def genRoseCPortHeader(bridgeParams: RoseAdapterParams): Unit = {
      val sb = new StringBuilder()
      sb.append("//This file was generated by RoSEBridge.scala, based on RoseAdapterParams\n")
      sb.append(s"#define ROSE_PORT_COUNT ${bridgeParams.dst_ports.seq.size}\n")
      sb.append(s"#define ROSE_PORT_WIDTH ${bridgeParams.width}\n")
      sb.append(f"#define ROSE_STATUS_ADDR 0x${bridgeParams.address}%x\n")
      sb.append("\n")

      sb.append(f"#define ROSE_TX_DATA_ADDR 0x${bridgeParams.address + 0x8}%x\n")
      sb.append(s"#define ROSE_TX_ENQ_READY (reg_read32(ROSE_STATUS_ADDR) & 0x1)\n")
      sb.append("\n")

      val idx_map = bridgeParams.genidxmap
      params.dst_ports.seq.zipWithIndex.foreach {
        case (port, i) => port.port_type match {
          case "DMA" => {
            sb.append(s"//${port.port_type}_${port.name}_port_channel_$i\n")
            // 3*4 is a magic number, 0x08 for status and 0x0C for tx_data
            sb.append(f"#define ROSE_DMA_CONFIG_COUNTER_ADDR_$i 0x${bridgeParams.address + (idx_map(i)+3+bridgeParams.dst_ports.seq.count(_.port_type != "DMA"))*4}%x\n")
            sb.append(f"#define ROSE_DMA_CURR_COUNTER_ADDR_$i 0x${bridgeParams.address + (idx_map(i)+3+bridgeParams.dst_ports.seq.size)*4}%x\n")
            sb.append(f"#define ROSE_DMA_BASE_ADDR_$i 0x${bridgeParams.dst_ports.seq(i).DMA_address}%x\n")
            sb.append(f"#define ROSE_DMA_BUFFER_$i (reg_read32(ROSE_STATUS_ADDR) & 0x${1<<(bridgeParams.dst_ports.seq.length-idx_map(i))}%x)\n")
            sb.append(f"#define ROSE_DMA_CURR_COUNTER_$i (reg_read32(ROSE_DMA_CURR_COUNTER_ADDR_$i))\n")
            sb.append("\n")
          }
          case "reqrsp" => {
            sb.append(s"//${port.port_type}_${port.name}_port_channel_$i\n")
            val index = idx_map(i)
            sb.append(f"#define ROSE_RX_DATA_ADDR_$i 0x${bridgeParams.address + 0xC + index*4}%x\n")
            sb.append(f"#define ROSE_RX_DATA_$i (reg_read32(ROSE_RX_DATA_ADDR_$i))\n")
            sb.append(f"#define ROSE_RX_DEQ_VALID_$i (reg_read32(ROSE_STATUS_ADDR) & 0x${1<<(params.dst_ports.seq.count(_.port_type != "DMA")-index)}%x)\n")
            sb.append("\n")
          }
        }
      }
      val fileWriter = new FileWriter(new File("../../../sw/generated-src/rose_c_header/rose_port.h"))
      fileWriter.write(sb.toString)
      fileWriter.close()
    }
    genRoseCPortHeader(params)
    // This method invocation is required to wire up all of the MMIO registers to
    // the simulation control bus (AXI4-lite)
    genCRFile()
  }
}
