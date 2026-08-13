from rose_packet import *
from socket_thread import SocketThread

import collections
import gymnasium as gym
import register_envs
from utils.logger import GymLogger
import time
import datetime
import faulthandler
import threading
import numpy as np
import os
import sys
import yaml
import zlib
from functools import reduce

# Port/host are env-overridable so multiple co-sim cells can run in parallel on one host
# (each on its own port). The parallel stress harness sets ROSE_SYNC_PORT per cell and
# passes the matching --rose-port to rose_spike_sim. Defaults preserve the single-run flow.
HOST = os.environ.get("ROSE_SYNC_HOST", "localhost")
SYNC_PORT = int(os.environ.get("ROSE_SYNC_PORT", "10001"))
DATA_PORT = int(os.environ.get("ROSE_DATA_PORT", "60002"))

class CONTROL_HEADERS:
    # fsim->gym
    CS_RESET        = 0xFF
    # sync->fsim
    CS_GRANT_TOKEN  = 0x80 
    CS_REQ_CYCLES   = 0x81 
    CS_RSP_CYCLES   = 0x82 
    CS_DEFINE_STEP  = 0x83
    CS_RSP_STALL    = 0x84
    CS_CFG_BW       = 0x85
    CS_CFG_ROUTE    = 0x86

# Utility functions
def stable_heap_push(heap, item):
    heap.append(item)
    return heap

def stable_heap_pop(heap):
    item = heap.pop(0)
    heap.sort() # stable
    return item

def default_action_for_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return 0
    elif isinstance(space, gym.spaces.Box):
        return (space.high + space.low) / 2.0
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros_like(space.nvec, dtype=int)
    elif isinstance(space, gym.spaces.MultiBinary):
        return np.zeros(space.n, dtype=int)
    elif isinstance(space, gym.spaces.Dict):
        return {key: default_action_for_space(sub_space) for key, sub_space in space.spaces.items()}
    else:
        raise ValueError(f"Don't know how to create a default action for space of type {type(space)}")

# useful for building the packet header
class DummySynchronizer:
    def __init__(self):
        pass

    def load_config(self):
        # Determine the path to the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the gym environment name from config_deploy_gym.yaml
        with open(os.path.join(script_dir, '../config/config_deploy_gym.yaml'), 'r') as f:
            config = yaml.safe_load(f)
            gym_env = config.get('gym_env', 'AirSimEnv-v0')  # Default to 'AirSimEnv-v0' if not found
        # Env-var override so a run can select a different env (e.g. the multisensor nav env)
        # without editing the committed config_deploy_gym.yaml default.
        gym_env = os.environ.get('ROSE_GYM_ENV', gym_env)
        
        # Load timing information from the config
        if 'firesim_step' in config:
            self.firesim_step = config['firesim_step']
        if 'firesim_freq' in config:
            self.firesim_freq = config['firesim_freq']
        # Env overrides so the stress harness can model a slower SoC clock (fewer cycles/step
        # for spike to functionally simulate -> faster wall clock) WITHOUT editing the
        # committed 1 GHz demo default. Keep firesim_step/firesim_freq == the same period so
        # the control rate (gym_timestep) is unchanged; only shrink both to cut the per-step
        # cycle budget down toward (but above) the TinyMPC solve cost.
        if os.environ.get('ROSE_FIRESIM_STEP'):
            self.firesim_step = int(os.environ['ROSE_FIRESIM_STEP'])
        if os.environ.get('ROSE_FIRESIM_FREQ'):
            self.firesim_freq = int(os.environ['ROSE_FIRESIM_FREQ'])
        if 'max_sim_time' in config:
            self.cycle_limit = config['max_sim_time'] * self.firesim_freq
        # ROSE_MAX_SIM_TIME overrides the config cap. Needed when a slow on-SoC
        # guest (e.g. the scalar_f16 fused-nav model) burns many firesim grants
        # per physics step under the freeze seam, so the default 12 s cap would
        # cut the episode short (~1.4 s of flight) before reaching a gate.
        if os.environ.get('ROSE_MAX_SIM_TIME'):
            self.cycle_limit = float(os.environ['ROSE_MAX_SIM_TIME']) * self.firesim_freq
        if 'render' in config:
            self.render = config['render']
        
        print(f"Using Gym environment: {gym_env}")
        return gym_env
    
    def load_gym_sim_config(self, gym_env):
        # Determine the path to the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the file name and open the specific config
        config_file = os.path.join(script_dir, f'../config/config_gym_{gym_env}.yaml')
        self.load_gym_sim_config_from_file(config_file)
        # print(f"loaded config: {gym_sim_config}")
    
    def load_gym_sim_config_from_file(self, file):
        with open(file, 'r') as f:
            gym_sim_config = yaml.safe_load(f)
        # Load the gym_timestep. This represents how much simulation time passes per env.step()
        if 'gym_timestep' in gym_sim_config:
            self.gym_timestep = gym_sim_config['gym_timestep']
        else:
            raise ValueError("gym_timestep not found in config file")
        
        # Load the custom **kwargs for the gym environment
        if 'gym_kwargs' in gym_sim_config:
            self.gym_kwargs = gym_sim_config['gym_kwargs']
        
        self.packet_bindings = {}
        for packet in gym_sim_config['packets']:
            hex_id = packet['id']
            self.packet_bindings[hex_id] = packet  # bind the id to the packet configuration
        
        self.channel_bandwidth = gym_sim_config['channel_bandwidth']
        self.n_fsim_nodes = gym_sim_config['n_fsim_nodes']
    
    def genRoSECPacketHeader(self):
        sb = ["//RoSE Control Packet Headers"]
        for k,v in CONTROL_HEADERS.__dict__.items():
            if not k.startswith("__"):
                sb.append(f"#define {k} 0x{v:02x}")
        sb.append("//RoSE Payload Packet Headers")
        for k,v in self.packet_bindings.items():
            sb.append(f"#define CS_{v['name'].upper()} 0x{k:02x}")

        # Resolve the RoSÉ repo root from $ROSE_DIR (set by rose-setup.sh), else derive
        # it from this file's location (deploy/hephaestus -> repo root is ../..).
        rose_dir = os.environ.get(
            "ROSE_DIR",
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
        out_path = os.path.join(rose_dir, "soc", "sw", "generated-src", "rose_c_header", "rose_packet.h")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(sb))

class Synchronizer(DummySynchronizer): 

    def __init__(self, host=HOST, sync_port=SYNC_PORT, data_port=DATA_PORT, firesim_step=10000, firesim_freq=1_000_000_000, yaml_path=None):
        super().__init__()

        # this node list is always written by server_thread during setup, 
        # and always only read by synchronizer on the main thread after server_thread exits
        self.nodes = []

        self.host = host
        self.sync_host = host
        self.sync_port = sync_port
        self.data_port = data_port

        self.cycle_limit = None

        # Load general simulation configurations
        self.firesim_freq = firesim_freq
        self.firesim_step = firesim_step
        self.render = False
        gym_env = self.load_config()

        if yaml_path != None:
            print(f"Loading config from file: {yaml_path}")
            self.load_gym_sim_config_from_file(yaml_path)
        else:
            print(f"Loading config from default: {gym_env}")
            self.load_gym_sim_config(gym_env)

        # disable_env_checker: the reset-time passive checker asserts reset() obs keys
        # match the observation_space, but cam_front/lowdim/tof_cross populate on the
        # first step (lazy camera render), not at reset(0). The co-sim serves obs from
        # post-step obs, so the reset-only check is spurious here.
        self.env = gym.make(gym_env, render_mode='rgb_array', disable_env_checker=True, **self.gym_kwargs)
        
        # Check if firesim step in seconds is a multiple of gym timestep
        self.firesim_period = self.firesim_step / self.firesim_freq
        tolerance = 1e-10  # define a small tolerance value
        remainder = self.firesim_period % self.gym_timestep
        if remainder > tolerance and self.gym_timestep - remainder > tolerance:  # checks if the remainder is significant
            raise ValueError(f"Firesim period ({self.firesim_period}) must be a multiple of gym timestep ({self.gym_timestep})")

        self.gym_step_per_firesim_step = round(self.firesim_period / self.gym_timestep)

        # Assign timing information to the Packet class
        Packet.firesim_step = self.firesim_step
        for cmd in self.packet_bindings.keys():
            if 'latency' in self.packet_bindings[cmd]:
                Packet.cmd_latency_dict[cmd] = self.packet_bindings[cmd]['latency'] / self.firesim_period

        # intialize frame counter
        self.count = 0

        # Initialize logger filenames
        self.filename_base = f'{gym_env}-cycles-{self.firesim_step}-gym-step-{self.gym_timestep}-freq-{self.firesim_freq}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

        # Initialize gym variables
        self.obs = None
        self.done = None
        print(f"Using firesim step: {self.firesim_step}, firesim freq: {self.firesim_freq}, gym timestep: {self.gym_timestep}, gym step per firesim step: {self.gym_step_per_firesim_step}")

    def _start_stall_watchdog(self):
        """Daemon thread that flags a stalled co-sim step and dumps thread stacks.

        Normal env.step() takes ~0.1-0.4 s; a render/reset a few seconds. A stall of
        tens of seconds means a hang. On stall we print the step it froze at and call
        faulthandler.dump_traceback(all_threads=True) so the stuck frame is unambiguous:
          - a frame in env.step / isaac / physx  -> Isaac/PhysX side (SoC is idle & fine)
          - a frame in check_token_exhaustion / get_firesim_cycles -> waiting on SoC ack
        Enabled by default; tune/disable via ROSE_SYNC_WATCHDOG_S (0 disables).
        """
        try:
            timeout_s = float(os.environ.get("ROSE_SYNC_WATCHDOG_S", "45"))
        except ValueError:
            timeout_s = 45.0
        if timeout_s <= 0:
            return

        def _watch():
            last_count = -1
            last_change = time.time()
            fired_for = -1
            while True:
                time.sleep(min(5.0, timeout_s / 2.0))
                if getattr(self, "done", False):
                    return
                c = getattr(self, "count", 0)
                now = time.time()
                if c != last_count:
                    last_count = c
                    last_change = now
                    continue
                # count has not advanced; check how long
                if now - last_change >= timeout_s and c != fired_for:
                    fired_for = c
                    print(f"\n[SYNC WATCHDOG] co-sim step frozen at count={c} for "
                          f"{now - last_change:.0f}s — dumping thread stacks to locate "
                          f"the stall (env.step=Isaac side; check_token_exhaustion/"
                          f"get_firesim_cycles=waiting on SoC ack):", flush=True)
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    sys.stderr.flush()

        t = threading.Thread(target=_watch, name="sync-stall-watchdog", daemon=True)
        t.start()

    def run(self):
        # Start running threads for each RTL simulator
        for socket_thread in self.nodes:
            socket_thread.start()
        self.start_time = time.time()

        # Stall watchdog: the co-sim occasionally hangs (guest idle at 0% CPU, sync
        # pinned busy). To attribute it — Isaac env.step() vs the SoC-ack busy-spin in
        # check_token_exhaustion()/get_firesim_cycles() — a daemon thread watches step
        # progress and, if self.count stops advancing for ROSE_SYNC_WATCHDOG_S seconds,
        # dumps every thread's Python stack (which pins the exact stuck line). Diagnostic
        # only: it never touches control flow, so normal runs are unaffected.
        self._start_stall_watchdog()

        # Initialize the logger
        log_dir = f'{os.path.dirname(os.path.abspath(__file__))}/logs'
        self.logger = GymLogger(self.env, self.firesim_period, self.start_time, self.packet_bindings, log_dir=log_dir, log_filename=f'runlog-{self.filename_base}.csv', video_filename=f'recording-{self.filename_base}.avi', max_duration=self.cycle_limit/self.firesim_freq, plot_filename=f'plot-{self.filename_base}.png')

        # TODO: This should take in a list of connections,
        # And send firesim steps to each
        for t in self.nodes:
            self.send_firesim_step(t)

        for i, t in enumerate(self.nodes):
            for j, bw in enumerate(self.channel_bandwidth[i]):
                print(f"Setting bandwidth to node {i} to channel {j} with {bw}")
                self.send_bw(j, bw, t)
        
        for cmd in self.packet_bindings.keys():
            if 'channel' in self.packet_bindings[cmd]:
                print(f"Sending route for 0x{cmd:02x} to channel {self.packet_bindings[cmd]['channel']}")
                for t in self.nodes:
                    self.send_route(cmd, self.packet_bindings[cmd]['channel'], t)

        self.obs = self.env.reset()
        self.done = False
        self.rew = 0
    
        # TODO get defaults from config
        self.action = default_action_for_space(self.env.action_space)
        self.default_action = default_action_for_space(self.env.action_space)

        obs, _ = self.env.reset()

        print("*** Starting RoSE Simulation Loop ***")

        while True:
            # Check to see if sim is finished
            self.check_task_termination()

            # Step robotics simulator
            for _ in range(self.gym_step_per_firesim_step):
                # Ensure that if the action is not latched, only use it once
                # before resetting to the default action
                self.obs, self.rew, self.done, _, _ =  self.env.step(self.action)
                self.action = self.default_action

            # Log observation and action at this timestep
            # self.logger.log_data(self.obs, self.action)
            # if self.render:
                # self.logger.display()

            # Log the rendered frame for this timestep
            # self.logger.log_rendering()

            # Step RTL simulation
            for t in self.nodes:
                self.grant_firesim_token(t)
            
            for t in self.nodes:
                self.check_token_exhaustion(t)
            
            self.count += 1

            # Process data from firesim
            for t in self.nodes:
                self.process_fsim_data_packets(t, self.count)

            # Process streaming packets
            for t in self.nodes:
                self.schedule_streaming_packets(t)

            # Process counts for debugging and logs
            self.process_count()

            # If simulation is done, break the loop
            if self.done:
                print("Simulation done!")
                break

        # Once the loop ends, close the logger to finalize logs and save video
        self.logger.close()
        # Hard-exit (see _bridge_disconnected_shutdown): exit()/sys.exit only unwinds
        # main and then blocks forever joining Isaac's non-daemon threads.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def process_count(self):
        if self.count % 20 == 0:
            print(f"Stepping simulation: {self.count} iters (sim time = {self.count * self.firesim_period}s)")
            self.logger.save_video()
        # if self.count >= 40:
        #     exit(0)

    def check_task_termination(self):
        if (self.cycle_limit is not None and self.count * self.firesim_step >= self.cycle_limit) or self.done:
            # Log end time
            end_time = time.time()
            elapsed_time = end_time - self.start_time

            # Print reason for exit
            if self.cycle_limit is not None and self.count * self.firesim_step >= self.cycle_limit:
                print("Terminated due to exceeding maximum cycles!")
            else:
                print("Terminated due to completing objective!")
            
            # Close all connections to simulators
            for socket_thread in self.nodes:
                socket_thread.kill()

            # Finalize logs/video, then hard-exit. This is the normal-termination path
            # (cycle_limit / objective done); it hit the same shutdown hang as the
            # disconnect path — exit() unwinds only main and blocks forever joining
            # Isaac's non-daemon threads. Mirror _bridge_disconnected_shutdown.
            try:
                self.logger.close()
            except Exception:
                pass
            time.sleep(1)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def send_firesim_step(self, target_thread):
        packet = Control_Packet(CONTROL_HEADERS.CS_DEFINE_STEP, 4, [self.firesim_step])
        target_thread.txqueue.append(packet)

    def send_bw(self, dst, bw, target_thread):
        packet = Control_Packet(CONTROL_HEADERS.CS_CFG_BW, 8, [dst, bw])
        target_thread.txqueue.append(packet)
    
    def send_route(self, header, channel, target_thread):
        packet = Control_Packet(CONTROL_HEADERS.CS_CFG_ROUTE, 8, [header, channel])
        target_thread.txqueue.append(packet)

    def grant_firesim_token(self, target_thread):
        packet = Control_Packet(CONTROL_HEADERS.CS_GRANT_TOKEN, 0, None)
        target_thread.txqueue.append(packet)

    def _bridge_disconnected_shutdown(self, target_thread):
        """The bridge (spike) closed the socket — co-sim over (spike exit / timeout /
        crash). Shut down cleanly instead of busy-spinning a core on a dead peer forever
        (the sync-side mirror of the spike's own disconnect-exit)."""
        print(f"[SYNC] bridge closed the connection at step {self.count} — co-sim ended; "
              f"shutting down (was waiting on a SoC ack that will never come).", flush=True)
        try:
            self.logger.close()
        except Exception:
            pass
        for st in self.nodes:
            st.kill()
        # Hard exit: sys.exit() only unwinds the main thread and then blocks waiting for
        # Isaac's many non-daemon background threads to join (they never do), leaving the
        # process alive and spinning. os._exit terminates now — the co-sim is over.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def check_token_exhaustion(self, target_thread):
        while True:
            if len(target_thread.sync_rxqueue) > 0:
                target_thread.sync_rxqueue.pop(0)
                break
            if target_thread.disconnected:
                self._bridge_disconnected_shutdown(target_thread)

    def retrieve_obs_push_packet(self, cmd, target_thread, packet_config):
        if packet_config['indices'] is not None:
            obs_data = self.obs
            for idx in packet_config['indices']:
                obs_data = obs_data[idx]
        if len(obs_data.shape) == 1:
            # Just a 1D array, process accordingly (send one response packet)
            #packet_arr = obs_data.view(np.uint32).tolist()
            packet_arr = np.frombuffer(obs_data.tobytes(), dtype=np.uint32).tolist()
            packet = Payload_Packet(cmd, len(packet_arr) * 4, packet_arr)  # You might need to adjust the multiplier
            stable_heap_push(target_thread.txpq, packet)
        else: 
            # 2D array, send the response in rows
            INPUT_DIM = obs_data.shape[0]
            packet_arr = obs_data.reshape(INPUT_DIM, -1)
            for row in packet_arr:
                row_packet_arr = row.view(np.uint32).tolist()
                # print(f"row_packet_arr: {row_packet_arr}")
                packet = Payload_Packet(cmd, len(row_packet_arr) * 4, row_packet_arr)  # You might need to adjust the multiplier
                stable_heap_push(target_thread.txpq, packet)

    def retrieve_obs_push_dma_frame(self, cmd, target_thread, packet_config):
        """Serve a large sensor frame (e.g. the HM01B0 FPV camera) over the DMA channel.

        The bridge's DMA engine memcpy's ONE served payload into guest DRAM at dma_base and
        raises the completion IRQ (see rose_spike_sim.cc deliver() ch0). So — unlike a 2D
        camera served row-per-packet on a reqrsp channel — a DMA frame MUST be a single
        contiguous payload, or successive packets would overwrite the same dma_base. We
        require a 1-D obs (flat frame) and emit exactly one Payload_Packet. Routing (cmd ->
        channel 0) is already configured via send_route() at startup.

        Logs the served frame's CRC32 (== Zephyr crc32_ieee on the guest) + byte length so the
        co-sim can confirm the exact frame landed intact on the SoC."""
        obs_data = self.obs
        for idx in (packet_config['indices'] or []):
            obs_data = obs_data[idx]
        obs_data = np.ascontiguousarray(obs_data)
        if obs_data.ndim != 1:
            raise ValueError("DMA frame '%s' must be a flat 1-D array (got shape %s); a DMA "
                             "payload is a single contiguous transfer" % (packet_config.get('name'), obs_data.shape))
        raw = obs_data.tobytes()
        if len(raw) % 4 != 0:
            raise ValueError("DMA frame '%s' byte length %d not a multiple of 4 (word packing)"
                             % (packet_config.get('name'), len(raw)))
        packet_arr = np.frombuffer(raw, dtype=np.uint32).tolist()
        packet = Payload_Packet(cmd, len(packet_arr) * 4, packet_arr)
        stable_heap_push(target_thread.txpq, packet)
        crc = zlib.crc32(raw) & 0xffffffff
        print("[dma-serve] cmd=0x%02x '%s' nbytes=%d crc32=0x%08x"
              % (cmd, packet_config.get('name'), len(raw), crc), flush=True)

    def get_firesim_cycles(self, target_thread):
        packet = Control_Packet(CONTROL_HEADERS.CS_REQ_CYCLES, 0, None)
        target_thread.txqueue.append(packet)

        while len(target_thread.sync_rxqueue) == 0:
            if target_thread.disconnected:
                self._bridge_disconnected_shutdown(target_thread)
        response = target_thread.sync_rxqueue.pop(0)
        return response.data[0]

    def process_fsim_data_packets(self, target_thread, count):
        while len(target_thread.data_rxqueue) > 0:
            self.process_fsim_data_packet(target_thread)
        while(len(target_thread.txpq) > 0 and target_thread.txpq[0].latency < 1):
            packet = stable_heap_pop(target_thread.txpq)
            # print(f"tagging packet: {count}")
            packet.tag_step(count)
            target_thread.txqueue.append(packet)
            # print(f"appended packet: {packet}")
        # Now, iterate through the rest of the queue, decrement latency by 1
        for blobs in target_thread.txpq:
            blobs.latency = blobs.latency - 1
            # blobs.packet.latency = blobs.latency

    def process_fsim_data_packet(self, target_thread):
        packet = target_thread.data_rxqueue.pop(0)
        cmd = packet.cmd
        print(f"Dequeued data packet: {packet}")
        
        if cmd == CONTROL_HEADERS.CS_RESET:
            print("Resetting environment")
            self.obs = self.env.reset()
            self.done = False
            self.rew = 0
            self.action = default_action_for_space(self.env.action_space)
            self.default_action = default_action_for_space(self.env.action_space)
            self.logger.count_reset()
            return

        # Guest vision-I/O diagnostic echo (cmd 0x21, FMPC_VISION_DBG): 5 words =
        # [cam_hash, tof_hash, low_hash, out0_fp16bits, out1_fp16bits]. Logged so we can
        # compare the FPGA guest's model inputs+output against the spike guest's.
        if packet.cmd == 0x23:
            d = packet.data
            ch = int(d[0]) if len(d) > 0 else -1
            hdr = int(d[1]) if len(d) > 1 else -1
            nb = int(d[2]) if len(d) > 2 else -1
            print(f"[RRDIAG] ch{ch} header=0x{hdr:x} num_bytes={nb}", flush=True)
            return

        if packet.cmd == 0x22:
            d = packet.data
            hdr = int(d[0]) if len(d) > 0 else -1
            nb = int(d[1]) if len(d) > 1 else -1
            print(f"[CAMDIAG] header=0x{hdr:x} num_bytes={nb} (expect header=0x11 num_bytes=5400)", flush=True)
            return

        if packet.cmd == 0x21:
            d = packet.data.view('uint32') if hasattr(packet.data, 'view') else packet.data
            print(f"[VDIAG] cam={int(d[0]):08x} tof={int(d[1]):08x} low={int(d[2]):08x} "
                  f"out0={int(d[3])&0xffff:04x} out1={int(d[4])&0xffff:04x}", flush=True)
            return

        packet_config = self.packet_bindings.get(packet.cmd)
        if not packet_config:
            print(f"Unknown packet cmd: {packet.cmd}")
            return
        
        # Retrieve observation related to the packet name
        if packet_config['type'] == 'reqrsp':
            _pre_txq = len(target_thread.txqueue)
            self.retrieve_obs_push_packet(cmd, target_thread, packet_config)
            if os.environ.get('ROSE_SERVE_DEBUG'):
                # Serve-side trace: after serving a reqrsp, whether the response was queued
                # (txq grew) and how deep the send queue is. A growing/stuck txq_depth means
                # the SEND path (socket->bridge) is backed up; a small one means the response
                # left the sync and any stall is downstream (bridge/RTL delivery to the guest).
                print(f"[SERVE] reqrsp cmd=0x{cmd:x} txq {_pre_txq}->{len(target_thread.txqueue)}", flush=True)

        # Large frames over the DMA channel (camera): single contiguous payload -> ch0 DMA.
        if packet_config['type'] == 'dma':
            self.retrieve_obs_push_dma_frame(cmd, target_thread, packet_config)

        if packet_config['type'] == 'stream':
            assert packet.num_bytes == 4, "Stream packets must be 4 bytes"
            if target_thread.stream_txqueue.get(packet.cmd) is None:
                print(f"Scheduled streaming cmd: {packet.cmd} with interval: {packet.data[0]}")
            else:
                print(f"Updated streaming cmd: {packet.cmd} with interval: {packet.data[0]}")    
            target_thread.stream_txqueue[packet.cmd] = packet.data[0]
        
        if 'action' in packet_config['type']:
            indices = packet_config['indices']
            data_to_assign = packet.data.view(self.action.dtype).copy()
            if indices is not None:
                # Use reduce to drill down into the nested structure
                target = reduce(lambda arr, idx: arr[idx], indices, self.action)
                target[:] = data_to_assign  # Modify the value in place
            else:
                self.action = data_to_assign
            if 'latch' in packet_config['type']:
                self.default_action = self.action.copy()
            # Optional, non-breaking hook: notify the env that a FRESH action packet was received
            # this iteration (a "receive" event, not a value change). Envs that implement a
            # freeze/one-coherent-tick-per-command control seam use this; others ignore it.
            try:
                base = self.env.unwrapped
                if hasattr(base, "on_action_received"):
                    base.on_action_received()
            except Exception:
                pass
            print(f"action: {data_to_assign}")
        
        self.logger.count_packet(cmd)
    
    def schedule_streaming_packets(self, target_thread):
        for cmd, interval in target_thread.stream_txqueue.items():
            if self.count % interval == 0:
                packet_config = self.packet_bindings.get(cmd)
                self.retrieve_obs_push_packet(cmd, target_thread, packet_config)
            # TODO: REMOVE THIS for the latency sweeping test
            if cmd == 0x41:
                curr_latency = Packet.cmd_latency_dict.get(cmd, 0) 
                new_latency = curr_latency + 0.05
                Packet.cmd_latency_dict[cmd] = new_latency
                print(f"Updated latency for cmd: {cmd} to {new_latency}")

if __name__ == "__main__":
    sync = Synchronizer()
    sync.run()