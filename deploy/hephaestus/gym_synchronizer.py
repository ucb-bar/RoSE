from rose_packet import RoSEPacket, Blob
from socket_thread import SocketThread



import collections
import gymnasium as gym
import register_envs
from utils.logger import GymLogger
import time
import datetime
import numpy as np
import os
import yaml
from functools import reduce


CS_RESET = 0x01

CS_GRANT_TOKEN  = 0x80 
CS_REQ_CYCLES   = 0x81 
CS_RSP_CYCLES   = 0x82 
CS_DEFINE_STEP  = 0x83
CS_RSP_STALL    = 0x84
CS_CFG_BW       = 0x85

CS_REQ_IMG      = 0x10
CS_RSP_IMG      = 0x11
CS_REQ_IMG_POLL = 0x16
CS_RSP_IMG_POLL = 0x17

HOST = "localhost"
SYNC_PORT = 10001
DATA_PORT = 60002


# Utility functions
def stable_heap_push(heap, item):
    heap.append(item)
    return heap

def stable_heap_pop(heap):
    item = heap.pop(0)
    heap.sort()
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


class Synchronizer: 

    def __init__(self, host=HOST, sync_port=SYNC_PORT, data_port=DATA_PORT, firesim_step=10000, firesim_freq=1_000_000_000):
    #def __init__(self, host=HOST, sync_port=SYNC_PORT, data_port=DATA_PORT, firesim_step=10_000_000):
        self.txqueue = []
        self.txpq = [] 
        self.data_rxqueue = []
        self.sync_rxqueue = []

        # TODO Remove unecessary ones
        self.host = host
        self.sync_host = host
        self.sync_port = sync_port
        self.data_port = data_port

        self.streaming_queue = collections.OrderedDict()

        self.num_sockets = 1
        self.cycle_limit = None

        # Load general simulation configurations
        self.firesim_freq = firesim_freq
        self.firesim_step = firesim_step
        self.render = False
        gym_env = self.load_config()


        self.load_gym_sim_config(gym_env)
        self.env = gym.make(gym_env, render_mode='rgb_array', **self.gym_kwargs)
        
        # Check if firesim step in seconds is a multiple of gym timestep
        self.firesim_period = self.firesim_step / self.firesim_freq
        tolerance = 1e-10  # define a small tolerance value
        remainder = self.firesim_period % self.gym_timestep
        if remainder > tolerance and self.gym_timestep - remainder > tolerance:  # checks if the remainder is significant
            raise ValueError(f"Firesim period ({self.firesim_period}) must be a multiple of gym timestep ({self.gym_timestep})")

        self.gym_step_per_firesim_step = round(self.firesim_period / self.gym_timestep)

        # Assign timing information to the RoSEPacket class
        RoSEPacket.firesim_step = self.firesim_step
        for cmd in self.packet_bindings.keys():
            if 'latency' in self.packet_bindings[cmd]:
                RoSEPacket.cmd_latency_dict[cmd] = self.packet_bindings[cmd]['latency'] / self.firesim_period
                RoSEPacket.cmd_latency_dict[cmd+1] = self.packet_bindings[cmd]['latency'] / self.firesim_period


        # Initialize sockets 
        self.socket_threads = []
        for i in range(self.num_sockets):
            self.socket_threads.append(SocketThread(self))

        # intialize frame counter
        self.count = 0

        # Initialize logger filenames
        self.filename_base = f'{gym_env}-cycles-{self.firesim_step}-gym-step-{self.gym_timestep}-freq-{self.firesim_freq}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

        # Initialize gym variables
        self.obs = None
        self.done = None
        print(f"Using firesim step: {self.firesim_step}, firesim freq: {self.firesim_freq}, gym timestep: {self.gym_timestep}, gym step per firesim step: {self.gym_step_per_firesim_step}")

    def run(self):
        # Start running threads for each RTL simulator
        for socket_thread in self.socket_threads:
            socket_thread.start()
        self.start_time = time.time()


        # Initialize the logger
        log_dir = f'{os.path.dirname(os.path.abspath(__file__))}/logs'
        self.logger = GymLogger(self.env, self.firesim_period, self.start_time, self.packet_bindings, log_dir=log_dir, log_filename=f'runlog-{self.filename_base}.csv', video_filename=f'recording-{self.filename_base}.avi', max_duration=self.cycle_limit/self.firesim_freq, plot_filename=f'plot-{self.filename_base}.png')

        # TODO: This should take in a list of connections,
        # And send firesim steps to each
        self.send_firesim_step()

        # TODO: Add bandwidth sending

        self.done = False
        self.rew = 0
    
        # TODO get defaults from config
        self.action = default_action_for_space(self.env.action_space)
        self.default_action = default_action_for_space(self.env.action_space)

        obs, _ = self.env.reset()

        print("Starting RoSE Simulation Loop")

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
            self.logger.log_data(self.obs, self.action)
            if self.render:
                self.logger.display()

            # Log the rendered frame for this timestep
            self.logger.log_rendering()

            # Step RTL simulation
            self.grant_firesim_token()

            # Process data from firesim
            self.process_fsim_data_packets()

            # TODO add any additional logging

            # Process counts for debugging and logs
            self.process_count()

            # If simulation is done, break the loop
            if self.done:
                print("Simulation done!")
                break

        # Once the loop ends, close the logger to finalize logs and save video
        self.logger.close()
        exit()

    def process_count(self):
        if self.count % 20 == 0:
            print(f"Stepping simulation: {self.count} iters (sim time = {self.count * self.firesim_period}s)")
            self.logger.save_video()
        # if self.count >= 40:
        #     exit(0)
        self.count += 1

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
            for socket_thread in self.socket_threads:
                socket_thread.kill()
            
            # terminate simulation
            time.sleep(1)
            exit()

    def send_firesim_step(self):
        packet = RoSEPacket()
        # packet.init(CS_DEFINE_STEP, 4, np.array([self.firesim_step], dtype=np.uint32))
        packet.init(CS_DEFINE_STEP, 4, [self.firesim_step])
        self.txqueue.append(packet)

    def send_bw(self, dst, bw):
        packet = RoSEPacket()
        packet.init(CS_CFG_BW, 8, [dst, bw])
        self.txqueue.append(packet)

    def grant_firesim_token(self):
        packet = RoSEPacket()
        packet.init(CS_GRANT_TOKEN, 0, None)
        # print("Printing txqueue")
        # for packet in self.txqueue:
        #     print(f"txqueue: {packet}")
        # print(f"Enqueuing new token: {packet}")
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())

        while True:
            if len(self.sync_rxqueue) > 0:
                self.sync_rxqueue.pop(0)
                break

    def get_firesim_cycles(self):
        packet = RoSEPacket()
        packet.init(CS_REQ_CYCLES, 0, None)
        self.txqueue.append(packet)

        while len(self.sync_rxqueue) == 0:
            pass
        response = self.sync_rxqueue.pop(0)
        return response.data[0]

    def process_fsim_data_packets(self):
        while len(self.data_rxqueue) > 0:
            self.process_fsim_data_packet()
        while(len(self.txpq) > 0 and self.txpq[0].latency < 1):
            packet_blob = stable_heap_pop(self.txpq)
            packet = packet_blob.packet
            self.txqueue.append(packet)
            # print(f"appended packet: {packet}")
        # Now, iterate through the rest of the queue, decrement latency by 1
        for blobs in self.txpq:
            blobs.latency = blobs.latency - 1
            blobs.packet.latency = blobs.latency

    def load_config(self):
        # Determine the path to the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the gym environment name from config_deploy_gym.yaml
        with open(os.path.join(script_dir, '../config/config_deploy_gym.yaml'), 'r') as f:
            config = yaml.safe_load(f)
            gym_env = config.get('gym_env', 'AirSimEnv-v0')  # Default to 'AirSimEnv-v0' if not found
        
        # Load timing information from the config
        if 'firesim_step' in config:
            self.firesim_step = config['firesim_step']
        if 'firesim_freq' in config:
            self.firesim_freq = config['firesim_freq']
        if 'max_sim_time' in config:
            self.cycle_limit = config['max_sim_time'] * self.firesim_freq
        if 'render' in config:
            self.render = config['render']

        
        print(f"Using Gym environment: {gym_env}")
        return gym_env

    def load_gym_sim_config(self, gym_env):
        # Determine the path to the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the file name and open the specific config
        config_file = os.path.join(script_dir, f'../config/config_gym_{gym_env}.yaml')
        with open(config_file, 'r') as f:
            gym_sim_config = yaml.safe_load(f)
        
        print(f"loaded config: {gym_sim_config}")

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

    def process_fsim_data_packet(self):
        packet = self.data_rxqueue.pop(0)
        cmd = packet.cmd
        print(f"Dequeued data packet: {packet}")
        
        if cmd == CS_RESET:
            print("Resetting environment")
            self.obs = self.env.reset()
            self.done = False
            self.rew = 0
            self.action = default_action_for_space(self.env.action_space)
            self.default_action = default_action_for_space(self.env.action_space)
            self.logger.count_reset()
            return

        packet_config = self.packet_bindings.get(packet.cmd)
        if not packet_config:
            print(f"Unknown packet cmd: {packet.cmd}")
            return
        
        
        # Retrieve observation related to the packet name
        if packet_config['type'] == 'reqrsp':
            obs_data = self.obs
            if packet_config['indices'] is not None:
                for idx in packet_config['indices']:
                    obs_data = obs_data[idx]
            if len(obs_data.shape) == 1:
                # Just a 1D array, process accordingly (send one response packet)
                #packet_arr = obs_data.view(np.uint32).tolist()
                packet_arr = np.frombuffer(obs_data.tobytes(), dtype=np.uint32).copy().tolist()
                packet = RoSEPacket()  # You might need to adjust this based on actual RoSEPacket initialization
                packet.init(cmd+1, len(packet_arr) * 4, packet_arr)  # You might need to adjust the multiplier
                blob = Blob(RoSEPacket.cmd_latency_dict.get(cmd+1,0), packet)
                stable_heap_push(self.txpq, blob)

            else: 
                # 2D array, send the response in rows
                INPUT_DIM = obs_data.shape[0]
                packet_arr = obs_data.reshape(INPUT_DIM, -1)
                for row in packet_arr:
                    row_packet_arr = row.view(np.uint32).copy().tolist()
                    # print(f"row_packet_arr: {row_packet_arr}")
                    packet = RoSEPacket()  # You might need to adjust this
                    packet.init(cmd+1, len(row_packet_arr) * 4, row_packet_arr)  # You might need to adjust the multiplier
                    blob = Blob(RoSEPacket.cmd_latency_dict.get(cmd+1, 0), packet)
                    stable_heap_push(self.txpq, blob)
        
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
            
            print(f"action: {data_to_assign}")
        
        self.logger.count_packet(cmd)


if __name__ == "__main__":

    sync = Synchronizer()
    sync.run()
        
