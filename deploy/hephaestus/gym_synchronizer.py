from rose_packet import RoSEPacket, Blob
from socket_thread import SocketThread


import collections
import gymnasium as gym
import register_envs
import time
import numpy as np
import os
import yaml

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


class Synchronizer: 

    #def __init__(self, host=HOST, sync_port=SYNC_PORT, data_port=DATA_PORT, firesim_step=10000):
    def __init__(self, host=HOST, sync_port=SYNC_PORT, data_port=DATA_PORT, firesim_step=10_000_000):
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

        gym_env = self.load_config()

        self.env = gym.make(gym_env)
        self.load_gym_sim_config(gym_env)

        # TODO Put this somewhere clean
        self.firesim_step = firesim_step
        RoSEPacket.firesim_step = firesim_step

        # TODO assign this from a config
        self.num_sockets = 1
        self.cycle_limit = None

        # Initialize sockets for 
        self.socket_threads = []
        for i in range(self.num_sockets):
            self.socket_threads.append(SocketThread(self))

        # intialize frame counter
        self.count = 0

        self.obs = None
        self.done = None


    def run(self):
        # Start running threads for each RTL simulator
        for socket_thread in self.socket_threads:
            socket_thread.start()
        self.start_time = time.time()

        # TODO: This should take in a list of connections,
        # And send firesim steps to each
        self.send_firesim_step()

        # TODO: Add bandwidth sending

        self.obs = self.env.reset()
        self.done = False
        self.rew = 0
        
        # TODO get defaults from config
        self.action = {
            'targets': [0, 0, 0, 0],
            'active': False
        }

        obs = self.env.reset()

        while True:
            # Check to see if sim is finished
            self.check_task_termination()

            # Step robotics simulator
            self.obs, self.rew, self.done, _ =  self.env.step(self.action)

            # Step RTL simulation
            self.grant_firesim_token()

            # Process data from firesim
            self.process_fsim_data_packets()

            # TODO add logging code

            # Process counts for debuggng and logs
            self.process_count()

    def process_count(self):
        if self.count % 20 == 0:
            print(f"Stepping simulation: {self.count} iters")
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
        packet.init(CS_DEFINE_STEP, 4, [self.firesim_step])
        self.txqueue.append(packet)

    def send_bw(self, dst, bw):
        packet = RoSEPacket()
        packet.init(CS_CFG_BW, 8, [dst, bw])
        self.txqueue.append(packet)

    def grant_firesim_token(self):
        # print("Enqueuing new token")
        packet = RoSEPacket()
        packet.init(CS_GRANT_TOKEN, 0, None)
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

    def process_fsim_data_packet(self):
        packet = self.data_rxqueue.pop(0)
        print(f"Dequeued data packet: {packet}")
        if packet.cmd == CS_REQ_IMG:
            INPUT_DIM = 56
            img_arr = self.obs['camera'].reshape(INPUT_DIM, INPUT_DIM * 3)

            for row in img_arr:
                img_packet_arr = row.view(np.uint32).tolist()
                packet = RoSEPacket()
                packet.init(CS_RSP_IMG, len(img_packet_arr)*4, img_packet_arr)
                blob = Blob(packet.latency, packet)
                stable_heap_push(self.txpq, blob)

    def process_fsim_data_packets(self):
        while len(self.data_rxqueue) > 0:
            self.process_fsim_data_packet()
        while(len(self.txpq) > 0 and self.txpq[0].latency < 1):
            packet_blob = stable_heap_pop(self.txpq)
            packet = packet_blob.packet
            self.txqueue.append(packet)
            print(f"appended packet: {packet}")

    def load_config(self):
        # Determine the path to the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the gym environment name from config_deploy_gym.yaml
        with open(os.path.join(script_dir, '../config/config_deploy_gym.yaml'), 'r') as f:
            config = yaml.safe_load(f)
            gym_env = config.get('gym_env', 'AirSimEnv-v0')  # Default to 'AirSimEnv-v0' if not found
        
        print(f"Using Gym environment: {gym_env}")
        return gym_env

    def load_gym_sim_config(self, gym_env):
        # Determine the path to the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the file name and open the specific config
        config_file = os.path.join(script_dir, f'../config/config_gym_{gym_env}.yaml')
        with open(config_file, 'r') as f:
            gym_sim_config = yaml.safe_load(f)
        
        self.packet_bindings = {}
        for packet in gym_sim_config['packets']:
            hex_id = packet['id']
            self.packet_bindings[hex_id] = packet  # bind the id to the packet configuration

    # def process_fsim_data_packet(self):
    #     packet = self.data_rxqueue.pop(0)
    #     print(f"Dequeued data packet: {packet}")

    #     packet_config = self.packet_bindings.get(packet.cmd)
    #     if not packet_config:
    #         print(f"Unknown packet cmd: {packet.cmd}")
    #         return
        
    #     # Retrieve observation related to the packet name
    #     if packet_config['type'] == 'reqrsp':
    #         obs_data = self.obs
    #         for idx in packet_config['indices']:
    #             obs_data = obs_data[idx]
    #         print(f"Extracted packet obs: {obs_data}")
    #         if len(obs_data.shape) == 1:
    #             # Just a 1D array, process accordingly (send one response packet)
    #             # Assuming packet preparation is similar for all 1D arrays (to be modified if needed)
    #             packet_arr = obs_data.tolist()
    #             packet = RoSEPacket()  # You might need to adjust this based on actual RoSEPacket initialization
    #             packet.init(packet.cmd, len(packet_arr) * 4, packet_arr)  # You might need to adjust the multiplier
    #             blob = Blob(packet.latency, packet)
    #             stable_heap_push(self.txpq, blob)

                
                
    #         elif len(obs_data.shape) >= 2:
    #             # 2D array, send the response in rows
    #             INPUT_DIM = 56  # This is hardcoded, consider making it dynamic if necessary
    #             packet_arr = obs_data.reshape(INPUT_DIM, INPUT_DIM * 3)
    #             print(f"packet_arr: {packet_arr}")
    #             for row in packet_arr:
    #                 row_packet_arr = row.view(np.uint32).tolist()
    #                 packet = RoSEPacket()  # You might need to adjust this
    #                 packet.init(packet.cmd, len(row_packet_arr) * 4, row_packet_arr)  # You might need to adjust the multiplier
    #                 blob = Blob(packet.latency, packet)
    #                 stable_heap_push(self.txpq, blob)


if __name__ == "__main__":

    sync = Synchronizer()
    sync.run()
        