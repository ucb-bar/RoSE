import os
from tokenize import String
from xmlrpc.client import Boolean
import synchronizer
import control_drone
import control_car

import argparse
import threading
import time

HOST = "localhost"  # Private aws IP
# AIRSIM_IP = "zr-desktop.cs.berkeley.edu"
AIRSIM_IP = "44.205.8.36"

SYNC_PORT = 10001  # Port to listen on (non-privileged ports are > 1023)
DATA_PORT = 60002  # Port to listen on (non-privileged ports are > 1023)

# Thread to track synchronization code
class SyncThread(threading.Thread):

    def __init__(self, args):
        threading.Thread.__init__(self)
        if args["Vehicle_type"] == "drone":
            control = control_drone.IntermediateDroneApi()
        elif args["Vehicle_type"] == "car":
            control = control_car.IntermediateCarApi()
        self.sync = synchronizer.Synchronizer(HOST, SYNC_PORT, DATA_PORT, firesim_step=args['Firesim_steps'], airsim_step=args['Airsim_steps'], control=control, airsim_ip=args['Airsim_ip'], cycle_limit=args['Cycle_limit'], vehicle=args["Vehicle_type"])
    
    def run(self):
        self.sync.run()

class FiresimThread(threading.Thread):

    def __init__(self, args):
        threading.Thread.__init__(self)
        self.sim_type = args['Rtl_sim']
    
    def run(self):
        if "MIDAS" in self.sim_type:
            self.run_midas()
        elif "FireSim" in self.sim_type:
            self.run_firesim()
        else:
            raise ValueError("Simulator not supported!")

    def run_midas(self):
        os.chdir(r"/scratch/$(whoami)/firesim/sim")
        os.system("echo printing directory")
        os.system("pwd")
        os.system("make TARGET_CONFIG=DDR3FRFCFS_WithDefaultFireSimBridges_WithFireSimConfigTweaks_chipyard.RoseTLRocketConfig SIM_BINARY=/scratch/$(whoami)/firesim/target-design/chipyard/tests/airsimpackettest.riscv run-verilator-debug")

    def run_firesim(self):
        # os.system("firesim infrasetup")
        os.system("firesim runworkload > firesim.log")

if __name__ == "__main__":
    arg_list = argparse.ArgumentParser()

    # Add arguments to the parser
    arg_list.add_argument("-a", "--Airsim-steps", type=int, default=1, help="airsim steps")
    arg_list.add_argument("-f", "--Firesim-steps", type=int, default=10000, help="firesim steps")
    arg_list.add_argument("-r", "--Rtl-sim", default="MIDAS", help="rtl simulator used(MIDAS or FireSim)")
    arg_list.add_argument("-i", "--Airsim-ip", default=AIRSIM_IP, help="IP address of airsim server")
    arg_list.add_argument("-c", "--Cycle-limit", type=int, default = None)
    arg_list.add_argument("-v", "--Vehicle-type", type=str, default = "drone", help = "The vehicle you want to simulate")
    args = vars(arg_list.parse_args())

    # control = control_drone.IntermediateDroneApi()
    # control.launchStabilizer(args['Airsim_ip'])
    
    sync_thread = SyncThread(args)
    sync_thread.start()
    while not sync_thread.sync.server_started:
        print("Waiting for synchronizer server to start...")
        time.sleep(0.1)
    firesim_thread = FiresimThread(args)
    firesim_thread.start()
    sync_thread.join()
    if 'MIDAS' in args['Rtl_sim']:
        time.sleep(1)
        os.system("pkill -9 VFireSim-debug")
    else:
        os.system("firesim kill")
        os.system("screen -XS guestmount quit")
        os.system("guestunmount /scratch/iansseijelly/firesim_run_temp/sim_slot_0/mountpoint")