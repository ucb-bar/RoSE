#!/usr/bin/env python3
import os
import signal
import threading
import argparse
import gym_synchronizer
from socket_thread import ServerThread

TASK = ["run", "build"]

# Thread to track synchronization code
class SyncThread(threading.Thread):

    def __init__(self, args):
        self.sync = gym_synchronizer.Synchronizer()
        threading.Thread.__init__(self)
    
    def run(self):
        self.sync.run()

class FiresimThread(threading.Thread):

    def __init__(self, args):
        threading.Thread.__init__(self)
    
    def run(self):
        self.run_firesim()

    def run_firesim(self):
        os.system("firesim kill > /dev/null")
        # os.system("firesim infrasetup > /dev/null")
        # os.system("screen -S switch0 -d -m 'gdb /scratch/iansseijelly/FIRESIM_RUNS_DIR/switch_slot0/switch0'")
        # os.system("./scratch/iansseijelly/FIRESIM_RUNS_DIR/sim_slot0/sim-run.sh")
        os.system("firesim runworkload > /dev/null")
        # os.system("firesim kill")
        os.system("firesim kill &")
        os.system("sleep 10")
        os.system("screen -XS guestmount quit")
        os.kill(os.getpid(), signal.SIGTERM)
        exit()

def construct_rose_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='The task to perform', choices=TASK, default='run')
    parser.add_argument('--target', type=str, help='[build]: The target C file to build', default='packettest')
    parser.add_argument('--use_trap', type=bool, help='[build]: Use the trap.S file', default=False)
    return parser

if __name__ == "__main__":
    args = construct_rose_argparser().parse_args()
    header = """
██████╗░░█████╗░░██████╗███████╗
██╔══██╗██╔══██╗██╔════╝██╔════╝
██████╔╝██║░░██║╚█████╗░█████╗░░
██╔══██╗██║░░██║░╚═══██╗██╔══╝░░
██║░░██║╚█████╔╝██████╔╝███████╗
╚═╝░░╚═╝░╚════╝░╚═════╝░╚══════╝
""" 
    print(header)

    if args.task == "build":
        print("[RoSE]:Building target file: " + args.target)
        dummy_sync = gym_synchronizer.DummySynchronizer()
        gym_env = dummy_sync.load_config()
        dummy_sync.load_gym_sim_config(gym_env)
        dummy_sync.genRoSECPacketHeader()
        # get current file directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        build_path = os.path.join(dir_path, "../..", "soc", "sw", "build_packettest.py")
        if args.use_trap:
            os.system("python3 " + build_path + " --target " + args.target + " --use_trap " + str(args.use_trap))
        else:
            os.system("python3 " + build_path + " --target " + args.target)
        exit()

    if args.task == "run":
        print("[RoSE]:Running RoSE")
        print("[RoSE]:Starting synchronizer thread")
        sync_thread = SyncThread(None)

        condition = threading.Condition()

        server_thread = ServerThread(sync_thread.sync, condition)
        server_thread.start()
        
        print("[RoSE]:Starting firesim thread")
        firesim_thread = FiresimThread(None)
        firesim_thread.start()

        while (server_thread.connected_sockets < server_thread.num_sockets):
            pass

        print("[RoSE]:Joining synchronizer thread")
        sync_thread.start()
        sync_thread.join()
        print("[RoSE]:Joining firesim thread")

        with condition:
            condition.notify()

        server_thread.join()
        print("[RoSE]:Joining server thread")
        
        firesim_thread.join()
        os.kill(os.getpid(), signal.SIGTERM)