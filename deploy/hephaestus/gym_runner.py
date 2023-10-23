import os
import signal
from tokenize import String
from xmlrpc.client import Boolean
import gym_synchronizer
import control_drone
import control_car

import argparse
import threading
import time

HOST = "localhost"  # Private aws IP
# AIRSIM_IP = "zr-desktop.cs.berkeley.edu"
AIRSIM_IP = "localhost"
# AIRSIM_IP = "44.205.8.36"

SYNC_PORT = 10001  # Port to listen on (non-privileged ports are > 1023)
DATA_PORT = 60002  # Port to listen on (non-privileged ports are > 1023)

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
        os.system("firesim runworkload > /dev/null")
        # os.system("firesim kill")
        os.system("firesim kill &")
        os.system("sleep 10")
        os.system("screen -XS guestmount quit")
        os.kill(os.getpid(), signal.SIGTERM)
        exit()

if __name__ == "__main__":
    
    print("Starting synchronizer thread")
    sync_thread = SyncThread(None)
    sync_thread.start()
    print("Starting firesim thread")
    firesim_thread = FiresimThread(None)
    firesim_thread.start()
    print("Joining synchronizer thread")
    sync_thread.join()
    print("Joining firesim thread")
    os.kill(os.getpid(), signal.SIGTERM)

    
    firesim_thread.join()