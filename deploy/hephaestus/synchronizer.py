
import socket
from unittest import case
import airsim
import time
import numpy as np
import os
import tempfile
import pprint
import cv2
import sys
import struct
import argparse

import threading
import traceback

import control_drone
import datetime

import pandas as pd

CS_GRANT_TOKEN  = 0x80 
CS_REQ_CYCLES   = 0x81 
CS_RSP_CYCLES   = 0x82 
CS_DEFINE_STEP  = 0x83
CS_RSP_STALL    = 0x84
CS_REQ_WAYPOINT = 0x01
CS_RSP_WAYPOINT = 0x02
CS_SEND_IMU     = 0x03
CS_REQ_ARM      = 0x04
CS_REQ_DISARM   = 0x05
CS_REQ_TAKEOFF  = 0x06

CS_REQ_IMG      = 0x10
CS_RSP_IMG      = 0x11

CS_REQ_DEPTH    = 0x12
CS_RSP_DEPTH    = 0x13

CS_SET_TARGETS  = 0x20

INTCMDS = [CS_GRANT_TOKEN, CS_REQ_CYCLES, CS_RSP_CYCLES, CS_DEFINE_STEP, CS_RSP_STALL, CS_RSP_IMG]

#HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
HOST = "localhost" # Private aws IP
#HOST = "172.31.30.244"
AIRSIM_IP = "zr-desktop.cs.berkeley.edu"

#PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
SYNC_PORT = 10001  # Port to listen on (non-privileged ports are > 1023)
DATA_PORT = 60002  # Port to listen on (non-privileged ports are > 1023)

INPUT_DIM = 56

class CoSimLogger:
    
    def __init__(self, client, cycles, frames, filename=None):
        self.client = client
        self.table = {}
        self.df =  pd.DataFrame({"frame": [0]})
        self.started = False 
        self.start_time = time.time()
        self.cycles = cycles
        self.frames = frames
        try:
            f = open('angle.txt', 'r')
            yaw = int(float(f.readline()))
        except:
            yaw = "unknown"

        if filename is None:
            filename = f'./logs/runlog-angle-{yaw}-cycles-{cycles}-frames-{frames}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
        self.filename = filename
    
    def start(self):
        self.started = True
        self.start_time = time.time()

    def add_row(self, frame):
        self.df = self.df.append({"frame": frame*self.frames, "cycles": frame*self.cycles, "real_time": time.time() - self.start_time}, ignore_index=True)
        # print(f"df row added: {self.df}")

    def log_targets(self, dat):
        self.record_row(dat)

    def log_true_kin(self):
        kin = self.client.simGetGroundTruthKinematics()
        dat = {}
        dat['ang_x_accel'] = kin.angular_acceleration.x_val
        dat['ang_y_accel'] = kin.angular_acceleration.y_val
        dat['ang_z_accel'] = kin.angular_acceleration.z_val
        dat['ang_x_vel']   = kin.angular_velocity.x_val
        dat['ang_y_vel']   = kin.angular_velocity.y_val
        dat['ang_z_vel']   = kin.angular_velocity.z_val
        dat['lin_x_accel'] = kin.linear_acceleration.x_val
        dat['lin_y_accel'] = kin.linear_acceleration.y_val
        dat['lin_z_accel'] = kin.linear_acceleration.z_val
        dat['lin_x_vel']   = kin.linear_velocity.x_val
        dat['lin_y_vel']   = kin.linear_velocity.y_val
        dat['lin_z_vel']   = kin.linear_velocity.z_val
        dat['x']   = kin.position.x_val
        dat['y']   = kin.position.y_val
        dat['z']   = kin.position.z_val
        depth = self.client.getDistanceSensorData("Distance").distance
        dat['depth'] = depth

        (pitch, roll, yaw) = airsim.utils.to_eularian_angles(kin.orientation)
        dat['pitch']   = pitch
        dat['roll']    = roll
        dat['yaw']     = yaw

        self.record_row(dat)
    
    def log_event(self, kind):
        if kind not in self.df.columns:
            self.df = self.df.assign(**{kind:np.nan}).copy()
        if np.isnan(self.df.iloc[-1][kind]):
            self.df.iloc[-1][kind] = 1
        else:
            self.df.iloc[-1][kind] += 1

    def record_row(self, data):
        for k, v in data.items():
            if k not in self.df.columns:
                self.df = self.df.assign(**{k:np.nan}).copy()
        for k, v in data.items():
            self.df.iloc[-1][k] = v

    def log_file(self):
        # print(self.df)
        self.df.to_csv(self.filename)

class CoSimPacket:
    def __init__(self):
        self.cmd = None
        self.num_bytes = None
        self.data = None

    def __str__(self):
        return "[cmd: 0x{:02X}, num_bytes: {:04d}, data: {}]".format(self.cmd, self.num_bytes, self.data)

    def init(self, cmd, num_bytes, data):
        self.cmd = cmd
        self.num_bytes = num_bytes
        self.data = data

    def decode(self, buffer):
        self.cmd = int.from_bytes(buffer[0:4], "little", signed="False")
        self.num_bytes = int.from_bytes(buffer[4:8], "little", signed="False")
        data_array = [] 
        for i in range(self.num_bytes // 4):    
            if self.cmd in INTCMDS:
                data_array.append(int.from_bytes(buffer[4 * i + 8 : 4 * i + 12],  "little", signed="False"))
            else:
                data_array.append(struct.unpack("f", buffer[4 * i + 8 : 4 * i + 12])[0])
        self.data = data_array

    def encode(self):
        buffer = self.cmd.to_bytes(4, 'little') + self.num_bytes.to_bytes(4, 'little')
        if self.num_bytes > 0:
            for datum in self.data:
                if self.cmd in INTCMDS:
                    buffer = buffer + datum.to_bytes(4, "little", signed=False)
                else:                
                    buffer = buffer + struct.pack("f", datum) 
        return buffer

class SocketThread (threading.Thread):
   def __init__(self, syn):
      threading.Thread.__init__(self)
      self.syn = syn
      self.killed = False

   def read_word(self):
        data = self.sync_conn.recv(1)
        datum = None
        if data:
            for i in range(3):
               while True:
                   datum = self.sync_conn.recv(1)
                   if datum:
                       data = data + datum
                       break
            return data
        return None

   def kill(self):
        self.killed = True

   def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.syn.sync_host, self.syn.sync_port))
            s.listen()
            self.syn.server_started = True
            self.sync_conn, self.sync_addr = s.accept()
            count = 0
            with self.sync_conn:
                print(f"sync_connected by {self.sync_addr}")
                while True:
                    if self.killed:
                        s.close()
                        raise SystemExit()
                    self.sync_conn.settimeout(0.1)
                    count = count + 1
                    if count > 100:
                        # print("Synchronizer heartbeat")
                        count = 0
                    try:
                        # cmd_data = self.sync_conn.recv(4)
                        cmd_data = self.read_word()
                        if cmd_data:
                            # self.sync_conn.setblocking(1)
                            queue = None
                            cmd = int.from_bytes(cmd_data, "little", signed="False")
                            # rxfile.write(f"{cmd}\n")
                            # print("Got command: 0x{:02X}".format(cmd))
                            if cmd > 0x80:
                                queue = self.syn.sync_rxqueue
                            else:
                                queue = self.syn.data_rxqueue
                            num_bytes = None
                            # print("Getting Num Bytes")
                            while True:
                                # num_bytes_data = self.sync_conn.recv(4)
                                num_bytes_data = self.read_word()
                                if num_bytes_data:
                                    num_bytes = int.from_bytes(num_bytes_data, "little", signed="False")
                                    # print(f"Num bytes: {num_bytes}")
                                    break
                            # rxfile.write(f"{num_bytes}\n")
                            data = []
                            for i in range(num_bytes//4):
                                # print(f"Getting Data {i}")
                                while True:
                                    if num_bytes:
                                        # datum = self.sync_conn.recv(4)
                                        datum = self.read_word()
                                        # print(f"Raw datum: {datum}")
                                        if cmd in INTCMDS:
                                            data.append(int.from_bytes(datum, "little", signed="False"))
                                        else:
                                            data.append(struct.unpack("f", datum)[0])
                                        # rxfile.write(f"{data[i]}\n")
                                        break
                            packet = CoSimPacket()
                            packet.init(cmd, num_bytes, data)
                            # print(f"Got packet: {packet}")
                            queue.append(packet)
                    except Exception as e:
                        # print(f"recv error: {e}")
                        # traceback.print_exc()
                        pass
                    if len(self.syn.txqueue) > 0:
                        packet = self.syn.txqueue.pop(0)
                        self.sync_conn.sendall(packet.encode())
                        # txfile.write(f"{packet.cmd}\n")
                        # txfile.write(f"{packet.num_bytes}\n")
                        # if packet.num_bytes > 0:
                        #     for datum in packet.data:
                        #         txfile.write(f"{datum}\n")



class Synchronizer:
    def __init__(self, host, sync_port, data_port, firesim_step = 10000, airsim_step = 10, control=None, airsim_ip=AIRSIM_IP, cycle_limit=None, vehicle="drone"):
        self.control   = control
        self.airsim_ip = airsim_ip
        self.sync_host = host
        self.data_host = host
        self.sync_port = sync_port
        self.data_port = data_port
        self.firesim_step = firesim_step
        self.airsim_step  = airsim_step
        self.packet = CoSimPacket()
        self.txqueue = []
        self.data_rxqueue = []
        self.sync_rxqueue = []

        self.cycle_limit = cycle_limit

        self.server_started = False

        print("Starting synchronizer code")
        print("Connecting to server")
        self.vehicle = vehicle
        if vehicle == "drone":
            self.client = airsim.MultirotorClient(ip=airsim_ip)
        elif vehicle == "car":
            self.client = airsim.CarClient(ip=airsim_ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.logger = CoSimLogger(self.client, cycles=self.firesim_step, frames=self.airsim_step)

        print("Connected to server")
        print("Pausing simulator...")
        self.client.simPause(True) 
    
    def run(self):
        socket_thread = SocketThread(self)
        socket_thread.start()
        start_time = time.time()
        self.send_firesim_step()
        self.control.launchStabilizer(self.airsim_ip)
        count = 0
        while True:
            if os.path.exists('reset.txt'):
                start_time = time.time()
                count = 0
                pose = self.client.simGetVehiclePose()
                try:
                    f = open('angle.txt', 'r')
                    yaw = float(f.readline()) * np.pi / 180
                    pose.orientation = airsim.utils.to_quaternion(0,0,yaw)
                except:
                    pose.orientation = airsim.utils.to_quaternion(0,0, np.pi)
                self.client.armDisarm(False)
                pose.position.x_val = 0
                pose.position.y_val = 0
                if self.vehicle == "drone":
                    pose.position.z_val = 1
                self.client.simSetVehiclePose(pose, ignore_collision=True)
                
                self.client.simContinueForFrames(1)
                self.client.armDisarm(True)
                self.control.targets['running'] = True
                self.logger = CoSimLogger(self.client, cycles=self.firesim_step, frames=self.airsim_step)
                self.logger.start()
                os.remove('reset.txt')
            if self.cycle_limit is not None and count * self.firesim_step >= self.cycle_limit:
                end_time = time.time()
                elapsed_time = end_time - start_time
                # Append-adds at last
                writestring = f"{elapsed_time}, {self.cycle_limit/elapsed_time}, {self.cycle_limit}, {self.firesim_step}, {self.airsim_step}"
                os.system(f"echo {writestring} >> sim_data_test.log")
                print(f"writestring {writestring}")
                print(f"Terminating Simulation at {elapsed_time}")
                socket_thread.kill()
                time.sleep(1)
                exit()
            if self.logger.started:
                # print("Logging...")
                self.logger.add_row(count)
                self.logger.log_true_kin()
                dat = {}
                dat['target_z'] = self.control.targets['z']   
                dat['target_x_vel'] = self.control.targets['x_vel']  
                dat['target_y_vel'] = self.control.targets['y_vel']  
                dat['target_yawrwate'] = self.control.targets['yawrate'] 
                self.logger.log_targets(dat)
            if count % 20 == 0:
                print("Stepping airsim")
                self.logger.log_file()
            #self.client.simContinueForFrames(self.airsim_step)
            self.client.simContinueForTime(self.airsim_step/100)
            if count % 20 == 0:
                print(f"Granting fsim token: {count}")
            count = count + 1
            self.grant_firesim_token()
            while True:
                if (self.client.simIsPause()):
                    break
            while len(self.data_rxqueue) > 0:
                self.process_fsim_data_packet()
        socket_thread.join()
    
    def send_firesim_step(self):
        packet = CoSimPacket()
        packet.init(CS_DEFINE_STEP, 4, [self.firesim_step])
        # print(f"Enqueuing step size: {packet}")
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())

    def grant_firesim_token(self):
        # print("Enqueuing new token")
        packet = CoSimPacket()
        packet.init(CS_GRANT_TOKEN, 0, None)
        self.txqueue.append(packet)
        #self.sync_conn.sendall(self.packet.encode())

        while True:
            if len(self.sync_rxqueue) > 0:
                self.sync_rxqueue.pop(0)
                break
            
    
    def get_firesim_cycles(self):
        packet = CoSimPacket()
        packet.init(CS_REQ_CYCLES, 0, None)
        self.txqueue.append(packet)

        while len(self.sync_rxqueue) == 0:
            pass
        response = self.sync_rxqueue.pop(0)
        return response.data[0]
    
    def send_test_firesim_data_packet(self):
        packet = CoSimPacket()

    def process_fsim_data_packet(self):
        packet = self.data_rxqueue.pop(0)
        print(f"Dequeued data packet: {packet}")
        if packet.cmd == CS_REQ_ARM:
            print("---------------------------------------------------")
            print("Arming...")
            print("---------------------------------------------------")
            pose = self.client.simGetVehiclePose()
            try:
                f = open('angle.txt', 'r')
                yaw = float(f.readline()) * np.pi / 180
                pose.orientation = airsim.utils.to_quaternion(0,0,yaw)
            except:
                pose.orientation = airsim.utils.to_quaternion(0,0, np.pi)
            self.client.armDisarm(False)
            pose.position.x_val = 0
            pose.position.y_val = 0
            if self.vehicle == "drone":
                pose.position.z_val = 1
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            
            self.client.simContinueForFrames(1)
            self.client.armDisarm(True)
            self.control.targets['running'] = True
            self.logger = CoSimLogger(self.client, cycles=self.firesim_step, frames=self.airsim_step)
            self.logger.start()
        elif packet.cmd == CS_REQ_TAKEOFF:
            print("---------------------------------------------------")
            print("Taking off...")
            print("---------------------------------------------------")
            if self.vehicle == "drone":
                self.client.takeoffAsync()
            self.control.targets['running'] = True

        elif packet.cmd == CS_RSP_WAYPOINT:
            print("---------------------------------------------------")
            print(f"Flying to: {packet.data[0]}, {packet.data[1]}, {packet.data[2]}")
            print("---------------------------------------------------")
            self.client.moveToPositionAsync(packet.data[0],packet.data[1],packet.data[2], packet.data[3])
        elif packet.cmd == CS_SET_TARGETS:
            print("---------------------------------------------------")
            print(f"z: {packet.data[0]}, x_vel: {packet.data[1]}, y_vel: {packet.data[2]}, yawrate: {packet.data[3]}")
            print("---------------------------------------------------")
            self.logger.log_event("target_req")
            self.control.targets['z']       = packet.data[0]
            self.control.targets['x_vel']   = packet.data[1]
            self.control.targets['y_vel']   = packet.data[2]
            self.control.targets['yawrate'] = packet.data[3]
        elif packet.cmd == CS_REQ_IMG:
            print("---------------------------------------------------")
            print("Got image request...")
            print("---------------------------------------------------")
            self.logger.log_event("img_req")
            rawImage = self.client.simGetImage("0", airsim.ImageType.Scene)
            # png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            cv2.imwrite('img/img.png', png)
            png = cv2.resize(png, (INPUT_DIM, INPUT_DIM))
            png_arr = png.reshape((INPUT_DIM,INPUT_DIM*3))
            print(png_arr.shape)
            print(png_arr)
            # png_arr = png.reshape((172_800))
            k = 0

            for row in png_arr:
                png_packet_arr = row.view(np.uint32).tolist()
                if k < 4:
                    print(len(png_packet_arr))
                    #print([hex(x) for x in png_packet_arr])
                    print([(i, hex(png_packet_arr[i])) for i in range(len(png_packet_arr))])
                    k += 1
                # print(png_packet_arr)
                packet = CoSimPacket()
                packet.init(CS_RSP_IMG, len(png_packet_arr)*4, png_packet_arr)
                self.txqueue.append(packet)
        elif packet.cmd == CS_REQ_DEPTH:
            print("---------------------------------------------------")
            print("Got depth request...")
            print("---------------------------------------------------")
            
            depth = self.client.getDistanceSensorData("Distance").distance
            packet = CoSimPacket()
            packet.init(CS_RSP_DEPTH, 4, [depth])
            self.txqueue.append(packet)

        else:
            pass
        



if __name__ == "__main__":
    arg_list = argparse.ArgumentParser()

    # Add arguments to the parser
    arg_list.add_argument("-a", "--Airsim-steps", type=int, default=1, help="airsim steps")
    arg_list.add_argument("-f", "--Firesim-steps", type=int, default=20000000, help="firesim steps")
    arg_list.add_argument("-c", "--Cycle-limit", type=int, default = None)
    args = vars(arg_list.parse_args())
    print(args)

    control = control_drone.IntermediateDroneApi()
    print(f"Firesim_step: {args['Firesim_steps']}")
    sync = Synchronizer(HOST, SYNC_PORT, DATA_PORT, firesim_step=args['Firesim_steps'], airsim_step=args['Airsim_steps'], cycle_limit=args['Cycle_limit'])
    sync.run()
    
    while True:
        rawImage = client.simGetImage("0", airsim.ImageType.Scene)
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_COLOR)
            png_arr = png.reshape((172_800))

        
def run_trailnet(ort_sess, img):
    img_BGR = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # im_f = img_BGR/255
    im_f = img_BGR.astype(np.float32)
    im_f = np.expand_dims(im_f,axis=0)
    # print(im_f.shape)
    im_f = im_f.transpose(0, 3,1,2)
