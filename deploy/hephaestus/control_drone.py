import airsim
import numpy as np
import os
import tempfile
import pprint
#import cv2
import sys

import threading
def stabilize(ip, targets):
	client = airsim.MultirotorClient(ip=ip)
	client.confirmConnection()
	client.enableApiControl(True)

	roll = 0
	pitch = 0.00
	yawrate = 0
	throttle = 0.6
	while True:
		kin = client.simGetGroundTruthKinematics()
		height = kin.position.z_val
		yaw = airsim.to_eularian_angles(kin.orientation)[2] * 180/np.pi
		x_vel = kin.linear_velocity.x_val * np.cos(np.radians(yaw)) + kin.linear_velocity.y_val * np.sin(np.radians(yaw))
		y_vel = -kin.linear_velocity.x_val * np.sin(np.radians(yaw)) + kin.linear_velocity.y_val * np.cos(np.radians(yaw))
		z_vel = kin.linear_velocity.z_val
		# print("yaw: {}, x_vel: {}, y_vel: {}".format(yaw, x_vel, y_vel))
		# targets['yawrate'] = np.sign(targets['yawrate']) * 2
		# targets['y_vel'] = np.sign(targets['y_vel'])
		throttle = 0.6 + 0.1*(height - targets["z"])
		if height > targets["z"] and z_vel > 0:
			throttle += z_vel*0.2
		if height < targets["z"] and z_vel < 0:
			throttle += z_vel*0.2
		pitch = (targets['x_vel'] - x_vel) *  0.1
		roll = (targets['y_vel'] - y_vel) * 0.2 # formal 0.4
		yawrate = targets['yawrate'] * 0.5 # former 0.4
		if (targets['yawrate'] > 0 and targets['y_vel'] > 0) or (targets['yawrate'] < 0 and targets['y_vel'] < 0):
			yawrate = 0
		# yawrate -= targets['y_vel'] * 0.3
		# print("Throttle: {:2f}, Z: {:2f}, t_y_vel: {:2f}, t_yr: {:2f}".format(throttle, height, targets['y_vel'], targets['yawrate']))
		client.moveByRollPitchYawrateThrottleAsync(roll, pitch, yawrate, throttle ,0.1) 

# def stabilize(ip, targets):
# 	client = airsim.MultirotorClient(ip=ip)
# 	client.confirmConnection()
# 	client.enableApiControl(True)
# 
# 	roll = 0
# 	pitch = 0.00
# 	yawrate = 0
# 	throttle = 0.6
# 	while True:
# 		kin = client.simGetGroundTruthKinematics()
# 		height = kin.position.z_val
# 		yaw = airsim.to_eularian_angles(kin.orientation)[2] * 180/np.pi
# 		x_vel = kin.linear_velocity.x_val * np.cos(np.radians(yaw)) + kin.linear_velocity.y_val * np.sin(np.radians(yaw))
# 		y_vel = -kin.linear_velocity.x_val * np.sin(np.radians(yaw)) + kin.linear_velocity.y_val * np.cos(np.radians(yaw))
# 		z_vel = kin.linear_velocity.z_val
# 		# print("yaw: {}, x_vel: {}, y_vel: {}".format(yaw, x_vel, y_vel))
# 		throttle = 0.6 + 0.1*(height - targets["z"])
# 		if height > targets["z"] and z_vel > 0:
# 			throttle += z_vel*0.2
# 		if height < targets["z"] and z_vel < 0:
# 			throttle += z_vel*0.2
# 		pitch = (targets['x_vel'] - x_vel) * 0.1
# 		roll = (targets['y_vel'] - y_vel) * 0.1
# 		yawrate = targets['yawrate'] *0.15
# 		if (targets['yawrate'] > 0 and targets['y_vel'] > 0) or (targets['yawrate'] < 0 and targets['y_vel'] < 0):
# 			yawrate = 0
# 		yawrate -= targets['y_vel'] * 0.3
# 		# print("Throttle: {:2f}, Z: {:2f}, t_y_vel: {:2f}, t_yr: {:2f}".format(throttle, height, targets['y_vel'], targets['yawrate']))
# 		if targets["running"]:
# 			client.moveByRollPitchYawrateThrottleAsync(roll, pitch, yawrate, throttle ,0.1) 
		
class IntermediateDroneApi: 
	
	def __init__(self):
		self.global_heading = 0
		self.targets = {
			"z" : -0.5,
			"x_vel" : 1,
			"y_vel"  : 0,
			"yawrate": 0,
			"running": False
		}

	# (yaw, timeout_sec=3e+38, margin=5, vehicle_name='')
	def rotateToYaw(self, client, theta):
		angle = client.simGetGroundTruthKinematics()
		
		yaw = airsim.to_eularian_angles(angle.orientation)[2] * 180/np.pi
		print("Yaw: {}\t Theta: {}".format(yaw,  yaw+theta))
		client.rotateToYawAsync(yaw + theta).join()
		# self.global_heading = self.global_heading + theta

	    #def moveByManualAsync(self, vx_max, vy_max, z_min, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
		# def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
	def moveInAngle(self, client, theta, speed, time):

		kin = client.simGetGroundTruthKinematics()
		
		height = kin.position.z_val
		print("height: {}".format(height))
		yaw = airsim.to_eularian_angles(kin.orientation)[2] * 180/np.pi
		heading = yaw + theta
		print("Heading: {}".format(heading))
		vx = np.cos(heading*np.pi/180) * speed
		vy = np.sin(heading*np.pi/180) * speed
		vz = -0.7 - height

		# client.rotateToYawAsync(yaw + theta).join()

		client.moveByVelocityAsync(vx, vy, vz, time, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()

	def launchStabilizer(self, ip): 
		stabilizer = threading.Thread(target=stabilize, args=(ip, self.targets,))
		stabilizer.start()
	
    # client.moveByRollPitchYawrateThrottleAsync(0,0.01,0,0.6,1) 
	
