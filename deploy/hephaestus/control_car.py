import airsim
import numpy as np
import os
import tempfile
import pprint
#import cv2
import sys

import threading
import time

def stabilize(ip, targets):
	client = airsim.CarClient(ip=ip)
	client.confirmConnection()
	client.enableApiControl(True)
	car_controls = airsim.CarControls()
	while True:
		# get kin and car state information
		# kin = client.simGetGroundTruthKinematics()
		car_state = client.getCarState()

		# set throttle
		if(car_state.speed < targets['x_vel']):
			car_controls.throttle = 1.0
		else:
			car_controls.throttle = 0.0

		# set steering 
		car_controls.steering = targets['yawrate'] * -0.4

		# print("Throttle: {:2f}, Steering: {:2f}".format(car_controls.throttle, car_controls.steering))
		client.setCarControls(car_controls) 
	
class IntermediateCarApi: 
	
	def __init__(self):
		self.global_heading = 0
		self.targets = {
			"z" : -0.5,
			"x_vel" : 0,
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
	
