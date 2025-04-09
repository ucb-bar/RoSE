import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np
import airsim

class AirSimEnv(gym.Env):

    def __init__(self, *args, vehicle="drone", airsim_ip="localhost", **kwargs):
        super(AirSimEnv, self ).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example: Action is a discrete choice of 3 possibilities

        # TODO Load from config:
        self.image_dim = 256
        self.img_id = 0

        # Action
        # Interpretation of targets:
        # [z, xvel, yvel, yawrate]
        self.action_space = spaces.Dict({
            "targets": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "active": spaces.Discrete(2)
        })

        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3), dtype=np.uint8),
        })
        print(f"obs space: {self.observation_space['camera'].shape}")

        # Initialize the AirSim simulation
        print("Connecting to AirSim server")

        self.vehicle = vehicle
        if self.vehicle == "drone":
            self.client = airsim.MultirotorClient(ip=airsim_ip)
        elif self.vehicle == "car":
            self.client = airsim.CarClient(ip=airsim_ip)
        else:
            raise ValueError("AirSim only supports 'drone' and 'car' configurations.")

        # Confirm connection, enable API control to AirSim
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Ensure ensure simulator execution is pasued. Only continue on step(). 
        self.client.simPause(True)
    
    def calc_observation(self):
        # Get camera data
        raw_left = self.client.simGetImage("front_left", airsim.ImageType.Scene)
        raw_right = self.client.simGetImage("front_right", airsim.ImageType.Scene)

        left_png = cv2.imdecode(airsim.string_to_uint8_array(raw_left), cv2.IMREAD_COLOR)
        right_png = cv2.imdecode(airsim.string_to_uint8_array(raw_right), cv2.IMREAD_COLOR)

        left_grey_img = cv2.cvtColor(left_png, cv2.COLOR_BGR2GRAY)
        right_grey_img = cv2.cvtColor(right_png, cv2.COLOR_BGR2GRAY)

        left_resized_grey_img = cv2.resize(left_grey_img, (self.image_dim, self.image_dim))
        right_resized_grey_img = cv2.resize(right_grey_img, (self.image_dim, self.image_dim))

        # Save images for debugging
        cv2.imwrite(f"/scratch/iansseijelly/RoSE/deploy/hephaestus/img/left.png", left_resized_grey_img)
        cv2.imwrite(f"/scratch/iansseijelly/RoSE/deploy/hephaestus/img/right.png", right_resized_grey_img)

        # interleave left and right images row-wise
        camera_observation = np.empty((self.image_dim * 2, self.image_dim), dtype=np.uint8)
        camera_observation[0::2, :] = right_resized_grey_img
        camera_observation[1::2, :] = left_resized_grey_img

        observation = {
            "camera": camera_observation,
        }
        
        return observation
    
    def reset(self, seed=None, options=None):
        # Reset the simulator
        # ...
        pose = self.client.simGetVehiclePose()

        # TODO Replace with Config
        self.initial_x = 0
        self.initial_y = 0
        self.airsim_step = 1

        # TODO Replace with Config
        try:
            f = open('angle.txt', 'r')
            yaw = float(f.readline()) * np.pi / 180
            pose.orientation = airsim.utils.to_quaternion(0,0,yaw)
        except:
            pose.orientation = airsim.utils.to_quaternion(0,0, 0)

        self.client.armDisarm(False)
        pose.position.x_val = self.initial_x
        pose.position.y_val = self.initial_y
        if self.vehicle == "drone":
            pose.position.z_val = 1
        self.client.simSetVehiclePose(pose, ignore_collision=True)

        self.client.simContinueForFrames(1)
        self.client.armDisarm(True)
        #self.control.targets['running'] = True

        #TODO Add Logging

        initial_observation = self.calc_observation()

        # Return the initial observation
        return initial_observation, {}
    
    def step(self, action):
        # Apply the action to the simulator
        # ...
        # Get state data

        # Interpretation of targets:
        # [z, xvel, yvel, yawrate]
        z = action['targets'][0]
        xvel = action['targets'][1]
        yvel = action['targets'][2]
        yawrate = action['targets'][3]

        if action['active']:
            kin = self.client.simGetGroundTruthKinematics()
            height = kin.position.z_val
            yaw = airsim.to_eularian_angles(kin.orientation)[2] * 180/np.pi
            x_vel = kin.linear_velocity.x_val * np.cos(np.radians(yaw)) + kin.linear_velocity.y_val * np.sin(np.radians(yaw))
            y_vel = -kin.linear_velocity.x_val * np.sin(np.radians(yaw)) + kin.linear_velocity.y_val * np.cos(np.radians(yaw))
            z_vel = kin.linear_velocity.z_val
            throttle = 0.6 + 0.1*(height - z)
            if height > z and z_vel > 0:
                throttle += z_vel*0.2
            if height < z and z_vel < 0:
                throttle += z_vel*0.2
            pitch = (x - x_vel) *  0.1
            roll = (y - y_vel) * 0.2 
            yawrate = yawrate * 0.5
            if (yawrate > 0 and yvel > 0) or (yawrate < 0 and yvel < 0):
                yawrate = 0
            self.client.moveByRollPitchYawrateThrottleAsync(roll, pitch, yawrate, throttle ,0.1) 
        self.client.simContinueForTime(self.airsim_step/100)

        # Continue simulation until simulation finishes simulating the current timestep
        while True:
            if (self.client.simIsPause()):
                break

        # Get the new observation, reward, and check if the episode is done
        # ...

        observation = self.calc_observation()
        reward      = 0 
        done        = False

        # Optionally, you can provide additional info (it's not used for training)
        info = {}

        return observation, reward, done, False, info
    
    def render(self, mode='human'):
        # Render the environment
        # You can use different modes, but 'human' is the most common
        # ...
        pass

    def close(self):
        # Close resources
        # ...
        pass

    def configure_logs(self):
        pass

