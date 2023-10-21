import os
import csv
import numpy as np
import cv2
import time

class GymLogger:
    def __init__(self, env, firesim_period, start_time, packet_bindings, log_dir='./logs', log_filename='data.csv', video_filename='rendering.avi'):
        self.env = env
        self.firesim_period = firesim_period
        self.start_time = start_time
        self.packet_bindings = packet_bindings
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)
        self.video_path = os.path.join(log_dir, video_filename)
        self.frames = []
        self.sim_time = 0.0
        self.resets = 0

        # Create the logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Initialize CSV logging
        self.csv_file = open(self.log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.header_written = False
        
        # Define packet tracking
        self.packet_counts = {}
        for key in self.packet_bindings.keys():
            self.packet_counts[key] = 0
        
        print(f"Packet counts: {self.packet_counts}")

    def log_data(self, obs, action):
        # Convert observation and action to flat list for CSV logging
        obs_list = self._to_flat_list(obs)
        action_list = self._to_flat_list(action)

        # Calculate times
        self.sim_time += self.firesim_period
        time_list = [self.sim_time, time.time() - self.start_time]
        
        # Put resets into lists
        reset_list = [self.resets]
        
        # Put packet counts in list
        packet_list = []
        for key in self.packet_bindings.keys():
            packet_list.append(self.packet_counts[key])
            

        # Write header for CSV if not yet written
        if not self.header_written:
            headers = ['sim_time', 'wall_time', 'resets'] + [f'obs_{i}' for i in range(len(obs_list))] + [f'action_{i}' for i in range(len(action_list))] + [self.packet_bindings[key]['name'] for key in self.packet_bindings.keys()]
            self.csv_writer.writerow(headers)
            self.header_written = True

        # Write observation and action data
        self.csv_writer.writerow(time_list + reset_list + obs_list + action_list + packet_list)
        
        # Reset packet counts for next step
        for key in self.packet_bindings.keys():
            self.packet_counts[key] = 0
            
    def count_packet(self, packet_id):
        self.packet_counts[packet_id] += 1
        
    def count_reset(self):
        self.resets += 1

    def _to_flat_list(self, item):
        if isinstance(item, np.ndarray):
            return item.flatten().tolist()
        elif isinstance(item, dict):
            return [v for k, v in item.items()]
        elif isinstance(item, (list, tuple)):
            return list(item)
        else:
            return [item]

    def log_rendering(self):
        frame = self.env.render()
        self.frames.append(frame)

    def save_video(self):
        if len(self.frames) == 0:
            return

        # Define codec using VideoWriter_fourcc and create VideoWriter object
        height, width, layers = self.frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for frame in self.frames:
            # OpenCV expects colors in BGR format, so convert from RGB to BGR
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

    def close(self):
        # Close CSV file
        self.csv_file.close()

        # Save the video
        self.save_video()