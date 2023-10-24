import os
import csv
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


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
        self.observations = []
        self.actions = []


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
        

        self.fig, self.axs = None, None
        
        print(f"Packet counts: {self.packet_counts}")

    def log_data(self, obs, action):
        # Convert observation and action to flat list for CSV logging
        obs_list = self._to_flat_list(obs)
        action_list = self._to_flat_list(action)
        
        self.observations.append(obs_list)
        self.actions.append(action_list)

        if self.fig is None:
            obs_arr = np.array(self.observations)
            action_arr = np.array(self.actions)

            num_obs = obs_arr.shape[1]
            num_actions = action_arr.shape[1]
            self.fig, self.axs = plt.subplots(num_obs + num_actions, 1, figsize=(10, 2 * (num_obs + num_actions)))

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
        self.observations = []
        self.actions = []
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


    def save_video(self, max_fps=120):
        if len(self.frames) == 0:
            return

        height, width, layers = self.frames[0].shape
        size = (width, height)

        # Calculate the original FPS
        original_fps = round(1/(self.firesim_period*2))

        # Determine if we need to skip frames to reduce the FPS
        if original_fps > max_fps:
            skip_frames = round(original_fps / max_fps)
            fps = max_fps
        else:
            skip_frames = 1
            fps = original_fps

        out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i, frame in enumerate(self.frames):
            if i % skip_frames == 0:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
    
    def display(self):
        if len(self.observations) == 0 or len(self.actions) == 0:
            print("No data to display.")
            return
            
        time_points = np.arange(0, len(self.observations) * self.firesim_period, self.firesim_period)
        obs_arr = np.array(self.observations)
        action_arr = np.array(self.actions)

        for i, ax in enumerate(self.axs):
            ax.clear()  # Clear current axis before updating
            
            if i < num_obs:
                ax.plot(time_points, obs_arr[:, i])
                ax.set_title(f'Observation {i}')
            else:
                ax.plot(time_points, action_arr[:, i - num_obs], 'r')
                ax.set_title(f'Action {i - num_obs}')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.pause(0.001)  # Use plt.pause instead of plt.show. This will redraw the updated figure


        # Render images
        frame = self.env.render()
        cv2.imshow(f'RoSE Frame', frame)  # Convert RGB to BGR
        cv2.waitKey(1)  # 1 ms delay to give time for the image to be drawn


    def close(self):
        # Close CSV file
        self.csv_file.close()

        # Save the video
        self.save_video()