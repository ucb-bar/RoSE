import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from mpl_toolkits.mplot3d import Axes3D


class LQREnvironment(gym.Env):
    def __init__(self):
        super(LQREnvironment, self).__init__()
        self.dt = 0.01
        self.A = np.eye(6)
        self.A[0, 1] = self.dt
        self.A[2, 3] = self.dt
        self.A[4, 5] = self.dt

        b = np.array([[0], [self.dt]])
        self.B = np.hstack((np.vstack((b, np.zeros((4,1)))), np.vstack((np.zeros((2,1)), b, np.zeros((2,1)))), np.vstack((np.zeros((4,1)), b))))

        self.state = None  # To be initialized later
        self.steps_beyond_done = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # Three continuous actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Six continuous states

    def reset(self):
        self.state = np.array([-1, 0.1, 0, 0.5, 1, 0.3])  # Example initial condition
        self.steps_beyond_done = 0
        return self.state

    def step(self, action):
        state = self.state
        action = np.clip(action, -1, 1)  # Ensure actions are within acceptable bounds
        next_state = self.A @ state + self.B @ action

        self.state = next_state.flatten()
        self.steps_beyond_done += 1  # Keeping track of number of steps

        done = self.steps_beyond_done > 100  # Example condition for episode to be over

        # You need to define the reward function based on your requirements.
        reward = 0  # Placeholder; this needs to be properly defined

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # This function should implement the rendering of the environment.
        # For simplicity, it is not implemented in this example.
        pass

# def dlqr(A, B, Q, R):
#     # LQR parameters
#     dt = 0.1
#     A1 = np.array([[1]])
#     B1 = np.array([[dt]])
#     Q1 = np.array([[100]])  # 2D array
#     R1 = np.array([[1]])    # 2D array
#     K1 = dlqr(A1, B1, Q1, R1)

#     Q2 = np.array([[100, 0], [0, 100]])
#     R2 = 1
#     A2 = np.array([[1, dt], [0, 1]])
#     B2 = np.array([[0], [dt]])
#     K2 = dlqr(A2, B2, Q2, R2)

#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

# Test the environment

# ... [Your LQREnvironment class code here] ...

# Discrete-time LQR implementation
def dlqr(A, B, Q, R):

    # Ensure A, B are two-dimensional arrays
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    
    # Solve the discrete-time algebraic Riccati equation
    X = solve_discrete_are(A, B, Q, R)
    
    # Compute the LQR gain
    # We need to ensure matrix multiplication is valid, handle both 1D and 2D cases
    if B.ndim == 2:
        K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
    else:
        K = np.linalg.inv((B.T @ X @ B).reshape(1, 1) + R) @ (B.T @ X @ A)

    return -K

# LQR parameters

# Test the environment
env = LQREnvironment()
x_star = np.array([0, 0, 0, 2.0, 0, 0])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



x_star = np.array([0, 0, 0, 2.0, 0, 0])

for i_episode in range(5):
    positions = []  # List to store (x, y, z) positions for this episode
    observation = env.reset()

    for t in range(100):
        # Store the position information
        x, y, z = observation[0], observation[2], observation[4]
        positions.append((x, y, z))

        # Compute control law using your LQR controller
        # TODO: Replace state with appropriate obs
        state = observation.reshape((6, 1))  # Reshape to column vector
        u = np.squeeze(np.array([
            np.clip(K2 @ (x_star[0:2] - observation[0:2]), -1, 1),
            np.clip(K1 * (x_star[3] - observation[3]), -1, 1),
            np.clip(K2 @ (x_star[4:6] - observation[4:6]), -1, 1)
        ]))

        # Step the environment
        # TODO: Update (n) steps with the same u
        
        observation, reward, done, info = env.step(u)
        if done:
            print(f"Episode finished after {t + 1} timesteps")
            break

    # Plotting the trajectory for this episode
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o')  # You can adjust colors and transparency here

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Agent\'s Trajectory over Multiple Episodes')
plt.show()

