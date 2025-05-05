# import required packages and custom environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

# Define the custom OpenAI Gym environment
class OT2Env(gym.Env):
    # Define the initialization function
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # Action space: 3 continuous values representing movements along x, y, z within [-1.2, 1.2]
        self.action_space = spaces.Box(low=-1.2, high=1.2, shape=(3,), dtype=np.float32)

        # Observation space: 6 continuous values representing x, y and z limits from task 9
        self.observation_space = spaces.Box(low=np.array([-1.927, -1.9105, -1.8805, -1.927, -1.9105, -1.8805]),
                                            high=np.array([2.073, 2.0895, 2.1195, 2.073, 2.0895, 2.1195]),
                                            dtype=np.float32)

        # Keep track of the number of steps
        self.steps = 0
        
    # Define the reset function
    def reset(self, seed=None):
        # Select seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Set a random goal position for the agent within the working envelope of the petitte
        self.goal_position = np.random.uniform(low=[-0.187, -0.1705, 0.1695],
                                               high=[0.253, 0.2195, 0.2896], size=3)

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # Process the observation: get pipette position and append the goal position
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        return observation, {}
        
    # Define the step function
    def step(self, action):
        # Append 0 for the drop action since we only control position
        action = np.append(action, 0.0)

        # Call the environment step function
        observation = self.sim.run([action])  # Pass the action as a list

        # Process the observation: get pipette position and append the goal position
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        # Calculate the reward as a negative function of the distance
        reward = -distance_to_goal * 0.5

        # Check if the task is complete (distance below a threshold)
        if distance_to_goal < 0.001:  # Threshold based on the required precision
            terminated = True
            reward += 50  # Positive reward for completing the task
        else:
            terminated = False

        # Check if the episode should be truncated (max steps exceeded)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Create an empty info dictionary
        info = {}  

        # Increment the number of steps
        self.steps += 1
        # Return the observation, reward, termination status, truncation status, and info
        return observation, reward, terminated, truncated, info

    # Define the render function
    def render(self, mode='human'):
        pass
    # Define the close function
    def close(self):
        self.sim.close()