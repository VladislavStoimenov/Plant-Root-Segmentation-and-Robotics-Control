import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self,  render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-1.927, -1.9105, -1.8805], dtype=np.float32), 
                                       high=np.array([2.073, 2.0895, 2.1195], dtype=np.float32), 
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1.927, -1.9105, -1.8805, -1.927, -1.9105, -1.8805], dtype=np.float32), 
                                            high=np.array([2.073, 2.0895, 2.1195, 2.073, 2.0895, 2.1195], dtype=np.float32), 
                                            dtype=np.float32)
        
        # keep track of the number of steps
        self.steps = 0
    
    def reset(self, seed=None):
        # Being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation environment
        observation = self.sim.reset(num_agents=1)

        # Validate that the observation is not None or empty
        if not observation:
            raise ValueError("Simulation reset returned an empty or invalid observation.")

        # Extract the pipette position for the first robot
        # Access the first robot's data in the observation dictionary
        first_robot_key = next(iter(observation))  # Get the first robot's key
        pipette_position = np.array(observation[first_robot_key]['pipette_position'], dtype=np.float32)

        # Set a random goal position
        self.goal_position = np.random.uniform(
            low=[-1.927, -1.9105, -1.8805],
            high=[2.073, 2.0895, 2.1195],
            size=(3,)
        ).astype(np.float32)

        # Combine pipette position with goal position for the full observation
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Reset step counter
        self.steps = 0

        return observation, {}
        
        
    def step(self, action):
        # Append a dummy drop action (0) to the action array
        action = np.append(action, 0)

        # Call the simulation's step function
        simulation_observation = self.sim.run([action])

        # Extract pipette position for the first robot
        first_robot_key = next(iter(simulation_observation))  # Get the first robot's key
        pipette_position = np.array(simulation_observation[first_robot_key]['pipette_position'], dtype=np.float32)

        # Combine pipette position with goal position
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Calculate the reward
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = -float(distance_to_goal)

        # Check termination conditions
        terminated = distance_to_goal < 0.05  # Terminate if pipette is close to the goal
        truncated = self.steps >= self.max_steps  # Terminate if step count exceeds max_steps

        # Add positive reward if the task is completed
        if terminated:
            reward += 10.0

        # Increment the step counter
        self.steps += 1

        # Return observation, reward, termination, truncation, and additional info
        info = {}
        return observation, reward, terminated, truncated, info



    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()