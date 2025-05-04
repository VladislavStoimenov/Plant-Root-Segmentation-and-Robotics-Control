from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

# instantiate your custom environment
env = OT2Env(render=False, max_steps=1000) # modify this to match your wrapper class

# Assuming 'wrapped_env' is your wrapped environment instance
#check_env(env)


import gymnasium as gym
import numpy as np

# Number of episodes
num_episodes = 5

for episode in range(num_episodes):
    observation = env.reset()
    terminated = False
    truncated = False
    step_count = 0

    print(f"\nStarting Episode {episode + 1}")

    while not (terminated or truncated):
        # Take a random action
        action = env.action_space.sample()

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Debugging output
        print(f"Episode: {episode + 1}, Step: {step_count + 1}, Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")

        step_count += 1

    print(f"Episode {episode + 1} finished after {step_count} steps.")