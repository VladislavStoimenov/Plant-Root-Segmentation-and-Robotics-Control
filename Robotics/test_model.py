#  Import the necessary libraries and custom environment
from ot2_gym_wrapper import OT2Env  
from stable_baselines3 import PPO
import numpy as np

# Define the parameters for testing the models
MODEL_PATHS = ["model_4096.zip", "model_8192.zip", "model_2048.zip"]  # List of paths to the saved PPO models
NUM_EPISODES = 10  # Number of episodes to test each model
FIXED_SEED = 42  # Fixed seed for reproducibility
FIXED_GOAL_POSITION = np.array([0.22, 0.22, 0.22])  # Define a fixed goal position

# Define a function to test the pre-trained PPO models
def test_ppo_model(model_paths, num_episodes=10):
    """
    Tests multiple pre-trained PPO models using a custom OT2 environment and determines the best performing model.

    Parameters:
        model_paths (list): List of paths to the saved PPO models.
        num_episodes (int): Number of episodes to run for each model.
    """
    # Initialize variables to track the best model and its average reward
    best_model = None
    best_average_reward = -np.inf

    # Iterate over the list of model paths
    for model_path in model_paths:
        # Load the model
        model = PPO.load(model_path)
        
        # Initialize the custom environment
        env = OT2Env(render=False)
        # list to store rewards for each episode
        all_rewards = []

        # for loop to run the episodes
        for episode in range(num_episodes):
            # Reset the environment with a fixed seed
            reset_output = env.reset(seed=FIXED_SEED)
            # Extract observation from reset output
            obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            # Set the goal position for the environment
            env.goal_position = FIXED_GOAL_POSITION  # Set the fixed goal position
            done = False
            episode_reward = 0

            # while loop to run the episode
            while not done:
                # Predict the action using the loaded model
                action, _states = model.predict(obs, deterministic=True)
                # Clip the action to the valid action space
                obs, reward, terminated, truncated, info = env.step(action)
                # Update the episode reward
                episode_reward += reward

                # Determine if the episode is done
                done = terminated or truncated
            # Append the episode reward to the list
            all_rewards.append(episode_reward)
            print(f"Model {model_path} - Episode {episode + 1}: Reward = {episode_reward}")
        
        # Close the environment after testing
        env.close()
        # Calculate the average reward for the model
        average_reward = np.mean(all_rewards)
        print(f"Model {model_path} - Average Reward over {num_episodes} episodes: {average_reward}")

        # Determine if this model is the best so far
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            best_model = model_path
    # Print the best model and its average reward
    print(f"Best Model: {best_model} with Average Reward: {best_average_reward}")

# Run the testing function
test_ppo_model(MODEL_PATHS, NUM_EPISODES)