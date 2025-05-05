import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from ot2_gym_wrapper import OT2Env

# Initialize Weights & Biases
wandb.init(
    project="Task 11",  # Project name in W&B
    entity="235030-buas",  # Replace with your W&B team
    config={
        "algorithm": "PPO",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "n_steps": 2048,  # Rollout Steps (adjust for 2048, 4096, and 8192)
    },
)

# Create the environment
env = OT2Env(render=False, max_steps=1000)

# Create a callback to save models and log metrics
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models", name_prefix="rl_model")

# Instantiate the RL model with rollout steps (n_steps)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=wandb.config.learning_rate,
    gamma=wandb.config.gamma,
    batch_size=wandb.config.batch_size,
    n_steps=wandb.config.n_steps,  # Rollout Steps
    verbose=1,
)

# Train the model
model.learn(
    total_timesteps=200000,  # Adjust as needed for sufficient training
    callback=[WandbCallback(), checkpoint_callback],
)

# Save the trained model
model.save(f"ppo_ot2_final_{wandb.config.n_steps}")

# Close the environment
env.close()