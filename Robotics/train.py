# Import the necessary libraries and environment
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import time
from ot2_gym_wrapper import OT2Env
import os 
from clearml import Task
import argparse
import wandb

# Initialize ClearML Task
task = Task.init(
    project_name='Mentor Group D/Group 2',
    task_name='Experiment_6_235030'
)
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name='default')

# Parse the arguments for learning process
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_steps', type=int, default=4096)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--timesteps', type=int, default=1000000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--clip_range', type=float, default=0.02)
parser.add_argument('--ent_coef', type=float, default=0.1)
parser.add_argument('--vf_coef', type=float, default=1)
args = parser.parse_args()

# Initialize Weights & Biases
os.environ['WANDB_API_KEY']='4b31bcfdf4d66049adafff1725fe3c970f0ff013' # API key for Weights & Biases
run = wandb.init(project='Experiment3', sync_tensorboard=True) # Initialize the Weights & Biases run

# Ensure necessary directories exist
os.makedirs(f"models/{run.id}_last", exist_ok=True)
os.makedirs(f"models_last/{run.id}_last", exist_ok=True)

# Create the environment
env = OT2Env(render=False, max_steps=1000)

# Create the model
model = PPO('MlpPolicy', env, verbose=1, # Create the PPO model with the MlpPolicy
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}",) # Log the tensorboard data to the correct folder

# Create the WandB callback
wandb_callback = WandbCallback(
    model_save_freq=10000, # Save the model every 10000 timesteps
    model_save_path=f'models/{run.id}', # Save the model to the correct folder
    verbose=2
)


# Train the model
time_steps = 5000000 #  define the number of timesteps to train the model for
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models_last/{run.id}_last/{time_steps*(i+1)}")

# Final save of the model
final_model_path = f"models_last/{run.id}_last/final_model"
model.save(final_model_path)

# Close the environment and end the WandB run
env.close()
wandb.finish()