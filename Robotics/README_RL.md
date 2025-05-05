# Reinforcement Learning Environment Setup and Training

This guide explains how to set up the environment and train three reinforcement learning models using PPO. The models vary by the `n_steps` parameter and are evaluated based on their performance.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- `pip` (Python package manager)
- Necessary Python libraries (see below)

### Required Python Libraries

Install the required dependencies using the following command:

```bash
pip install gymnasium numpy stable-baselines3 wandb clearml
```

## Setting Up the Environment

1. **Custom Environment**: The RL models are trained using a custom OpenAI Gym environment defined in `ot2_gym_wrapper.py`.

    - This environment simulates a robot's pipette movements, with the goal of reaching a specific target position.
    - **Action Space**: Continuous values representing movements along `x`, `y`, and `z`.
    - **Observation Space**: Continuous values of the robot's current and goal positions.

2. **Environment Setup**:

    - Import the environment in your training or testing scripts as shown in `train.py` or `test_model.py`.

3. **Reset and Step Functions**:
    - Use `reset()` to initialize the environment.
    - Use `step(action)` to simulate an action and receive the next state, reward, and termination status.

## Training the Models

To train the models, use `train.py`. The script allows hyperparameter tuning via command-line arguments. The models are trained with the following key hyperparameters:

- **Learning Rate**: `0.0001`
- **Batch Size**: `128`
- **Timesteps**: `1,000,000`
- **Gamma**: `0.99`
- **Clip Range**: `0.02`
- **Entropy Coefficient**: `0.1`
- **Value Function Coefficient**: `1`
- **n_steps**: `4096`, `8192`, or `2048` (different for the three models)

### Usage

Run the training script as follows:

```bash
python train.py --learning_rate 0.0001 --batch_size 128 --n_steps 4096 --n_epochs 10 --timesteps 1000000 --gamma 0.99 --clip_range 0.02 --ent_coef 0.1 --vf_coef 1
```

Change the value of `--n_steps` to `8192` or `2048` to train the other models.

The models will be saved in the `models_last/` directory with unique identifiers.

## Testing the Models

Use `test_model.py` to evaluate the trained models. Update the `MODEL_PATHS` variable to include the paths to your saved models:

```python
MODEL_PATHS = ["model_4096.zip", "model_8192.zip", "model_2048.zip"]
```

Run the script:

```bash
python test_model.py
```

The script evaluates each model over a fixed number of episodes (default: 10) and reports the average rewards.

## Example Results

- `n_steps = 4096`: Average Reward = 47.932625432608816
- `n_steps = 8192`: Average Reward = -2.699725305713808
- `n_steps = 2048`: Average Reward = 47.922632844917416

## Why `n_steps = 4096` is the Best Choice

After testing the models with different `n_steps` values (2048, 4096, and 8192), the model with `n_steps = 4096` achieved the best performance. Here are the key reasons:

1. **Balanced Learning Stability**: With `n_steps = 4096`, the model collects a sufficient number of samples per update to ensure stable gradient estimates without introducing excessive variance.

2. **Computational Efficiency**: Compared to `8192`, which requires more memory and computational time per update, `4096` strikes a better balance between training speed and performance.

3. **Convergence Rate**: The model with `n_steps = 4096` converged faster during training compared to the others. This indicates that the number of steps aligns well with the dynamics of the custom environment.

4. **Reward Consistency**: In testing, the `4096` model consistently achieved higher average rewards across episodes, indicating better generalization and robustness.

Thus, `n_steps = 4096` is recommended for future training in this environment.

## Notes

- Modify the `render` parameter in `OT2Env` to visualize the environment during testing.
- The `test_wrapper.py` script demonstrates the environment's functionality using random actions.

## Additional Information

- **Weights & Biases** (`wandb`): Integrated for experiment tracking.
- **ClearML**: Used for remote task execution and resource tracking.
