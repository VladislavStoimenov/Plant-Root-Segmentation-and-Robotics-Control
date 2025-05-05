# Robot Simulation Environment

This `README.md` provides instructions for setting up the simulation environment, a list of dependencies, and details about the working envelope of the pipette.

---

## Environment Setup

Follow these steps to set up the simulation environment:

1. **Install Python**:
    Ensure you have Python 3.7 or later installed. 

2. **Clone the Repository**:
   Download or clone the repository containing the simulation files:

    git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git
    cd Y2B-2023-OT2_Twin

3. **Install Required Libraries**:
   Install the dependencies using `pip`:

    pip install pybullet

4. **Run the Simulation**:
   Use the provided Jupyter Notebook (`task_9.ipynb`) to run the simulation:
   
    jupyter notebook  - `task_9.ipynb`

## Dependencies

The simulation requires the following Python library:

    `pybullet`: Physics simulation and rendering engine.

To install all dependencies, run:

    pip install pybullet
         


## Working Envelope of the Pipette

The pipette's working envelope defines the maximum range of movement in the simulation environment. It is a cube with the following boundaries:

- **X-axis**: `[-0.187, 0.253]`
- **Y-axis**: `[-0.1705, 0.2199]`
- **Z-axis**: `[0.1196, 0.2905]`

### Corner Points of the Envelope
The envelope consists of 8 corners, defined as follows:

| Corner | X Coordinate | Y Coordinate | Z Coordinate |
|--------|--------------|--------------|--------------|
| 1      | -0.187       | -0.1705      | 0.1196       |
| 2      | -0.187       | -0.1705      | 0.2905       |
| 3      | -0.187       |  0.2199      | 0.1694       |
| 4      | -0.187       |  0.2199      | 0.2895       |
| 5      |  0.2531      | -0.1705      | 0.1695       |
| 6      |  0.2531      | -0.1705      | 0.2895       |
| 7      |  0.2531      |  0.2195      | 0.1685       |
| 8      |  0.2531      |  0.2195      | 0.2895       |


## Running the Simulation

To move the pipette to all 8 corners of the working envelope, use the provided Python code or Jupyter Notebook (`task_9.ipynb`). The simulation will sequentially:

1. Move the pipette to each corner.

2. Log the pipette's position after each movement.

3. Render the environment (if enabled).

Follow these steps:

- Open `task_9.ipynb` in Jupyter Notebook.

- Execute the code cells to initialize and run the simulation.

- Observe the pipette's movements and logged states in the terminal or output.

## Troubleshooting

Simulation Not Rendering:

- Ensure that the render parameter is set to True when initializing the Simulation class.

Dependency Issues:
- Double-check the installed libraries and their versions using: `pip list`

Python Version:
- Verify that your Python version meets the minimum requirements (3.7 or later).

Unexpected Behavior:
- Check the code logic, especially the coordinates for corner points and the action definitions.

For further assistance, refer to the PyBullet documentation or open an issue in the repository.

## Additional Notes

- Always close the simulation properly using the `sim.close()` method to prevent resource leaks.

- Modify the `num_agents` parameter in the Simulation class to simulate multiple pipettes if needed.

- Customize the working envelope and movement patterns by adjusting the `x_limits`, `y_limits`, and `z_limits` variables.  








# Reinforcement Learning with Custom OT2 Gym Environment

This repository contains Python scripts for training and testing reinforcement learning (RL) models in a custom simulation environment. The project uses a custom OpenAI Gym environment for controlling a robotic system, enabling fine-grained movement and goal-based tasks.

---

## File Overview

### 1. `ot2_gym_wrapper.py`
- **Purpose**: Implements a custom OpenAI Gym environment, `OT2Env`.
- **Features**:
  - Simulates a robotic system with configurable agent movements along x, y, and z axes.
  - Provides action and observation spaces tailored for the task.
  - Includes reward calculation based on the agent's distance to a predefined goal.
  - Supports termination conditions based on goal proximity or maximum step count.
  - Extends flexibility through rendering and custom initialization options.
- **Dependencies**: `gymnasium`, `numpy`, and `sim_class` (for simulation).

---

### 2. `train.py`
- **Purpose**: Trains a reinforcement learning model using the Proximal Policy Optimization (PPO) algorithm.
- **Features**:
  - Integrates ClearML for experiment tracking and remote task execution.
  - Uses Weights & Biases (WandB) for logging and visualization.
  - Configurable via command-line arguments for hyperparameter tuning (e.g., learning rate, batch size, timesteps).
  - Periodically saves trained models during training.
  - Ensures results reproducibility by managing logging directories and seeds.
- **Dependencies**:
  - `stable-baselines3` (for PPO algorithm),
  - `wandb` (for experiment logging),
  - `clearml` (for remote execution),
  - `argparse` (for CLI argument parsing).

---

### 3. `test_model.py`
- **Purpose**: Tests pre-trained PPO models and determines the best performing model.
- **Features**:
  - Loads and evaluates multiple saved PPO models.
  - Runs episodes in the `OT2Env` environment and calculates rewards.
  - Supports reproducibility with fixed seeds and predefined goal positions.
  - Identifies the model with the highest average reward.
- **Dependencies**:
  - `stable-baselines3` (for model loading and prediction),
  - `numpy` (for mathematical operations).

---

### 4. `PID.py`
- **Purpose**: Implements a PID controller for controlling system dynamics.
- **Features**:
  - Provides a configurable proportional-integral-derivative (PID) controller.
  - Includes methods to compute outputs based on current error and past states.
  - Designed for flexible integration with other modules.
- **Dependencies**: None (pure Python).

---

### 5. `PID_test.py`
- **Purpose**: Demonstrates and tests the functionality of the `PIDController` class.
- **Features**:
  - Utilizes `PIDController` to control a robotic pipette in the `OT2Env` environment.
  - Visualizes performance with matplotlib.
  - Includes debugging outputs for step-by-step evaluation of positions and actions.
  - Demonstrates goal-reaching using PID control.
- **Dependencies**:
  - `numpy` (for numerical operations),
  - `matplotlib` (for visualization),
  - `ot2_gym_wrapper` (custom environment),
  - `PID.py` (PID controller).

---

## Getting Started

### Prerequisites
- Python 3.8+
- Install required libraries:
  ```bash
  pip install gymnasium stable-baselines3 wandb clearml numpy matplotlib
  ```

### Training a Model
1. Customize hyperparameters using the command-line interface:
   ```bash
   python train.py --learning_rate 0.0001 --batch_size 128 --timesteps 1000000
   ```
2. Monitor training via Weights & Biases or ClearML dashboards.

### Testing Models
- Run the test script to evaluate pre-trained models:
  ```bash
  python test_model.py
  ```

### Running PID Test
- Execute the PID controller test script to observe its performance:
  ```bash
  python PID_test.py
  ```
- The script generates plots for each axis (X, Y, Z) showing position and action over time.

### Customization
- Modify the `OT2Env` implementation in `ot2_gym_wrapper.py` to adapt the simulation environment.
- Update model paths in `test_model.py` for evaluation.
- Adjust PID gains and parameters in `PID_test.py` to tune controller performance.

---

## Project Structure
```
.
├── ot2_gym_wrapper.py  # Custom OpenAI Gym environment
├── train.py            # Training script using PPO
├── test_model.py       # Model testing and evaluation script
├── PID.py              # PID controller implementation
├── PID_test.py         # PID controller testing and visualization script
```

---