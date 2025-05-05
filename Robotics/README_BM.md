
# Benchmarking Analysis

## Overview
This project involves benchmarking tasks aimed at evaluating the performance of RL (Reinforcement Learning) and PID (Proportional-Integral-Derivative) controllers in achieving precise goal positions within a simulated environment.

## Setup
The setup involves the following steps:
1. **Environment Initialization**:
   The `OT2Env` environment is initialized with rendering disabled. This simulates a robotic environment where goal positions are set, and controllers guide the robot to achieve them.

2. **PID Controller Configuration**:
   PID controllers for the x, y, and z axes are configured with the following parameters:
   - **Kp (Proportional Gain)**: 27, 45, 18
   - **Ki (Integral Gain)**: 0.5, 0.1, 2.8
   - **Kd (Derivative Gain)**: 2.8, 2, 7.8

   These parameters are tuned to optimize the control performance.

3. **Library Imports**:
   Required libraries such as NumPy, Matplotlib, and a custom environment library are imported. Ensure these dependencies are installed before running the notebook.

Example setup code:
```python
# Load required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from some_environment_module import OT2Env
from some_pid_library import PIDController

# Setup environment
env = OT2Env(render=False)
pid_x = PIDController(27, 0.5, 2.8, setpoint=1)
pid_y = PIDController(45, 0.1, 2, setpoint=1)
pid_z = PIDController(18, 2.8, 7.8, setpoint=1)
```

## Benchmarking Details
The benchmarking compares the performance of two controllers:

- **RL Controller**:
  Utilizes a pre-trained reinforcement learning model to predict actions based on the current state of the environment.

- **PID Controller**:
  Uses classical control theory to compute control signals based on proportional, integral, and derivative terms.

Example benchmarking code:
```python
def benchmark_rl_controller(env, goal_positions):
    total_time = 0
    total_accuracy = 0
    for goal_pos in goal_positions:
        obs, _ = env.reset()
        env.goal_position = goal_pos[:3]
        start_time = time.time()
        while True:
            action, _ = rl_model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        total_time += time.time() - start_time
        total_accuracy += np.linalg.norm(goal_pos[:3] - obs[:3])
    return total_time / len(goal_positions), total_accuracy / len(goal_positions)
```

## Results
Key results and observations from the benchmarking are summarized below:

- **RL Controller**:
  - Average Execution Time: **5.32 seconds**
  - Average Distance from Goal Position: **1 mm**

- **PID Controller**:
  - Average Execution Time: **4.85 seconds**
  - Average Distance from Goal Position: **1.24 mm**

Comparison plots:
- Execution Time: PID Controller is slightly faster than RL.
- Accuracy: PID Controller achieves better accuracy than RL.

## How to Run
1. Install dependencies using `requirements.txt` or `conda`.
2. Execute the notebook step-by-step to replicate the benchmarking process:
    - Run setup cells to configure the environment.
    - Execute benchmarking code blocks.
3. Review the results in the notebook for detailed insights.

## Future Work
- Explore additional goal positions and scenarios.
- Optimize the RL model for faster inference and improved accuracy.
- Implement parallelized evaluations to enhance benchmarking speed.
