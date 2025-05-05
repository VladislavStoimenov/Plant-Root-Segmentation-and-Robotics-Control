# import required libraries
import time
import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
import pandas as pd

# Constants
PLATE_SIZE_MM = 150  # Physical size of the plate in mm
PLATE_SIZE_PIXELS = 2796  # Size of the plate in pixels (image dimensions)
PLATE_POSITION_ROBOT = np.array([0.10775, 0.088 - 0.026, 0.057])  # Adjusted position of the plate's top-left corner in the robot's coordinate system
INOCULATION_DEPTH = -0.3  # Depth to go down for inoculation (relative to z-axis)

# Function to calculate the conversion factor between pixel space and mm space
def calculate_conversion_factor(plate_size_mm, plate_size_pixels):
    # returns the conversion factor between pixel space and mm space
    return plate_size_mm / plate_size_pixels

# Function to convert pixel coordinates to mm
def convert_pixel_to_mm(pixel_coords, conversion_factor):
    # Extract x and y values from the string and multiply by the conversion factor
    x, y = map(int, pixel_coords.strip("()").split(","))
    return np.array([x, y]) * conversion_factor

# Function to convert mm coordinates (relative to plate) to robot space
def convert_mm_to_robot_space(root_tip_mm, plate_position_robot):
    # Normalize mm values to a smaller scale (meters)
    normalized_mm = root_tip_mm / 1000
    # Add a fixed z-coordinate for inoculation and combine with the plate's robot space position
    root_tip_mm_with_z = np.append(normalized_mm, [0.171])
    # Calculate the robot coordinates by adding the plate's position in robot space
    robot_coords = root_tip_mm_with_z + plate_position_robot
    print(f"Normalized MM coords: {normalized_mm} -> Robot coords: {robot_coords}")
    return robot_coords

# Load root tip data from the CSV file
csv_path = "Root-Tips_plates.csv"  # Path to the CSV file containing root tip data
root_tip_data = pd.read_csv(csv_path)

# Initialize the simulation environment
env = OT2Env(render=True)  # Enable rendering for visual output
obs, info = env.reset()  # Reset the environment and get the initial observation

# Get the current plate image from the simulation
image_path = env.sim.get_plate_image()  # Get the image file path of the current plate
current_image = image_path.split("/")[-1].split(".")[0]  # Extract the base name without extension
base_image_name = current_image.split('_root_mask_plant_')[0]  # Extract base plant ID

# Filter the root tip data for the current plate
filtered_data = root_tip_data[root_tip_data['Plant ID'].str.startswith(base_image_name)]

# Calculate the conversion factor between pixel space and mm space
conversion_factor = calculate_conversion_factor(PLATE_SIZE_MM, PLATE_SIZE_PIXELS)
print(f"Conversion factor (mm/pixel): {conversion_factor}")

# Generate the goal positions in the robot's coordinate space
goal_positions = []
# for loop to iterate over the filtered data
for _, row in filtered_data.iterrows():
    pixel_coords = row['Tip (px)']
    # if the pixel coordinates are invalid, skip them
    if pixel_coords == "(0, 0)":  # Skip invalid coordinates
        print(f"Skipping invalid pixel coordinates: {pixel_coords}")
        continue

    # Convert pixel coordinates to mm and then to robot space
    root_tip_mm = convert_pixel_to_mm(pixel_coords, conversion_factor)
    root_tip_robot = convert_mm_to_robot_space(root_tip_mm, PLATE_POSITION_ROBOT)
    # appednd the root tip robot coordinates to the goal positions list
    goal_positions.append(root_tip_robot)

print(f"Goal positions in robot space: {goal_positions}")

# Load the trained PPO model for controlling the robot
model = PPO.load("model_4096")

# Control loop for processing each goal position
for i, goal_pos in enumerate(goal_positions, start=1):
    print(f"\nGoal {i}: Moving towards position {goal_pos}")
    env.goal_position = goal_pos[:3]  # Set the robot's goal position (x, y, z)
    step_count = 0

    # Main loop to move the robot to the goal position
    while True:
        step_count += 1

        # Use the model to predict the next action and apply it
        action, _states = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Calculate the distance between the robot and the goal position
        distance = np.linalg.norm(goal_pos[:3] - obs[:3])
        print(f"Step {step_count}: Current position = {obs[:3]}, Goal position = {goal_pos[:3]}, Distance = {distance:.4f}")

        # Check if the robot has reached the goal position
        if distance < 0.001:  # Threshold for determining if the goal is reached
            print(f"Inoculation at Goal {i}...")

            # Move the pipette down for inoculation
            inoculation_pos = goal_pos.copy()
            inoculation_pos[2] -= 0.1  # Lower the z-coordinate for inoculation
            env.goal_position = inoculation_pos[:3]
            for _ in range(50):  # Allow up to 50 steps to reach the inoculation depth
                action, _states = model.predict(obs, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, rewards, terminated, truncated, info = env.step(action)
                inoculation_distance = np.linalg.norm(inoculation_pos[:3] - obs[:3])
                if inoculation_distance < 0.001:  # Check if inoculation depth is reached
                    break

            # Simulate the inoculation action
            print("Dropping inoculum...")
            time.sleep(0.1)  # Allow time for the liquid to appear visually
            action = np.array([0, 0, 0, 1])  # Action for inoculation
            obs, rewards, terminated, truncated, info = env.step(action)
            time.sleep(0.2)  # Delay for visualization

            print(f"Goal {i}: Inoculum dropped at position {goal_pos}.")

            # Return the pipette to its original height
            print(f"Returning to original height after inoculation for Goal {i}...")
            return_pos = goal_pos.copy()
            env.goal_position = return_pos[:3]
            for _ in range(50):  # Allow up to 50 steps to return to original height
                action, _states = model.predict(obs, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, rewards, terminated, truncated, info = env.step(action)
                return_distance = np.linalg.norm(return_pos[:3] - obs[:3])
                if return_distance < 0.001:  # Check if returned to the original height
                    break

            break  # Move to the next goal

# Wait after the last inoculation for visualization
time.sleep(5)
env.close()