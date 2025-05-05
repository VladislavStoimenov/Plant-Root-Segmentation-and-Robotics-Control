# import required libraries
import time
import numpy as np
from ot2_gym_wrapper import OT2Env
import pandas as pd
from PID import PIDController

# define constants
PLATE_SIZE_MM = 150  # Physical size of the plate in mm
PLATE_SIZE_PIXELS = 2796  # Plate image dimensions in pixels
PLATE_POSITION_ROBOT = np.array([0.10775, 0.088 - 0.026, 0.057])  # Adjusted plate top-left position in robot coordinates
INOCULATION_DEPTH = -0.1  # Depth to go down for inoculation (relative to z-axis)

# Function to calculate conversion factor between pixel space and mm space
def calculate_conversion_factor(plate_size_mm, plate_size_pixels):
   # returns the conversion factor between pixel space and mm space
   return plate_size_mm / plate_size_pixels

# Function to convert pixel coordinates to mm
def convert_pixel_to_mm(pixel_coords, conversion_factor):
    # Extract x and y values from the string and multiply by the conversion factor
    x, y = map(int, pixel_coords.strip("()").split(","))
    return np.array([x, y]) * conversion_factor

# Function to convert mm coordinates to robot space
def convert_mm_to_robot_space(root_tip_mm, plate_position_robot):
    normalized_mm = root_tip_mm / 1000  # Normalize mm values to meters
    root_tip_mm_with_z = np.append(normalized_mm, [0.171])  # Add fixed z-coordinate for inoculation
    robot_coords = root_tip_mm_with_z + plate_position_robot # Combine with plate's robot space position
    print(f"Normalized MM coords: {normalized_mm} -> Robot coords: {robot_coords}") # print normalized and robot coordinates
    return robot_coords

# Load root tip data for the robot images from the CSV file
csv_path = "Root-Tips_plates.csv"  # Path to the CSV file containing root tip data
root_tip_data = pd.read_csv(csv_path)

# Initialize the simulation environment
env = OT2Env(render=True)  # Enable rendering for visualization
obs, info = env.reset()  # Reset the environment and get the initial observation

# Get the current plate image from the simulation
image_path = env.sim.get_plate_image()
# Extract the base name without extension
current_image = image_path.split("/")[-1].split(".")[0]
base_image_name = current_image.split('_root_mask_plant_')[0]

# Filter the root tip data for the current plate
filtered_data = root_tip_data[root_tip_data['Plant ID'].str.startswith(base_image_name)]

# Calculate the conversion factor between pixel space and mm space
conversion_factor = calculate_conversion_factor(PLATE_SIZE_MM, PLATE_SIZE_PIXELS)
print(f"Conversion factor (mm/pixel): {conversion_factor}")

# Generate the goal positions in the robot's coordinate space
goal_positions = []
# for loop to iterate over the filtered data
for _, row in filtered_data.iterrows():
    # get the pixel coordinates
    pixel_coords = row['Tip (px)']
    # if the pixel coordinates are invalid, skip them
    if pixel_coords == "(0, 0)":  # Skip invalid coordinates
        print(f"Skipping invalid pixel coordinates: {pixel_coords}")
        continue

    # Convert pixel coordinates to mm and then to robot space
    root_tip_mm = convert_pixel_to_mm(pixel_coords, conversion_factor)
    root_tip_robot = convert_mm_to_robot_space(root_tip_mm, PLATE_POSITION_ROBOT)
    # append the root tip robot coordinates to the goal positions list
    goal_positions.append(root_tip_robot)

print(f"Goal positions in robot space: {goal_positions}")

# Initialize PID controllers for each axis
#pid_x = PIDController(45, 0.5, 3, setpoint=1)
#pid_y = PIDController(45, 0.1, 0.1, setpoint=1)
#pid_z = PIDController(45, 0.1, 5, setpoint=1)

pid_x = PIDController(27, 0.5, 2.8, setpoint=1)
pid_y = PIDController(45, 0.1, 2, setpoint=1)
pid_z = PIDController(18, 2.8, 7.8, setpoint=1)


# Control loop to process each goal position
for i, goal_pos in enumerate(goal_positions, start=1):
    print(f"\nGoal {i}: Moving towards position {goal_pos}")

    # Set PID setpoints to the current goal position
    pid_x.setpoint = goal_pos[0]
    pid_y.setpoint = goal_pos[1]
    pid_z.setpoint = goal_pos[2]

    # while loop to move the robot to the goal position
    while True:
        # Current position of the pipette
        pipette_position = obs[:3]

        # Calculate PID outputs for each axis
        dt = 0.02  # Time step for PID updates
        action_x = pid_x.update(pipette_position[0], dt)
        action_y = pid_y.update(pipette_position[1], dt)
        action_z = pid_z.update(pipette_position[2], dt)
        # Combine the actions into a single array
        action = np.array([action_x, action_y, action_z])

        # Apply the calculated action to the environment
        obs, rewards, terminated, truncated, info = env.step(action)

        # Calculate the distance to the goal position
        distance = np.linalg.norm(goal_pos[:3] - obs[:3])
        print(f"Current position: {obs[:3]}, Goal position: {goal_pos[:3]}, Distance: {distance:.4f}")

        # Check if the goal position is reached
        if distance < 0.001:  # Threshold for reaching the goal
            print(f"Inoculation at Goal {i}...")

            # Move the pipette down for inoculation
            inoculation_pos = goal_pos.copy()
            inoculation_pos[2] += INOCULATION_DEPTH  # Adjust z-coordinate for inoculation
            pid_x.setpoint, pid_y.setpoint, pid_z.setpoint = inoculation_pos # Set PID setpoints for inoculation

            # Move the pipette to the inoculation depth
            for _ in range(50):  # Limit the number of steps for inoculation
                # Update PID controllers for each axis
                action_x = pid_x.update(obs[0], dt)
                action_y = pid_y.update(obs[1], dt)
                action_z = pid_z.update(obs[2], dt)
                # Combine the actions into a single array
                action = np.array([action_x, action_y, action_z])
                obs, rewards, terminated, truncated, info = env.step(action)
                inoculation_distance = np.linalg.norm(inoculation_pos[:3] - obs[:3])
                # Check if the inoculation depth is reached
                if inoculation_distance < 0.001:  # Threshold for inoculation depth
                    break

            # Simulate inoculation process
            print("Dropping inoculum...")
            action = np.array([0, 0, 0, 1])  # action for inoculation
            # Apply the action to the environment
            obs, rewards, terminated, truncated, info = env.step(action)
            time.sleep(0.2)  # Pause for visualization

            # Return the pipette to its original height
            print(f"Returning to original height after inoculation for Goal {i}...")
            pid_x.setpoint, pid_y.setpoint, pid_z.setpoint = goal_pos
            # for loop to return the pipette to the original height
            for _ in range(50):  # Limit the number of steps for returning to height
                action_x = pid_x.update(obs[0], dt)
                action_y = pid_y.update(obs[1], dt)
                action_z = pid_z.update(obs[2], dt)
                action = np.array([action_x, action_y, action_z])
                obs, rewards, terminated, truncated, info = env.step(action)
                return_distance = np.linalg.norm(goal_pos[:3] - obs[:3])
                # Check if the original height is reached withing the threshold
                if return_distance < 0.002:  # Threshold for returning height
                    break

            break  # Move to the next goal

# Wait after the last inoculation for visualization
time.sleep(5)
# Close the environment
env.close()
