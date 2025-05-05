# import required libraries
import numpy as np
from PID import PIDController
from ot2_gym_wrapper import OT2Env
import matplotlib.pyplot as plt

# function to run the PID controller
def main():

    # define the PID controller gains for each axis
    Kp = [27, 45, 18]
    Ki = [0.5, 0.1, 2.8]
    Kd = [2.8, 2, 7.8]


    # Initialize PID controllers for x, y, z
    pid_x = PIDController(Kp[0], Ki[0], Kd[0])
    pid_y = PIDController(Kp[1], Ki[1], Kd[1])
    pid_z = PIDController(Kp[2], Ki[2], Kd[2])

    # Initialize the OpenAI Gym environment
    env = OT2Env(render=True)  # Enable rendering for visualization
    # Define start and goal positions to test the controller on fixed positions
    start_position = np.array([0.05, 0.05, 0.18])
    goal_position = np.array([0.15, 0.15, 0.25])

    # Set PID setpoints to the goal position
    pid_x.setpoint = goal_position[0]
    pid_y.setpoint = goal_position[1]
    pid_z.setpoint = goal_position[2]

    # Reset environment to start position
    observation, _ = env.reset()
    observation[:3] = start_position

    # Lists to store data for plotting
    steps = []
    positions_x = []
    positions_y = []
    positions_z = []
    actions_x = []
    actions_y = []
    actions_z = []

    for step in range(500):  # Reduce step limit for faster completion
        # Get current pipette position
        pipette_position = observation[:3]
        
        # Calculate PID outputs for each axis
        action_x = pid_x.update(pipette_position[0], dt=0.05)
        action_y = pid_y.update(pipette_position[1], dt=0.05)
        action_z = pid_z.update(pipette_position[2], dt=0.05)

        # Combine actions into a single array
        action = np.array([action_x, action_y, action_z])

        # Pass the action to the environment step function
        observation, reward, terminated, truncated, _ = env.step(action)

         # Store data for plotting
        steps.append(step)
        positions_x.append(pipette_position[0])
        positions_y.append(pipette_position[1])
        positions_z.append(pipette_position[2])
        actions_x.append(action_x)
        actions_y.append(action_y)
        actions_z.append(action_z)

        # Calculate distance to goal position
        distance_to_goal = np.linalg.norm(np.array(pipette_position) - np.array(goal_position))
        # Print information for each step
        print(f"Step {step + 1}:")
        print(f"  Pipette Position: {pipette_position}")
        print(f"  Goal Position: {goal_position}")
        print(f"  Distance to Goal: {distance_to_goal}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}\n")

        # Early stopping condition if close enough to the goal
        if distance_to_goal < 0.001:
            print(f"Goal reached within tolerance at step {step + 1}.")
            break
        
        if terminated or truncated:
            break

    # Close the environment
    env.close()


    # Plotting the results
    plt.figure(figsize=(12, 8))
    # Plot position and action for each axis
    plt.subplot(3, 1, 1)
    plt.plot(steps, positions_x, label='Position X')
    plt.plot(steps, actions_x, label='Action X')
    plt.axhline(y=goal_position[0], color='r', linestyle='--', label='Goal X')
    plt.xlabel('Step')
    plt.ylabel('X Axis')
    plt.legend()
    # Repeat for Y and Z axes
    plt.subplot(3, 1, 2)
    plt.plot(steps, positions_y, label='Position Y')
    plt.plot(steps, actions_y, label='Action Y')
    plt.axhline(y=goal_position[1], color='r', linestyle='--', label='Goal Y')
    plt.xlabel('Step')
    plt.ylabel('Y Axis')
    plt.legend()
    # Repeat for Z axis
    plt.subplot(3, 1, 3)
    plt.plot(steps, positions_z, label='Position Z')
    plt.plot(steps, actions_z, label='Action Z')
    plt.axhline(y=goal_position[2], color='r', linestyle='--', label='Goal Z')
    plt.xlabel('Step')
    plt.ylabel('Z Axis')
    plt.legend()
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
# Run the main function
if __name__ == "__main__":
    main()