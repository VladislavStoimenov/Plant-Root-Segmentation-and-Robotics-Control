from sim_class import Simulation

# Initialize the simulation with one robot
sim = Simulation(num_agents=1)

# Define the corner velocities for the working envelope
# Velocities are adjusted to move to min/max for X, Y, Z
corner_velocities = [
    [-1, -1, -1],  # (x_min, y_min, z_min)
    [1, -1, -1],   # (x_max, y_min, z_min)
    [-1, 1, -1],   # (x_min, y_max, z_min)
    [1, 1, -1],    # (x_max, y_max, z_min)
    [-1, -1, 1],   # (x_min, y_min, z_max)
    [1, -1, 1],    # (x_max, y_min, z_max)
    [-1, 1, 1],    # (x_min, y_max, z_max)
    [1, 1, 1],     # (x_max, y_max, z_max)
]

# List to store the observed coordinates
working_envelope = []

# Move to each corner and record coordinates
for velocities in corner_velocities:
    actions = [[velocities[0], velocities[1], velocities[2], 0]]  # 0 = No drop command
    observations = sim.run(actions)  # Simulate the action
    coordinates = observations[0]   # Assume the first agent's state contains the coordinates
    working_envelope.append(coordinates)

# Print the working envelope coordinates
print("Working Envelope Coordinates:")
for i, coord in enumerate(working_envelope):
    print(f"Corner {i + 1}: {coord}")
