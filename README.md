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
   
    jupyter notebook  - task_9.ipynb

## Dependencies

The simulation requires the following Python library:

    `pybullet`: Physics simulation and rendering engine.

To install all dependencies, run:

    pip install pybullet
         


## Working Envelope of the Pipette

The pipette's working envelope defines the maximum range of movement in the simulation environment. It is a cube with the following boundaries:

- **X-axis**: `[-1.927, 2.073]`
- **Y-axis**: `[-1.9105, 2.0895]`
- **Z-axis**: `[-1.8805, 2.1195]`

### Corner Points of the Envelope
The envelope consists of 8 corners, defined as follows:

| Corner | X Coordinate | Y Coordinate | Z Coordinate |
|--------|--------------|--------------|--------------|
| 1      | -1.927       | -1.9105      | -1.8805      |
| 2      | -1.927       | -1.9105      | 2.1195       |
| 3      | -1.927       | 2.0895       | -1.8805      |
| 4      | -1.927       | 2.0895       | 2.1195       |
| 5      | 2.073        | -1.9105      | -1.8805      |
| 6      | 2.073        | -1.9105      | 2.1195       |
| 7      | 2.073        | 2.0895       | -1.8805      |
| 8      | 2.073        | 2.0895       | 2.1195       |


## Running the Simulation

To move the pipette to all 8 corners of the working envelope, use the provided notebook - `task_9.ipynb`. The pipette will sequentially visit each corner, and the state will be logged for every position.

## Notes

- Ensure the simulation environment is properly configured before running the code.
- If issues are encountered, refer to the documentation for `pybullet` or open an issue in the repository.