# Description: This file contains the PIDController class, which is a simple implementation of a PID controller.
# Define the PIDController class
class PIDController:
    # Initialize the PID controller with the specified gains and setpoint
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        # Initialize gains and setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        # Initialize integral and previous error
        self.integral = 0.0
        self.previous_error = 0.0

    # Update the PID controller with the current value and time step
    def update(self, current_value, dt):
        # Compute error
        error = self.setpoint - current_value

        # Update integral
        self.integral += error * dt
              
        # Compute derivative
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0

        # Compute PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Update previous error
        self.previous_error = error

        return output