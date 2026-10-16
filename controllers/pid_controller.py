
class PIDController:
    def __init__(self, kp=0.15, ki=0.005, kd=0.08, dt=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error * self.dt
        self.integral = max(-100, min(100, self.integral))

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        u = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return max(-1.0, min(1.0, u))
