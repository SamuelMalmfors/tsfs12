
class PIDController:
    def __init__(
        self,
        kp=0.15,
        ki=0.005,
        kd=0.08,
        dt=0.5,
        max_du=None,
        integral_limit=100.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        # Max change in output per step (None = no rate limit)
        self.max_du = max_du

        # Bound on the integral state (anti-windup)
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = None
        self.prev_u = 0.0

    def update(self, error):
        # Avoid derivative kick on the first step
        if self.prev_error is None:
            self.prev_error = error

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        candidate = self.integral + error * self.dt

        u = (
            self.kp * error
            + self.ki * candidate
            + self.kd * derivative
        )

        # Anti-windup: only integrate while the output is not
        # saturated, or when the error pulls it back out
        if -1.0 < u < 1.0 or error * u < 0:
            self.integral = max(
                -self.integral_limit,
                min(self.integral_limit, candidate)
            )

        u = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        u = max(-1.0, min(1.0, u))

        if self.max_du is not None:
            u = max(
                self.prev_u - self.max_du,
                min(self.prev_u + self.max_du, u)
            )

        self.prev_u = u

        return u
