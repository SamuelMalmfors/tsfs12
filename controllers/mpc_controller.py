
import cvxpy as cp
import numpy as np


class MPCController:
    def __init__(
        self,
        dt=0.1,
        horizon=30,
        min_distance=8.0,
        time_gap=1.5,
        cruise_speed=80.0 / 3.6,
        w_gap=1.0,
        w_speed=0.5,
        w_accel=0.05,
        w_jerk=0.1,
        w_safety=1000.0,
        max_decel=-8.0,
    ):
        self.dt = dt
        self.N = horizon
        self.min_distance = min_distance
        self.time_gap = time_gap
        self.cruise_speed = cruise_speed

        self.max_accel = 2.5
        self.max_decel = max_decel

        # Optimization variables
        self.gap = cp.Variable(self.N + 1)
        self.velocity = cp.Variable(self.N + 1)
        self.accel = cp.Variable(self.N)

        # Slack for the soft safety distance
        self.slack = cp.Variable(self.N + 1, nonneg=True)

        # Current measured state
        self.initial_gap = cp.Parameter()
        self.initial_speed = cp.Parameter(nonneg=True)
        self.lead_speed = cp.Parameter(nonneg=True)
        self.prev_accel = cp.Parameter(value=0.0)

        # Cost weights
        self.w_gap = w_gap
        self.w_speed = w_speed
        self.w_accel = w_accel
        self.w_jerk = w_jerk
        self.w_safety = w_safety

        constraints = [
            self.gap[0] == self.initial_gap,
            self.velocity[0] == self.initial_speed,
        ]

        cost = 0

        for k in range(self.N):
            a = self.accel[k]
            v = self.velocity[k]
            d = self.gap[k]

            # Longitudinal prediction model
            constraints += [
                self.velocity[k + 1] ==
                v + a * self.dt,

                self.gap[k + 1] ==
                d + (self.lead_speed - v) * self.dt
                - 0.5 * a * self.dt**2,

                a >= self.max_decel,
                a <= self.max_accel,

                self.velocity[k + 1] >= 0,
                self.velocity[k+1] <= self.cruise_speed
            ]

            desired_gap = (
                self.min_distance +
                self.time_gap * v
            )

            gap_error = d - desired_gap

            # Penalize distance tracking error
            cost += self.w_gap * cp.square(gap_error)

            # Penalize speed above cruise speed
            cost += self.w_speed * cp.square(
                cp.pos(v - self.cruise_speed)
            )

            # Penalize excessive acceleration
            cost += self.w_accel * cp.square(a)

            # Penalize changes in acceleration,
            # including the step from the last applied value
            previous = self.prev_accel if k == 0 else self.accel[k - 1]

            cost += self.w_jerk * cp.square(a - previous)

        if self.w_safety > 0:
            # Soft safety distance: gap >= min_distance
            for k in range(1, self.N + 1):
                constraints += [
                    self.gap[k] >= self.min_distance - self.slack[k]
                ]

            cost += self.w_safety * cp.sum_squares(self.slack)

        # Penalize terminal distance error
        terminal_error = (
            self.gap[self.N]
            - self.min_distance
            - self.time_gap * self.velocity[self.N]
        )

        cost += 2.0 * cp.square(terminal_error)

        self.problem = cp.Problem(
            cp.Minimize(cost),
            constraints
        )

    def update(self, gap, ego_speed, lead_speed):
        self.initial_gap.value = float(gap)
        self.initial_speed.value = max(0.0, float(ego_speed))
        self.lead_speed.value = max(0.0, float(lead_speed))

        try:
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False
            )

            if (
                self.problem.status not in
                (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                or self.accel.value is None
            ):
                raise RuntimeError("MPC optimization failed")

            acceleration = float(self.accel.value[0])

        except (cp.error.SolverError, RuntimeError) as exc:
            print(f"MPC solver warning: {exc}")

            # Conservative fallback
            acceleration = self.max_decel

        self.prev_accel.value = acceleration

        return acceleration
