
# Controller tunings used in the presentation.
# "bad" presets are deliberately mistuned to show typical failure modes.

PID_PRESETS = {
    # Gain far too low, P only: reacts late to the lead braking,
    # large distance error and collision in emergency braking
    "bad": dict(
        kp=0.04,
        ki=0.0,
        kd=0.0,
        max_du=None
    ),

    # PD with rate limit on the output to keep jerk down
    "good": dict(
        kp=0.20,
        ki=0.0,
        kd=0.20,
        max_du=0.04
    ),
}

MPC_PRESETS = {
    # Comfort weighted far above distance, braking limited to
    # "comfortable" -4 m/s^2 and no safety constraint:
    # smooth but too slow to react when the lead brakes hard
    "bad": dict(
        horizon=30,
        w_gap=0.05,
        w_speed=0.5,
        w_accel=2.0,
        w_jerk=5.0,
        w_safety=0.0,
        max_decel=-4.0
    ),

    # 3 s horizon, comfort terms and soft safety distance
    "good": dict(
        horizon=30,
        w_gap=1.0,
        w_speed=0.5,
        w_accel=0.05,
        w_jerk=0.1,
        w_safety=1000.0
    ),
}
