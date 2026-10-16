
def get_lead_target_speed(t, scenario):
    """
    Return lead vehicle target speed in m/s.
    """

    if scenario == "gentle":
        # 80 -> 50 km/h over 8 seconds
        if t < 10.0:
            kmh = 80.0
        elif t < 18.0:
            kmh = 80.0 - (30.0 / 8.0) * (t - 10.0)
        else:
            kmh = 50.0

    elif scenario == "emergency":
        # Full braking is applied in main.py at t=10 s
        if t < 10.0:
            kmh = 80.0
        else:
            kmh = 0.0

    elif scenario == "stop_and_go":
        # Alternating acceleration and braking
        if t < 10.0:
            kmh = 40.0

        elif t < 15.0:
            kmh = 40.0 * (15.0 - t) / 5.0

        elif t < 20.0:
            kmh = 0.0

        elif t < 28.0:
            kmh = 40.0 * (t - 20.0) / 8.0

        elif t < 35.0:
            kmh = 40.0

        elif t < 40.0:
            kmh = 40.0 * (40.0 - t) / 5.0

        elif t < 45.0:
            kmh = 0.0

        else:
            kmh = min(
                40.0,
                5.0 * (t - 45.0)
            )

    else:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    return kmh / 3.6
