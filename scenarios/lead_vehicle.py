
def get_lead_target_speed(t, scenario):
    """
    Return lead vehicle target speed in m/s.
    """

    # The lead drives below the ego cruise speed (80 km/h) so the
    # ego can always close a gap, and the braking starts at t=15 s
    # after the start transient has settled.

    if scenario == "gentle":
        # 70 -> 40 km/h over 8 seconds
        if t < 15.0:
            kmh = 70.0
        elif t < 23.0:
            kmh = 70.0 - (30.0 / 8.0) * (t - 15.0)
        else:
            kmh = 40.0

    elif scenario == "emergency":
        # Braking to standstill, rate limited to LEAD_MAX_BRAKE
        # (6 m/s^2) in main.py
        if t < 15.0:
            kmh = 70.0
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
