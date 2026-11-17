"""
M10 — Time-to-Collision (Sec 14).

TTC = distance / closing_speed, only meaningful when the object is actually
closing (closing_speed > min_valid_closing_speed_mps). Otherwise returns
float('inf') (no collision course under current motion) — never claim a
numeric TTC when the object is stationary/receding relative to the camera.

Per Sec 14: TTC is one *feature* of the risk model, not a standalone
collision predictor.
"""


def compute_ttc(cfg, distance_m, closing_speed_mps):
    if distance_m is None or closing_speed_mps is None:
        return None  # insufficient history yet

    min_speed = cfg["ttc"]["min_valid_closing_speed_mps"]
    if closing_speed_mps <= min_speed:
        return float("inf")  # not on a closing trajectory

    return distance_m / closing_speed_mps
