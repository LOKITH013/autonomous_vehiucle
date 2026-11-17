"""
M9/M10 — Distance proxy + relative velocity (Sec 12, Sec 13).

IMPORTANT: `estimate_distance_proxy()` is the TEMPORARY bounding-box-based
proxy the spec explicitly calls out (Sec 12): it exists only to exercise the
motion/TTC pipeline end-to-end, and it is NOT a validated metric distance
measurement. Do not report its output as real-world distance in results —
mark it "proxy (unvalidated)" everywhere it's logged or displayed (Sec 42
requires "Depth ✓" to mean the validated M9 replacement, not this).

Replacing this function's internals (with monocular depth / calibrated
geometry) is the entire scope of M9 and requires no changes anywhere else,
since MotionTracker only depends on receiving a distance value in meters.
"""
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class MotionEstimate:
    closing_speed_mps: float | None = None
    image_velocity_x_px_s: float | None = None
    image_velocity_y_px_s: float | None = None
    image_direction: str = "UNKNOWN"


def estimate_distance_proxy(cfg, cls_name, bbox_xyxy):
    """
    Pinhole-camera approximation:
        distance_m = (real_width_m * focal_length_px) / bbox_width_px
    Returns None if the class has no known width or bbox is degenerate.
    """
    known_widths = cfg["distance_proxy"]["known_widths_m"]
    focal_length_px = cfg["distance_proxy"]["focal_length_px"]

    real_width_m = known_widths.get(cls_name)
    if real_width_m is None:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    bbox_width_px = max(x2 - x1, 1e-6)

    return (real_width_m * focal_length_px) / bbox_width_px


class MotionTracker:
    """
    Keeps short per-track histories to estimate distance range rate and
    image-plane direction. Positive range rate means distance is shrinking.
    """

    def __init__(self, history_len=5, direction_deadband_px_s=5.0):
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.direction_deadband_px_s = direction_deadband_px_s

    def update(self, track_id, distance_m, timestamp=None):
        if track_id is None or distance_m is None:
            return None
        return self.observe(track_id, distance_m, timestamp=timestamp).closing_speed_mps

    def observe(self, track_id, distance_m, center_xy=None, timestamp=None):
        """Update one track and estimate relative range rate and image-plane motion."""
        if track_id is None or distance_m is None:
            if track_id is None:
                return MotionEstimate()
        t = timestamp if timestamp is not None else time.time()
        hist = self.history[track_id]
        if hist and t <= hist[-1][0]:
            return MotionEstimate()
        hist.append((t, distance_m, center_xy))

        distance_samples = [(sample_t, d) for sample_t, d, _ in hist if d is not None]
        position_samples = [
            (sample_t, center) for sample_t, _, center in hist if center is not None
        ]

        closing_speed = None
        if len(distance_samples) >= 2:
            closing_speed = -self._slope(distance_samples)

        velocity_x = velocity_y = None
        direction = "UNKNOWN"
        if len(position_samples) >= 2:
            velocity_x = self._slope(
                [(sample_t, center[0]) for sample_t, center in position_samples]
            )
            velocity_y = self._slope(
                [(sample_t, center[1]) for sample_t, center in position_samples]
            )
            direction = self._image_direction(velocity_x, velocity_y)

        return MotionEstimate(
            closing_speed_mps=closing_speed,
            image_velocity_x_px_s=velocity_x,
            image_velocity_y_px_s=velocity_y,
            image_direction=direction,
        )

    @staticmethod
    def _slope(samples):
        mean_t = sum(t for t, _ in samples) / len(samples)
        mean_value = sum(value for _, value in samples) / len(samples)
        variance = sum((t - mean_t) ** 2 for t, _ in samples)
        if variance == 0:
            return 0.0
        covariance = sum(
            (t - mean_t) * (value - mean_value) for t, value in samples
        )
        return covariance / variance

    def _image_direction(self, velocity_x, velocity_y):
        deadband = self.direction_deadband_px_s
        moving_x = abs(velocity_x) > deadband
        moving_y = abs(velocity_y) > deadband
        if not moving_x and not moving_y:
            return "STATIONARY"

        horizontal = "RIGHT" if velocity_x > 0 else "LEFT"
        vertical = "DOWN" if velocity_y > 0 else "UP"
        if moving_x and moving_y:
            if abs(velocity_x) > 2 * abs(velocity_y):
                return horizontal
            if abs(velocity_y) > 2 * abs(velocity_x):
                return vertical
            return f"{vertical}-{horizontal}"
        return horizontal if moving_x else vertical

    def latest_distance(self, track_id):
        hist = self.history.get(track_id)
        if not hist:
            return None
        return next(
            (distance for _, distance, _ in reversed(hist) if distance is not None),
            None,
        )

    def prune(self, active_ids):
        """Drop history for IDs the tracker has lost (Sec 11 / max_lost_frames)."""
        stale = [tid for tid in self.history if tid not in active_ids]
        for tid in stale:
            del self.history[tid]
