"""
M5 — Lane detection (Sec 8).

Classical OpenCV pipeline:
    Grayscale → Gaussian blur → Canny edges → ROI mask →
    Hough lines → left/right separation → averaging →
    lane overlay + lateral position estimate.

No neural network required — this makes it:
  * GPU-free (runs on the same laptop as inference)
  * Fully explainable (every step is a documented CV operation)
  * Easy to swap out later for a learned model (CLRNet / UFLDv2)

Returns a LaneResult dataclass consumed by pipeline.py and risk_engine.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class LaneResult:
    """All lane information for one frame."""
    left_line:  Optional[Tuple[int, int, int, int]] = None   # (x1,y1,x2,y2)
    right_line: Optional[Tuple[int, int, int, int]] = None
    lane_center_x:     Optional[float] = None   # pixel x of lane midpoint
    frame_center_x:    Optional[float] = None   # pixel x of frame centre
    lateral_deviation: Optional[float] = None   # positive = drifting right
    position: str = "UNKNOWN"                   # LEFT / CENTER / RIGHT / UNKNOWN
    lane_detected: bool = False


class LaneDetector:
    """
    Lane detector with short-gap temporal stabilization.
    All tunable parameters come from config.yaml → lane section.
    """

    def __init__(self, cfg: dict):
        lc = cfg.get("lane", {})
        # Canny thresholds
        self.canny_low  = lc.get("canny_low",  50)
        self.canny_high = lc.get("canny_high", 150)
        # Hough parameters
        self.hough_threshold    = lc.get("hough_threshold",    20)
        self.hough_min_line_len = lc.get("hough_min_line_len", 20)
        self.hough_max_line_gap = lc.get("hough_max_line_gap", 300)
        # ROI: fraction of frame height where the trapezoid starts from bottom
        self.roi_top_frac   = lc.get("roi_top_frac",   0.58)  # top of ROI
        self.roi_top_width  = lc.get("roi_top_width",  0.08)  # half-width at top
        # Lane-position tolerance (fraction of frame width either side of centre)
        self.centre_tol = lc.get("centre_tolerance", 0.05)
        self.processing_width = max(1, int(lc.get("processing_width", 640)))
        self.max_missing_frames = max(0, int(lc.get("max_missing_frames", 3)))
        self._previous_lines = (None, None)
        self._missing_frames = 0

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> LaneResult:
        frame_h, frame_w = frame.shape[:2]
        scale = min(1.0, self.processing_width / frame_w)
        if scale < 1.0:
            work_w = self.processing_width
            work_h = max(1, round(frame_h * scale))
            frame = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        result = LaneResult(frame_center_x=frame_w / 2)

        # 1. Greyscale + blur + edges
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # 2. Trapezoidal ROI mask
        mask = self._roi_mask(edges, h, w)

        # 3. Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            mask,
            rho=1, theta=np.pi / 180,
            threshold=max(1, round(self.hough_threshold * scale)),
            minLineLength=max(1, round(self.hough_min_line_len * scale)),
            maxLineGap=max(1, round(self.hough_max_line_gap * scale)),
        )

        # 4. Separate and average left / right lines
        left_line = right_line = None
        if lines is not None:
            left_line = self._average_lines(lines, w, h, side="left")
            right_line = self._average_lines(lines, w, h, side="right")
        if scale < 1.0:
            left_line = self._scale_line(left_line, 1.0 / scale)
            right_line = self._scale_line(right_line, 1.0 / scale)

        previous_left, previous_right = self._previous_lines
        if left_line is None and right_line is None:
            self._missing_frames += 1
            if self._missing_frames > self.max_missing_frames:
                self._previous_lines = (None, None)
                return result
            left_line, right_line = previous_left, previous_right
        else:
            if left_line is None or right_line is None:
                self._missing_frames += 1
            else:
                self._missing_frames = 0
            if self._missing_frames <= self.max_missing_frames and left_line is None:
                left_line = previous_left
            if self._missing_frames <= self.max_missing_frames and right_line is None:
                right_line = previous_right
            left_line = self._smooth_line(previous_left, left_line)
            right_line = self._smooth_line(previous_right, right_line)
            self._previous_lines = (left_line, right_line)

        result.left_line  = left_line
        result.right_line = right_line
        result.lane_detected = (left_line is not None) or (right_line is not None)

        # 5. Lateral position
        if left_line and right_line:
            result.lane_center_x = (left_line[0] + right_line[0]) / 2
            result.lateral_deviation = result.lane_center_x - (frame_w / 2)
            result.position = self._classify_position(result.lateral_deviation, frame_w)
        elif left_line:
            result.position = "RIGHT"   # only left lane visible → car is right of it
        elif right_line:
            result.position = "LEFT"

        return result

    @staticmethod
    def _scale_line(line, factor):
        if line is None:
            return None
        return tuple(round(value * factor) for value in line)

    @staticmethod
    def _smooth_line(previous, current, current_weight=0.65):
        if current is None or previous is None:
            return current
        return tuple(
            round(old * (1.0 - current_weight) + new * current_weight)
            for old, new in zip(previous, current)
        )

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """Draw lane lines and position HUD onto frame (in-place)."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        if result.left_line:
            x1, y1, x2, y2 = result.left_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)

        if result.right_line:
            x1, y1, x2, y2 = result.right_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Filled lane polygon
        if result.left_line and result.right_line:
            pts = np.array([
                [result.left_line[0],  result.left_line[1]],
                [result.left_line[2],  result.left_line[3]],
                [result.right_line[2], result.right_line[3]],
                [result.right_line[0], result.right_line[1]],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 0))

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # HUD text
        pos_color = {"CENTER": (0, 220, 0), "LEFT": (0, 165, 255),
                     "RIGHT": (0, 165, 255), "UNKNOWN": (128, 128, 128)}
        color = pos_color.get(result.position, (255, 255, 255))
        label = f"Lane: {result.position}"
        if result.lateral_deviation is not None:
            label += f"  dev={result.lateral_deviation:+.0f}px"
        cv2.putText(frame, label, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _roi_mask(self, edges, h, w):
        top_y  = int(h * self.roi_top_frac)
        top_hw = int(w * self.roi_top_width)
        cx     = w // 2
        polygon = np.array([[
            (0,        h),
            (cx - top_hw, top_y),
            (cx + top_hw, top_y),
            (w,        h),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _average_lines(self, lines, w, h, side):
        """
        Filter lines by slope sign (left = negative slope in image coords,
        right = positive), then fit a single averaged line.
        """
        coords = []
        for line in lines:
            pts = np.array(line).flatten()
            if len(pts) != 4:
                continue
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # reject near-horizontal lines
            if abs(slope) < 0.4:
                continue
            if side == "left"  and slope < 0 and x1 < w * 0.55 and x2 < w * 0.55:
                coords.extend([(x1, y1), (x2, y2)])
            elif side == "right" and slope > 0 and x1 > w * 0.45 and x2 > w * 0.45:
                coords.extend([(x1, y1), (x2, y2)])

        if len(coords) < 2:
            return None

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        try:
            poly = np.polyfit(ys, xs, 1)   # x = f(y) — more stable for near-vertical lines
        except (np.linalg.LinAlgError, TypeError):
            return None

        y_bot = h
        y_top = int(h * self.roi_top_frac)
        x_bot = int(np.polyval(poly, y_bot))
        x_top = int(np.polyval(poly, y_top))
        return (x_bot, y_bot, x_top, y_top)

    def _classify_position(self, deviation, w):
        tol = w * self.centre_tol
        if abs(deviation) <= tol:
            return "CENTER"
        return "RIGHT" if deviation > 0 else "LEFT"
