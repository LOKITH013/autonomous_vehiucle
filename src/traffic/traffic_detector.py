"""
M6 — Traffic sign and traffic light recognition (Sec 9).

Strategy:
  The IDD fine-tuned model already detects two relevant classes:
    • "traffic light"  — bounding box from detector
    • "traffic sign"   — bounding box from detector

  This module adds a SECOND layer on top of those detections:

  Traffic LIGHTS → colour analysis on the detected bounding-box ROI.
      1. Crop the bbox from the frame.
      2. Convert to HSV.
      3. Require enough bright, saturated pixels, then find the dominant
         red / yellow / green signal color in the crop.
      4. Return "RED", "YELLOW", "GREEN", or "UNKNOWN".

  Traffic SIGNS → presence detection only.
      Sub-classification of sign type (stop / speed limit / warning …)
      requires a dedicated annotated sign dataset which is not available
      in IDD-D.  We therefore report sign PRESENCE honestly ("SIGN_DETECTED")
      rather than inventing a sub-classifier. Documented as future work.

  Limitations (as required by spec Sec 10, 39):
    - Colour analysis is sensitive to lighting / occlusion.
    - Night / rain conditions reduce accuracy.
    - NOT a certified safety system.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ── HSV colour ranges for traffic light states ──────────────────────
# Hue is 0-179 in OpenCV.  Red wraps around, so two ranges needed.
_RED_1  = (np.array([0,  40, 80]), np.array([10, 255, 255]))
_RED_2  = (np.array([160, 40, 80]), np.array([179, 255, 255]))
_YELLOW = (np.array([15, 40, 80]), np.array([35, 255, 255]))
_GREEN  = (np.array([36,  40, 80]), np.array([90, 255, 255]))


@dataclass
class TrafficLightState:
    state: str = "UNKNOWN"     # RED / YELLOW / GREEN / UNKNOWN
    confidence: float = 0.0    # dominance among bright, saturated signal-color pixels
    bbox: Optional[List[int]] = None
    track_id: Optional[int] = None
    detection_confidence: float = 0.0


@dataclass
class TrafficResult:
    lights: List[TrafficLightState] = field(default_factory=list)
    signs_detected: int = 0
    dominant_light: str = "UNKNOWN"   # highest-confidence light state


class TrafficDetector:
    """
    Post-processes YOLO detection output for traffic lights and signs.
    No additional model needed — leverages the IDD fine-tuned detector.
    """

    def __init__(self, cfg: dict):
        tc = cfg.get("traffic", {})
        self.min_confidence = tc.get("min_light_confidence", 0.10)
        self.min_bbox_area  = tc.get("min_bbox_area_px",     200)
        self.min_color_pixel_fraction = tc.get("min_color_pixel_fraction", 0.01)
        self.min_color_dominance = tc.get("min_color_dominance", 0.55)

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, detections: list) -> TrafficResult:
        """
        Parameters
        ----------
        frame      : BGR frame (numpy array)
        detections : list of dicts from Detector.track_frame()
                     Each dict has keys: cls_name, xyxy, conf, id

        Returns
        -------
        TrafficResult
        """
        result = TrafficResult()

        for det in detections:
            cls = det["cls_name"]

            if cls == "traffic sign":
                result.signs_detected += 1

            elif cls == "traffic light":
                x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
                height, width = frame.shape[:2]
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < self.min_bbox_area:
                    continue
                state, conf = self._classify_light(frame, x1, y1, x2, y2)
                result.lights.append(TrafficLightState(
                    state=state, confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    track_id=det.get("id"),
                    detection_confidence=det.get("conf", 0.0),
                ))

        # Unknown readings must not mask a valid state from another signal.
        recognized = [
            light for light in result.lights
            if light.state != "UNKNOWN" and light.confidence >= self.min_confidence
        ]
        if recognized:
            best = max(recognized, key=lambda light: light.confidence)
            result.dominant_light = best.state

        return result

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, result: TrafficResult) -> np.ndarray:
        """Draw traffic-light state and sign count onto frame."""
        STATE_COLORS = {
            "RED":     (0, 0, 255),
            "YELLOW":  (0, 220, 255),
            "GREEN":   (0, 230, 0),
            "UNKNOWN": (180, 180, 180),
        }

        for light in result.lights:
            if light.bbox is None:
                continue
            x1, y1, x2, y2 = light.bbox
            color = STATE_COLORS.get(light.state, (180, 180, 180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = (
                f"TL:{light.state} c{light.confidence:.0%}/"
                f"d{light.detection_confidence:.0%}"
            )
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # HUD line
        tl_text = (
            f"TrafficLight: {result.dominant_light} "
            f"({len(result.lights)} detected)"
        )
        tl_color = STATE_COLORS.get(result.dominant_light, (180, 180, 180))
        cv2.putText(frame, tl_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tl_color, 2)

        if result.signs_detected:
            cv2.putText(frame, f"Signs: {result.signs_detected}",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 50), 2)
        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _classify_light(self, frame, x1, y1, x2, y2):
        """
        Returns (state: str, confidence: float).
        """
        # Use the entire bounding box
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return "UNKNOWN", 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_px = hsv.shape[0] * hsv.shape[1]

        masks = {
            "RED":    cv2.bitwise_or(
                          cv2.inRange(hsv, *_RED_1),
                          cv2.inRange(hsv, *_RED_2)),
            "YELLOW": cv2.inRange(hsv, *_YELLOW),
            "GREEN":  cv2.inRange(hsv, *_GREEN),
        }

        counts = {state: cv2.countNonZero(mask) for state, mask in masks.items()}
        colored_px = sum(counts.values())
        color_fraction = colored_px / max(total_px, 1)
        if color_fraction < self.min_color_pixel_fraction or colored_px == 0:
            return "UNKNOWN", 0.0

        best_state = max(counts, key=counts.get)
        dominance = counts[best_state] / colored_px
        if dominance < self.min_color_dominance:
            return "UNKNOWN", dominance
        return best_state, dominance
