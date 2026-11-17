"""
M7 — Hazard Detection / Obstacle in Path (Sec 10, Sec 16).

Determines whether a detected object is physically in the ego vehicle's
future path. This converts a generic "object detection" into a specific
"hazard warning".

Uses the lane geometry (M5) if available. If lane lines are missing,
falls back to a standard central corridor assumption.
"""

from typing import Dict, Any

class HazardDetector:
    def __init__(self, cfg: dict):
        hc = cfg.get("hazard", {})
        # Fallback corridor: fraction of frame width considered "in path"
        # if lane detection fails (default: center 40%)
        self.fallback_corridor_width_frac = hc.get("fallback_corridor_width_frac", 0.4)

    def check_in_path(self, det: dict, lane_result: Any, frame_width: int, frame_height: int) -> bool:
        """
        Check if the object's footprint is inside the ego vehicle's path.
        
        det: Detection dictionary with 'xyxy'
        lane_result: LaneResult object from M5
        """
        x1, y1, x2, y2 = det["xyxy"]
        
        # Object footprint is the center-bottom of the bounding box
        footprint_x = (x1 + x2) / 2.0
        footprint_y = y2
        
        # Scenario A: We have valid lane lines from M5
        if lane_result and lane_result.left_line and lane_result.right_line:
            lx_bot, ly_bot, lx_top, ly_top = lane_result.left_line
            rx_bot, ry_bot, rx_top, ry_top = lane_result.right_line
            
            # Interpolate lane x-coordinates at the object's y level
            # Using simple linear interpolation: x = x_bot + (y - y_bot) * (x_top - x_bot) / (y_top - y_bot)
            
            def get_x_at_y(x_b, y_b, x_t, y_t, target_y):
                if y_t == y_b:  # Avoid div by zero
                    return x_b
                return x_b + (target_y - y_b) * (x_t - x_b) / (y_t - y_b)

            # Only check if the object is reasonably within the vertical bounds of the road
            if footprint_y >= min(ly_top, ry_top):
                left_lane_x = get_x_at_y(lx_bot, ly_bot, lx_top, ly_top, footprint_y)
                right_lane_x = get_x_at_y(rx_bot, ry_bot, rx_top, ry_top, footprint_y)
                
                # Check if footprint is between the lines
                if left_lane_x <= footprint_x <= right_lane_x:
                    return True
                return False

        # Scenario B: No lane lines (fallback to central corridor)
        center_x = frame_width / 2.0
        corridor_half_width = (frame_width * self.fallback_corridor_width_frac) / 2.0
        
        if (center_x - corridor_half_width) <= footprint_x <= (center_x + corridor_half_width):
            return True
            
        return False
