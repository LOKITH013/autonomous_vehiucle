"""
M9 — Depth Map Distance Estimation.

Replaces the pinhole camera bounding-box proxy with a monocular depth network
(MiDaS) for distance estimation. Since MiDaS outputs relative inverse depth,
we use a scaling factor to roughly calibrate it to metric distance.
"""

import cv2
import torch
import numpy as np

class DepthEstimator:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("depth", {})
        self.scale_factor = self.cfg.get("scale_factor", 1000.0)
        self.use_midas = self.cfg.get("use_midas", True)
        
        if self.use_midas:
            print("Loading MiDaS small model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load MiDaS Small for CPU-friendly inference
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
        else:
            self.model = None

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs MiDaS on the frame and returns a relative depth map.
        (Higher values = closer to camera).
        """
        if not self.use_midas:
            return None
            
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize prediction to match original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def get_object_distance(self, depth_map: np.ndarray, bbox: list) -> float:
        """
        Estimates the metric distance to an object using the depth map.
        Extracts the median depth within the object's bounding box and
        scales it to meters.
        """
        if depth_map is None:
            return None
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Ensure bbox is within frame bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        # Extract the depth region for the bounding box
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # Use median to ignore outliers (e.g., background pixels in the bbox)
        median_depth = np.median(roi)
        
        # MiDaS outputs relative inverse depth (disparity).
        # distance = scale_factor / inverse_depth
        if median_depth <= 0:
            return None
            
        distance_m = self.scale_factor / median_depth
        return distance_m

    def apply_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Converts the depth map into a color heatmap for visualization.
        """
        if depth_map is None:
            return None
            
        # Normalize to 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max == depth_min:
            return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
            
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        depth_uint8 = (255 * depth_norm).astype(np.uint8)
        
        # Apply colormap (INFERNO looks good for depth)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return depth_color
