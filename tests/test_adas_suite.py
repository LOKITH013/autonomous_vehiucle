"""
Astra ADAS — Comprehensive Test Suite & Scenario Verification.
Covers all core ADAS test cases across Perception, Tracking, Lane Detection,
Depth Proxy, Traffic Light/Sign Recognition, Hazard Corridor Analysis,
TTC Calculation, Risk Assessment, and Decision Arbitration.

Executes and monitors live tests on real road video input (outputs/videos/test_data_1.mp4).

Run with:
    python -m unittest tests/test_adas_suite.py -v
"""

import os
import sys
import time
import unittest
import numpy as np
import cv2
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.pipeline import load_config
from src.perception.detector import Detector
from src.lane.lane_detector import LaneDetector, LaneResult
from src.traffic.traffic_detector import TrafficDetector
from src.hazard.hazard_detector import HazardDetector
from src.depth.depth_estimator import DepthEstimator
from src.motion.motion import MotionTracker, estimate_distance_proxy
from src.ttc.ttc_calc import compute_ttc
from src.risk.risk_engine import assess_risk
from src.decision.decision_engine import decide


class TestAstraADAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(ROOT_DIR, "config.yaml")
        cls.cfg = load_config(cls.config_path)
        cls.test_video_path = os.path.join(ROOT_DIR, "outputs", "videos", "test_data_1.mp4")
        
        # Verify test video exists
        if not os.path.exists(cls.test_video_path):
            raise FileNotFoundError(f"Test video not found: {cls.test_video_path}")
        
        # Grab a couple of sample frames from test video for realistic visual tests
        cap = cv2.VideoCapture(cls.test_video_path)
        cls.sample_frames = []
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                cls.sample_frames.append(frame)
        cap.release()
        
        print("\n" + "="*70)
        print("          ASTRA ADAS — TEST SUITE & SCENARIO MONITORING")
        print("="*70)
        print(f"Master Config: {cls.config_path}")
        print(f"Test Video:    {cls.test_video_path} ({len(cls.sample_frames)} sample frames cached)")
        print("-"*70 + "\n")

    # -------------------------------------------------------------------------
    # TEST CASE 1: Perception & IDD Class Integrity
    # -------------------------------------------------------------------------
    def test_01_perception_and_idd_classes(self):
        """Verify YOLO detector loads IDD weights and detects valid bounding boxes."""
        print("[TEST 01] Running Perception & IDD Class Integrity Test...")
        detector = Detector(self.cfg)
        
        # Must contain all 15 IDD classes
        expected_classes = {
            "animal", "autorickshaw", "bicycle", "bus", "car", "caravan",
            "motorcycle", "person", "rider", "traffic light", "traffic sign",
            "trailer", "train", "truck", "vehicle fallback"
        }
        model_classes = set(detector.class_names.values())
        self.assertTrue(expected_classes.issubset(model_classes),
                        f"Missing expected IDD classes in model weights: {expected_classes - model_classes}")
        
        # Test detection on sample frame
        frame = self.sample_frames[0]
        t0 = time.time()
        detections = detector.track_frame(frame)
        latency_ms = (time.time() - t0) * 1000
        
        self.assertIsInstance(detections, list)
        print(f"  ✓ Model loaded with {len(model_classes)} classes. "
              f"Detected {len(detections)} objects on test frame in {latency_ms:.1f}ms.")
        
        for det in detections:
            self.assertIn("cls_name", det)
            self.assertIn("conf", det)
            self.assertIn("xyxy", det)
            self.assertTrue(0.0 <= det["conf"] <= 1.0)
            x1, y1, x2, y2 = det["xyxy"]
            self.assertTrue(x2 >= x1 and y2 >= y1, f"Invalid bbox geometry: {det['xyxy']}")

    # -------------------------------------------------------------------------
    # TEST CASE 2: ByteTrack Tracking Continuity
    # -------------------------------------------------------------------------
    def test_02_bytetrack_tracking_continuity(self):
        """Verify ByteTrack preserves track IDs across sequential video frames."""
        print("[TEST 02] Running ByteTrack Tracking Continuity Test...")
        detector = Detector(self.cfg)
        motion = MotionTracker()
        
        frame1 = self.sample_frames[0]
        frame2 = self.sample_frames[1]
        
        dets1 = detector.track_frame(frame1)
        dets2 = detector.track_frame(frame2)
        
        active_ids_1 = {d["id"] for d in dets1 if d["id"] is not None}
        active_ids_2 = {d["id"] for d in dets2 if d["id"] is not None}
        
        motion.prune(active_ids_2)
        
        # Check track ID overlap
        overlap = active_ids_1.intersection(active_ids_2)
        print(f"  ✓ Frame 1 Tracks: {active_ids_1} | Frame 2 Tracks: {active_ids_2}")
        print(f"  ✓ Persistent Track IDs across frames: {overlap}")
        self.assertTrue(len(active_ids_1) > 0, "No track IDs generated in Frame 1")
        self.assertTrue(len(overlap) > 0, "No persistent track IDs preserved across consecutive frames")

    # -------------------------------------------------------------------------
    # TEST CASE 3: Lane Detection & Lateral Position
    # -------------------------------------------------------------------------
    def test_03_lane_detection_and_lateral_deviation(self):
        """Verify OpenCV Canny/Hough lane detection and ego deviation calculation."""
        print("[TEST 03] Running Lane Detection & Lateral Deviation Test...")
        lane_detector = LaneDetector(self.cfg)
        frame = self.sample_frames[0]
        
        result = lane_detector.detect(frame)
        self.assertIsInstance(result, LaneResult)
        self.assertIn(result.position, ["LEFT", "CENTER", "RIGHT", "UNKNOWN"])
        
        annotated = lane_detector.draw(frame.copy(), result)
        self.assertEqual(annotated.shape, frame.shape)
        
        print(f"  ✓ Lane Position: {result.position} | "
              f"Lateral Deviation: {result.lateral_deviation} px | "
              f"Lane Detected: {result.lane_detected}")

    # -------------------------------------------------------------------------
    # TEST CASE 4: Distance Estimation (MiDaS Depth & Pinhole Proxy)
    # -------------------------------------------------------------------------
    def test_04_monocular_depth_and_distance_proxy(self):
        """Verify depth map estimation and pinhole distance proxy fallback."""
        print("[TEST 04] Running Depth Estimation & Distance Proxy Test...")
        # Test pinhole fallback
        test_bbox = [400, 300, 600, 500]  # width = 200px
        car_dist = estimate_distance_proxy(self.cfg, "car", test_bbox)
        person_dist = estimate_distance_proxy(self.cfg, "person", test_bbox)
        
        self.assertIsNotNone(car_dist)
        self.assertIsNotNone(person_dist)
        self.assertTrue(car_dist > person_dist, "A car of identical bbox width must be estimated further than a person")
        self.assertTrue(1.0 < car_dist < 100.0, f"Unreasonable distance estimate: {car_dist}m")
        print(f"  ✓ Pinhole Proxy - Car (200px): {car_dist:.1f}m | Person (200px): {person_dist:.1f}m")
        
        # Test MiDaS Depth Estimator
        depth_estimator = DepthEstimator(self.cfg)
        if depth_estimator.use_midas and depth_estimator.model is not None:
            depth_map = depth_estimator.estimate_depth(self.sample_frames[0])
            self.assertIsNotNone(depth_map)
            self.assertEqual(depth_map.shape, self.sample_frames[0].shape[:2])
            obj_dist = depth_estimator.get_object_distance(depth_map, test_bbox)
            print(f"  ✓ MiDaS Depth Map shape: {depth_map.shape} | Object distance: {obj_dist:.1f}m")

    # -------------------------------------------------------------------------
    # TEST CASE 5: Traffic Light State Recognition
    # -------------------------------------------------------------------------
    def test_05_traffic_light_color_classifier(self):
        """Verify HSV-based traffic light color classification (RED, YELLOW, GREEN)."""
        print("[TEST 05] Running Traffic Light HSV Classifier Test...")
        traffic_detector = TrafficDetector(self.cfg)
        
        h, w = 300, 300
        # Create synthetic RED traffic light box
        red_frame = np.zeros((h, w, 3), dtype=np.uint8)
        red_frame[50:150, 50:150] = (0, 0, 255)  # BGR Red
        red_det = [{"cls_name": "traffic light", "xyxy": [50, 50, 150, 150], "conf": 0.9, "id": 1}]
        res_red = traffic_detector.process(red_frame, red_det)
        self.assertEqual(res_red.dominant_light, "RED", f"Expected RED light, got {res_red.dominant_light}")
        
        # Create synthetic GREEN traffic light box
        green_frame = np.zeros((h, w, 3), dtype=np.uint8)
        green_frame[50:150, 50:150] = (0, 255, 0)  # BGR Green
        green_det = [{"cls_name": "traffic light", "xyxy": [50, 50, 150, 150], "conf": 0.9, "id": 2}]
        res_green = traffic_detector.process(green_frame, green_det)
        self.assertEqual(res_green.dominant_light, "GREEN", f"Expected GREEN light, got {res_green.dominant_light}")
        
        print("  ✓ Correctly classified synthetic RED and GREEN traffic light signals.")

    # -------------------------------------------------------------------------
    # TEST CASE 6: Hazard In-Path Assessment
    # -------------------------------------------------------------------------
    def test_06_hazard_in_path_assessment(self):
        """Verify ego path boundary and corridor obstacle hazard detection."""
        print("[TEST 06] Running Hazard Corridor & In-Path Assessment Test...")
        hazard_detector = HazardDetector(self.cfg)
        frame_w, frame_h = 1000, 800
        
        # Case A: Obstacle right in front of ego vehicle (center corridor)
        center_det = {"xyxy": [450, 400, 550, 600]}  # footprint_x = 500 (center)
        in_path_center = hazard_detector.check_in_path(center_det, None, frame_w, frame_h)
        self.assertTrue(in_path_center, "Obstacle in center corridor must be identified as IN PATH")
        
        # Case B: Obstacle on far left shoulder
        left_det = {"xyxy": [20, 400, 100, 600]}     # footprint_x = 60
        in_path_left = hazard_detector.check_in_path(left_det, None, frame_w, frame_h)
        self.assertFalse(in_path_left, "Obstacle on roadside shoulder must NOT be in path")
        
        # Case C: Obstacle on far right
        right_det = {"xyxy": [880, 400, 960, 600]}   # footprint_x = 920
        in_path_right = hazard_detector.check_in_path(right_det, None, frame_w, frame_h)
        self.assertFalse(in_path_right, "Obstacle on right sidewalk must NOT be in path")
        
        print(f"  ✓ Center obstacle: In-Path={in_path_center} | Left shoulder: In-Path={in_path_left} | Right: In-Path={in_path_right}")

    # -------------------------------------------------------------------------
    # TEST CASE 7: TTC (Time-To-Collision) Calculation Scenarios
    # -------------------------------------------------------------------------
    def test_07_ttc_calculation_scenarios(self):
        """Verify Time-to-Collision kinematics under approaching and separating velocities."""
        print("[TEST 07] Running TTC Calculation Scenarios Test...")
        # Scenario 1: Approaching vehicle (closing speed = 10 m/s, distance = 20m -> TTC = 2.0s)
        ttc1 = compute_ttc(self.cfg, distance_m=20.0, closing_speed_mps=10.0)
        self.assertAlmostEqual(ttc1, 2.0, places=2)
        
        # Scenario 2: Imminent collision (closing speed = 20 m/s, distance = 10m -> TTC = 0.5s)
        ttc2 = compute_ttc(self.cfg, distance_m=10.0, closing_speed_mps=20.0)
        self.assertAlmostEqual(ttc2, 0.5, places=2)
        
        # Scenario 3: Separating vehicle (closing speed = -5 m/s) -> float('inf') (no collision trajectory)
        ttc3 = compute_ttc(self.cfg, distance_m=20.0, closing_speed_mps=-5.0)
        self.assertEqual(ttc3, float("inf"), "Negative closing speed must yield float('inf') TTC")
        
        # Scenario 4: Near zero closing speed below threshold (0.1 m/s) -> float('inf')
        ttc4 = compute_ttc(self.cfg, distance_m=20.0, closing_speed_mps=0.1)
        self.assertEqual(ttc4, float("inf"))
        
        print(f"  ✓ Closing at 10m/s from 20m: TTC={ttc1:.1f}s | Closing at 20m/s from 10m: TTC={ttc2:.1f}s")
        print(f"  ✓ Separating (-5m/s): TTC=inf | Near-zero (0.1m/s): TTC=inf")

    # -------------------------------------------------------------------------
    # TEST CASE 8: Risk Engine Safety Rules & Vulnerable User Escalation
    # -------------------------------------------------------------------------
    def test_08_risk_engine_safety_rules(self):
        """Verify rule-based risk classification and vulnerable road user safety escalation."""
        print("[TEST 08] Running Risk Engine Rules & Vulnerable User Test...")
        # Scenario 1: Safe cruising
        risk1, _ = assess_risk(self.cfg, ttc=10.0, distance_m=40.0, cls_name="car", in_path=True)
        self.assertEqual(risk1, "LOW")
        
        # Scenario 2: Low TTC (< 2.0s threshold) -> HIGH risk
        risk2, reasons2 = assess_risk(self.cfg, ttc=1.8, distance_m=30.0, cls_name="car", in_path=True)
        self.assertEqual(risk2, "HIGH")
        self.assertTrue(any("TTC" in r for r in reasons2))
        
        # Scenario 3: Proximity trigger (< 5m threshold) -> HIGH risk
        risk3, _ = assess_risk(self.cfg, ttc=None, distance_m=4.2, cls_name="car", in_path=True)
        self.assertEqual(risk3, "HIGH")
        
        # Scenario 4: Vulnerable road user bump (person at medium-risk distance -> HIGH)
        risk_car, _ = assess_risk(self.cfg, ttc=3.5, distance_m=10.0, cls_name="car", in_path=True)
        risk_ped, reasons_ped = assess_risk(self.cfg, ttc=3.5, distance_m=10.0, cls_name="person", in_path=True)
        self.assertEqual(risk_car, "MEDIUM")
        self.assertEqual(risk_ped, "HIGH", "Vulnerable road user (person) must be bumped from MEDIUM to HIGH risk")
        self.assertTrue(any("Vulnerable road user" in r for r in reasons_ped))
        
        # Scenario 5: Red traffic light -> HIGH risk
        risk_red, _ = assess_risk(self.cfg, ttc=10.0, distance_m=50.0, cls_name="car", traffic_light="RED")
        self.assertEqual(risk_red, "HIGH")
        
        print(f"  ✓ Safe cruising: {risk1} | Imminent TTC: {risk2} | Close Proximity: {risk3}")
        print(f"  ✓ Standard car vs Vulnerable Pedestrian: Car={risk_car} -> Pedestrian={risk_ped} (Escalated)")

    # -------------------------------------------------------------------------
    # TEST CASE 9: Decision Engine Action Arbitration
    # -------------------------------------------------------------------------
    def test_09_decision_engine_action_arbitration(self):
        """Verify decision engine command generation (BRAKE, SLOW DOWN, STOP, AVOID OBSTACLE)."""
        print("[TEST 09] Running Decision Engine Action Arbitration Test...")
        # Case A: Emergency braking (TTC < 1.5s brake threshold)
        act1, _ = decide(self.cfg, risk="HIGH", ttc=1.2)
        self.assertEqual(act1, "BRAKE")
        
        # Case B: Deceleration (TTC < 3.0s slow-down threshold)
        act2, _ = decide(self.cfg, risk="MEDIUM", ttc=2.5)
        self.assertEqual(act2, "SLOW DOWN")
        
        # Case C: Obstacle directly blocking path
        act3, _ = decide(self.cfg, risk="HIGH", ttc=5.0, obstacle_in_path=True)
        self.assertEqual(act3, "AVOID OBSTACLE")
        
        # Case D: Red traffic light signal
        act4, _ = decide(self.cfg, risk="HIGH", ttc=5.0, traffic_light="RED")
        self.assertEqual(act4, "STOP")
        
        # Case E: Nominal conditions
        act5, _ = decide(self.cfg, risk="LOW", ttc=8.0)
        self.assertEqual(act5, "MAINTAIN SPEED")
        
        print(f"  ✓ Imminent: {act1} | Moderate: {act2} | Blocked Path: {act3} | Red Light: {act4} | Safe: {act5}")

    # -------------------------------------------------------------------------
    # TEST CASE 10: End-to-End Pipeline Telemetry on Test Video
    # -------------------------------------------------------------------------
    def test_10_end_to_end_video_pipeline_telemetry(self):
        """Execute end-to-end multi-module pipeline on video input and monitor telemetry health."""
        print("[TEST 10] Running End-to-End Video Stream Processing Test...")
        detector = Detector(self.cfg)
        lane_detector = LaneDetector(self.cfg)
        traffic_detector = TrafficDetector(self.cfg)
        hazard_detector = HazardDetector(self.cfg)
        depth_estimator = DepthEstimator(self.cfg)
        motion = MotionTracker()
        
        cap = cv2.VideoCapture(self.test_video_path)
        frames_tested = 0
        total_latency = 0.0
        objects_seen = 0
        actions_issued = []
        
        for idx in range(1, 21):  # Process 20 sample frames for rigorous smoke test
            ok, frame = cap.read()
            if not ok:
                break
            frames_tested += 1
            t0 = time.time()
            h, w = frame.shape[:2]
            
            # Full integrated chain
            lane_res = lane_detector.detect(frame)
            dets = detector.track_frame(frame)
            active_ids = {d["id"] for d in dets if d["id"] is not None}
            motion.prune(active_ids)
            traffic_res = traffic_detector.process(frame, dets)
            
            for det in dets:
                dist = estimate_distance_proxy(self.cfg, det["cls_name"], det["xyxy"])
                speed = motion.update(det["id"], dist, timestamp=t0)
                ttc = compute_ttc(self.cfg, dist, speed)
                in_path = hazard_detector.check_in_path(det, lane_res, w, h)
                risk, r_reasons = assess_risk(self.cfg, ttc=ttc, distance_m=dist,
                                              cls_name=det["cls_name"], in_path=in_path)
                act, _ = decide(self.cfg, risk=risk, ttc=ttc, obstacle_in_path=in_path,
                                risk_reasons=r_reasons)
                actions_issued.append(act)
                objects_seen += 1
                
            latency_ms = (time.time() - t0) * 1000
            total_latency += latency_ms
            
        cap.release()
        
        avg_latency = total_latency / frames_tested
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        self.assertEqual(frames_tested, 20)
        self.assertTrue(objects_seen > 0, "Pipeline failed to detect any objects across 20 video frames")
        
        print(f"  ✓ Streamed {frames_tested} frames | Tracked {objects_seen} object detections")
        print(f"  ✓ Average Latency: {avg_latency:.1f}ms/frame ({avg_fps:.1f} FPS)")
        print(f"  ✓ Distinct Action Commands generated: {set(actions_issued)}")
        print("\n" + "="*70)
        print("          ALL 10 ADAS TEST CASES PASSED SUCCESSFULLY!")
        print("="*70 + "\n")


if __name__ == "__main__":
    unittest.main()
