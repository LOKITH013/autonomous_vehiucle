import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.motion.motion import MotionTracker
from src.perception.detector import Detector
from src.traffic.traffic_detector import TrafficDetector
from src.lane.lane_detector import LaneDetector


class TestMotionEstimates(unittest.TestCase):
    def test_motion_uses_video_time_and_reports_image_direction(self):
        tracker = MotionTracker()

        first = tracker.observe(7, 20.0, center_xy=(100, 200), timestamp=0.0)
        second = tracker.observe(7, 15.0, center_xy=(120, 190), timestamp=1.0)

        self.assertIsNone(first.closing_speed_mps)
        self.assertAlmostEqual(second.closing_speed_mps, 5.0)
        self.assertAlmostEqual(second.image_velocity_x_px_s, 20.0)
        self.assertAlmostEqual(second.image_velocity_y_px_s, -10.0)
        self.assertEqual(second.image_direction, "UP-RIGHT")

    def test_motion_reports_stationary_below_deadband(self):
        tracker = MotionTracker(direction_deadband_px_s=5.0)
        tracker.observe(2, 10.0, center_xy=(20, 20), timestamp=0.0)
        estimate = tracker.observe(2, 10.0, center_xy=(22, 21), timestamp=1.0)
        self.assertEqual(estimate.image_direction, "STATIONARY")

    def test_legacy_update_returns_closing_speed(self):
        tracker = MotionTracker()
        self.assertIsNone(tracker.update(1, 10.0, timestamp=0.0))
        self.assertAlmostEqual(tracker.update(1, 8.0, timestamp=1.0), 2.0)
        self.assertIsNone(tracker.update(1, None, timestamp=2.0))


class TestTrafficSignalDetection(unittest.TestCase):
    def setUp(self):
        self.detector = TrafficDetector({
            "traffic": {
                "min_light_confidence": 0.1,
                "min_bbox_area_px": 50,
                "min_color_pixel_fraction": 0.01,
                "min_color_dominance": 0.55,
            }
        })

    def test_clips_boxes_counts_signals_and_ignores_unknown_dominant(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = (0, 255, 0)
        detections = [
            {"cls_name": "traffic light", "xyxy": [0, 0, 20, 20], "conf": 0.8, "id": 1},
            {"cls_name": "traffic light", "xyxy": [-10, -10, 110, 110], "conf": 0.9, "id": 2},
            {"cls_name": "traffic sign", "xyxy": [30, 30, 60, 60], "conf": 0.7, "id": 3},
        ]

        result = self.detector.process(frame, detections)

        self.assertEqual(len(result.lights), 2)
        self.assertEqual(result.signs_detected, 1)
        self.assertEqual(result.dominant_light, "GREEN")
        self.assertEqual(result.lights[1].bbox, [0, 0, 100, 100])
        self.assertEqual(result.lights[1].track_id, 2)

    def test_ambiguous_color_is_unknown(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 20:50] = (0, 0, 255)
        frame[20:80, 50:80] = (0, 255, 0)
        detections = [
            {"cls_name": "traffic light", "xyxy": [0, 0, 100, 100], "conf": 0.9, "id": 1}
        ]

        result = self.detector.process(frame, detections)

        self.assertEqual(result.lights[0].state, "UNKNOWN")
        self.assertEqual(result.dominant_light, "UNKNOWN")

    def test_dim_signal_color_is_recognized(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = (0, 100, 0)
        detection = [
            {"cls_name": "traffic light", "xyxy": [40, 40, 60, 60], "conf": 0.8, "id": 4}
        ]

        result = self.detector.process(frame, detection)

        self.assertEqual(result.dominant_light, "GREEN")


class TestDetectionThresholds(unittest.TestCase):
    def test_signal_classes_use_lower_threshold_without_lowering_vehicle_threshold(self):
        class FakeBox:
            def __init__(self, cls_id, confidence, track_id):
                self.cls = np.array([cls_id])
                self.conf = np.array([confidence])
                self.id = np.array([track_id])
                self.xyxy = np.array([[0, 0, 20, 20]], dtype=float)

        raw_boxes = [
            FakeBox(0, 0.10, 1),
            FakeBox(1, 0.07, 2),
            FakeBox(2, 0.09, 3),
        ]
        track = Mock()
        callbacks = {}
        def fake_track(*args, **kwargs):
            callbacks["on_predict_postprocess_end"](
                SimpleNamespace(results=[SimpleNamespace(boxes=raw_boxes)])
            )
            return [SimpleNamespace(boxes=[FakeBox(0, 0.3, 10)])]

        track.side_effect = fake_track
        model = SimpleNamespace(
            names={0: "car", 1: "traffic light", 2: "traffic sign"},
            track=track,
            add_callback=lambda event, callback: callbacks.setdefault(event, callback),
        )
        cfg = {
            "detection": {
                "weights": "unused.pt",
                "conf_threshold": 0.22,
                "track_conf_threshold": 0.01,
                "class_conf_thresholds": {
                    "traffic light": 0.03,
                    "traffic sign": 0.05,
                },
                "iou_threshold": 0.55,
                "relevant_classes": ["car", "traffic light", "traffic sign"],
            },
            "tracking": {"tracker_yaml": "unused.yaml"},
        }

        with patch("src.perception.detector.YOLO", return_value=model):
            detector = Detector(cfg)
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            detections = detector.track_frame(frame)
            next_detections = detector.track_frame(frame)

        self.assertEqual(
            [det["cls_name"] for det in detections],
            ["car", "traffic light", "traffic sign"],
        )
        self.assertEqual(track.call_args.kwargs["conf"], 0.01)
        self.assertIsNotNone(detections[1]["id"])
        self.assertEqual(detections[1]["id"], next_detections[1]["id"])


class TestLaneStability(unittest.TestCase):
    def test_lane_lines_are_retained_for_brief_detection_dropout(self):
        detector = LaneDetector({"lane": {"processing_width": 640, "max_missing_frames": 2}})
        detected_lines = np.array([[[130, 400, 240, 250]], [[510, 400, 370, 250]]])
        with patch("src.lane.lane_detector.cv2.HoughLinesP", side_effect=[detected_lines, None, None, None]):
            first = detector.detect(np.zeros((400, 640, 3), dtype=np.uint8))
            second = detector.detect(np.zeros((400, 640, 3), dtype=np.uint8))
            third = detector.detect(np.zeros((400, 640, 3), dtype=np.uint8))
            fourth = detector.detect(np.zeros((400, 640, 3), dtype=np.uint8))

        self.assertTrue(first.lane_detected)
        self.assertTrue(second.lane_detected)
        self.assertTrue(third.lane_detected)
        self.assertFalse(fourth.lane_detected)
        self.assertAlmostEqual(first.lateral_deviation, 0.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
