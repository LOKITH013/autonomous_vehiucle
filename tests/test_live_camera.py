import unittest

import numpy as np

from src.dashboard.live_camera import draw_live_detections


class TestLiveCameraOverlay(unittest.TestCase):
    def test_draws_clipped_detection_box_and_preserves_frame_shape(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        detections = [{
            "cls_name": "car",
            "conf": 0.91,
            "xyxy": [-5, 10, 40, 50],
        }]

        result = draw_live_detections(frame, detections)

        self.assertEqual(result.shape, (80, 100, 3))
        self.assertTrue(np.array_equal(result[10, 0], [0, 255, 0]))

    def test_empty_detections_show_status_text(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        original = frame.copy()

        result = draw_live_detections(frame, [])

        self.assertEqual(result.shape, original.shape)
        self.assertFalse(np.array_equal(result, original))
        self.assertTrue(np.any(result[15:30, 5:90]))


if __name__ == "__main__":
    unittest.main()
