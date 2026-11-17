import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from src.evaluation.benchmark_m4 import (
    AP_CONF_THRESHOLD,
    IDD_CLASSES,
    calculate_ap,
    evaluate_model,
)


class BenchmarkMetricsTests(unittest.TestCase):
    def test_perfect_detection_has_coco_style_ap(self):
        ap = calculate_ap(np.array([1.0]), np.array([1.0]))

        self.assertAlmostEqual(ap, 0.995, places=3)

    def test_absent_classes_are_undefined_and_operating_metrics_use_fixed_confidence(self):
        class FakeModel:
            def predict(self, image, conf, imgsz, verbose):
                assert conf == AP_CONF_THRESHOLD
                assert imgsz == 640
                assert image.shape == (100, 100, 3)
                assert verbose is False
                box = SimpleNamespace(
                    cls=np.array([0]),
                    conf=np.array([0.9]),
                    xyxy=np.array([[25.0, 25.0, 75.0, 75.0]]),
                )
                return [SimpleNamespace(boxes=[box])]

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "sample.jpg")
            label_path = os.path.join(temp_dir, "sample.txt")
            self.assertTrue(cv2.imwrite(image_path, np.zeros((100, 100, 3), dtype=np.uint8)))
            with open(label_path, "w", encoding="utf-8") as label_file:
                label_file.write("0 0.5 0.5 0.5 0.5\n")

            metrics, _ = evaluate_model(
                FakeModel(),
                [image_path],
                temp_dir,
            )

        animal = metrics["animal"]
        self.assertEqual(animal["gt_count"], 1)
        self.assertAlmostEqual(animal["precision"], 1.0)
        self.assertAlmostEqual(animal["recall"], 1.0)
        self.assertAlmostEqual(animal["f1"], 1.0)
        self.assertAlmostEqual(animal["ap50"], 0.995, places=3)
        self.assertTrue(
            all(metrics[name]["ap50"] is None for name in IDD_CLASSES[1:])
        )

    def test_class_specific_operating_threshold_changes_precision_recall(self):
        class FakeModel:
            def predict(self, image, conf, imgsz, verbose):
                boxes = [
                    SimpleNamespace(
                        cls=np.array([0]),
                        conf=np.array([0.1]),
                        xyxy=np.array([[25.0, 25.0, 75.0, 75.0]]),
                    ),
                    SimpleNamespace(
                        cls=np.array([0]),
                        conf=np.array([0.9]),
                        xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
                    ),
                ]
                return [SimpleNamespace(boxes=boxes)]

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "sample.jpg")
            label_path = os.path.join(temp_dir, "sample.txt")
            self.assertTrue(cv2.imwrite(image_path, np.zeros((100, 100, 3), dtype=np.uint8)))
            with open(label_path, "w", encoding="utf-8") as label_file:
                label_file.write("0 0.5 0.5 0.5 0.5\n")

            metrics, _ = evaluate_model(
                FakeModel(),
                [image_path],
                temp_dir,
                operating_conf_thresholds={"animal": 0.05},
            )

        self.assertAlmostEqual(metrics["animal"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["animal"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
