import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.dashboard import api


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        with api._jobs_lock:
            api._jobs.clear()
        self.client = TestClient(api.app)

    def test_health_reports_project_identity_and_model_availability(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["project"], "Astra ADAS")
        self.assertIn("checkpoint_available", response.json())

    def test_video_path_traversal_is_rejected(self):
        response = self.client.get("/api/videos/../config.yaml/file")

        self.assertIn(response.status_code, (400, 404))

    def test_live_endpoint_rejects_non_image_payload_without_loading_detector(self):
        with patch.object(api, "_detector", None):
            response = self.client.post(
                "/api/live/detect",
                content=b"not an image",
                headers={"Content-Type": "image/jpeg"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid image", response.json()["detail"])
        self.assertIsNone(api._detector)

    def test_run_requires_exactly_one_video_source(self):
        response = self.client.post("/api/runs", data={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("exactly one", response.json()["detail"])

    def test_video_upload_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/api/runs",
            files={"upload": ("notes.txt", b"not a video", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("MP4, AVI, or MOV", response.json()["detail"])

    def test_project_video_run_is_queued_and_scheduled(self):
        with patch.object(api, "_process_run") as process_run:
            response = self.client.post(
                "/api/runs",
                data={"video_name": "test_data_1.mp4", "max_frames": "25"},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(response.json()["source_name"], "test_data_1.mp4")
        process_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
