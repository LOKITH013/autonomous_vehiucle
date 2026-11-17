"""
M3 — IDD object detection and ByteTrack association (Sec 7).

Thin wrapper around Ultralytics YOLO. The checkpoint, input resolution, and
thresholds come from config.yaml. Also exposes `.track()` so this same object drives M8 (tracking)
via Ultralytics' built-in ByteTrack integration — no separate tracker code
needed unless you want to swap in DeepSORT later.
"""
from ultralytics import YOLO


class Detector:
    def __init__(self, cfg):
        self.model = YOLO(cfg["detection"]["weights"])
        self.conf = cfg["detection"]["conf_threshold"]
        self.iou = cfg["detection"]["iou_threshold"]
        self.imgsz = cfg["detection"].get("imgsz", 640)
        self.class_conf_thresholds = cfg["detection"].get(
            "class_conf_thresholds", {}
        )
        self.track_conf = cfg["detection"].get(
            "track_conf_threshold",
            min([self.conf, *self.class_conf_thresholds.values()]),
        )
        self.relevant_classes = set(cfg["detection"]["relevant_classes"])
        self.tracker_yaml = cfg["tracking"]["tracker_yaml"]
        self.class_names = self.model.names  # id -> name, from loaded weights
        self._raw_candidates = []
        self.max_lost_frames = cfg.get("tracking", {}).get("max_lost_frames", 30)
        self._special_tracks = {}
        self._next_special_track_id = 1_000_000
        self.model.add_callback(
            "on_predict_postprocess_end", self._capture_raw_candidates
        )

    def _capture_raw_candidates(self, predictor):
        self._raw_candidates = []
        for result in predictor.results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                self._raw_candidates.append({
                    "id": None,
                    "cls_name": self.class_names.get(cls_id, str(cls_id)),
                    "conf": float(box.conf[0]),
                    "xyxy": box.xyxy[0].tolist(),
                })

    def track_frame(self, frame):
        """
        Run detection + ByteTrack association on a single frame.
        Returns a list of dicts: {id, cls_name, conf, xyxy}
        `id` is None if the tracker hasn't assigned a stable ID yet.
        """
        self._raw_candidates = []
        for track in self._special_tracks.values():
            track["missed"] += 1
        results = self.model.track(
            frame,
            persist=True,
            conf=self.track_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker_yaml,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                if self.relevant_classes and cls_name not in self.relevant_classes:
                    continue
                conf = float(box.conf[0])
                if conf < self.class_conf_thresholds.get(cls_name, self.conf):
                    continue
                track_id = int(box.id[0]) if box.id is not None else None
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                detections.append({
                    "id": track_id,
                    "cls_name": cls_name,
                    "conf": conf,
                    "xyxy": xyxy,
                })

        for candidate in self._raw_candidates:
            cls_name = candidate["cls_name"]
            if cls_name not in self.class_conf_thresholds:
                continue
            if candidate["conf"] < self.class_conf_thresholds[cls_name]:
                continue
            if any(
                detection["cls_name"] == cls_name
                and self._box_iou(detection["xyxy"], candidate["xyxy"]) >= 0.5
                for detection in detections
            ):
                continue
            detections.append(candidate)

        self._assign_special_track_ids(detections)
        self._special_tracks = {
            track_id: track
            for track_id, track in self._special_tracks.items()
            if track["missed"] <= self.max_lost_frames
        }
        return detections

    def _assign_special_track_ids(self, detections):
        used_track_ids = set()
        for detection in detections:
            if detection["cls_name"] not in self.class_conf_thresholds:
                continue
            matches = [
                (self._box_iou(track["xyxy"], detection["xyxy"]), track_id, track)
                for track_id, track in self._special_tracks.items()
                if track["cls_name"] == detection["cls_name"]
                and track["missed"] <= self.max_lost_frames
                and track_id not in used_track_ids
            ]
            best_match = max(matches, default=(0.0, None, None), key=lambda item: item[0])
            if best_match[0] >= 0.1:
                track_id, track = best_match[1], best_match[2]
                track["xyxy"] = detection["xyxy"]
                track["missed"] = 0
            else:
                track_id = self._next_special_track_id
                self._next_special_track_id += 1
                self._special_tracks[track_id] = {
                    "cls_name": detection["cls_name"],
                    "xyxy": detection["xyxy"],
                    "missed": 0,
                }
            detection["id"] = track_id
            used_track_ids.add(track_id)

    @staticmethod
    def _box_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0
