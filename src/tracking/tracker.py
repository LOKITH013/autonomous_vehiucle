"""
M8 — Object tracking (Sec 11).

ByteTrack is already wired in via `Detector.track_frame()` (Ultralytics'
built-in `model.track(..., tracker="bytetrack.yaml")`), which is the
"initial tracker" the spec calls for. This file is the extension point if
you later want to swap in DeepSORT (Sec 11: "DeepSORT may be investigated
as an alternative") without touching pipeline code.

To use DeepSORT instead:
    pip install deep-sort-realtime
    from deep_sort_realtime.deepsort_tracker import DeepSort
and implement DeepSortTracker below with the same `.update(detections)`
interface, then have the pipeline call it on raw (untracked) detections
from `model.predict()` instead of `model.track()`.
"""


class DeepSortTracker:
    """Placeholder — not implemented. See module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSORT integration is a documented future option (Sec 11), "
            "not yet implemented. Use Detector.track_frame() (ByteTrack) "
            "for now."
        )

    def update(self, detections):
        raise NotImplementedError
