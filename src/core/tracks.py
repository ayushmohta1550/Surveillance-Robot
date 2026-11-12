from __future__ import annotations

from dataclasses import dataclass
from math import hypot


BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    track_id: str
    bbox: BBox
    last_frame: int


class SpatialTemporalTracker:
    """Small deterministic IoU tracker for associating per-person crops across frames.

    Model outputs must be joined through its returned track ID. Identity and
    attributes are never associated merely by their array position.
    """

    def __init__(self, minimum_iou: float = 0.3, max_frame_gap: int = 10):
        self.minimum_iou = minimum_iou
        self.max_frame_gap = max_frame_gap
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1

    def update(self, frame_number: int, detections: list[BBox]) -> list[str]:
        available = {
            track_id: track for track_id, track in self._tracks.items()
            if frame_number - track.last_frame <= self.max_frame_gap
        }
        assignments: list[str] = []
        used: set[str] = set()
        for bbox in detections:
            candidates = [
                (iou(bbox, track.bbox), track_id)
                for track_id, track in available.items() if track_id not in used
            ]
            score, track_id = max(candidates, default=(0.0, ""))
            if score < self.minimum_iou:
                track_id = str(self._next_id)
                self._next_id += 1
            self._tracks[track_id] = _Track(track_id, bbox, frame_number)
            used.add(track_id)
            assignments.append(track_id)
        return assignments
