"""Video person tracking and quality-based observation selection.

The module owns no model repository code.  It uses the existing YOLO pose
weights from SapiensID, assigns persistent IDs, and produces stable per-track
artifacts consumed by SapiensID, OpenPAR, and PoseC3D adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ..core.io import write_json
from ..core.tracks import BBox, SpatialTemporalTracker


@dataclass
class Observation:
    frame_index: int
    timestamp: float
    bbox: BBox
    confidence: float
    quality: float
    visibility: dict[str, bool]
    keypoints: np.ndarray = field(repr=False)
    keypoint_scores: np.ndarray = field(repr=False)
    crop: np.ndarray = field(repr=False)


@dataclass
class TrackBuffer:
    track_id: str
    first_seen: float
    last_seen: float
    observations: list[Observation] = field(default_factory=list)
    timeline: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation, keep_best: int) -> None:
        self.last_seen = observation.timestamp
        self.timeline.append(observation)
        self.observations.append(observation)
        self.observations.sort(key=lambda item: item.quality, reverse=True)
        del self.observations[keep_best:]


def _visibility(keypoint_scores: list[float]) -> dict[str, bool]:
    visible = lambda indices: bool(keypoint_scores) and max(
        keypoint_scores[index] for index in indices
    ) >= 0.30
    return {
        "upper_body": visible([5, 6]),
        "lower_body": visible([11, 12, 13, 14, 15, 16]),
        "carried_items": visible([7, 8, 9, 10]),
        "face": visible([0, 1, 2, 3, 4]),
    }


def _quality(crop: np.ndarray, confidence: float, visibility: dict[str, bool], frame_area: int) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 1.0)
    brightness = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    area = min(float(crop.shape[0] * crop.shape[1]) / max(frame_area * 0.15, 1.0), 1.0)
    body = (float(visibility["upper_body"]) + float(visibility["lower_body"])) / 2.0
    face = float(visibility["face"])
    return round(0.30 * confidence + 0.25 * sharpness + 0.15 * exposure + 0.15 * area + 0.10 * body + 0.05 * face, 6)


def _clip_box(box: BBox, width: int, height: int, padding: float = 0.08) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def scan_video(
    video_path: Path,
    output_dir: Path,
    yolo_weights: Path,
    detection_fps: float = 5.0,
    confidence: float = 0.30,
    max_detections: int = 20,
    keep_best: int = 3,
) -> dict:
    """Detect, track, and save the strongest observations for every unique person."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(source_fps / max(detection_fps, 0.1)))

    model = YOLO(str(yolo_weights))
    tracker = SpatialTemporalTracker(minimum_iou=0.20, max_frame_gap=stride * 3)
    tracks: dict[str, TrackBuffer] = {}
    peak_snapshot: np.ndarray | None = None
    peak_boxes: list[tuple[str, BBox]] = []
    sampled_frames = 0
    frame_index = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % stride:
            continue
        sampled_frames += 1
        result = model.predict(
            frame,
            conf=confidence,
            iou=0.70,
            imgsz=640,
            max_det=max_detections,
            verbose=False,
        )[0]
        boxes = [tuple(map(float, box.tolist())) for box in result.boxes.xyxy.cpu()]
        track_ids = tracker.update(frame_index, boxes)
        current: list[tuple[str, BBox]] = []
        for index, (track_id, box, score) in enumerate(zip(track_ids, boxes, result.boxes.conf.cpu().tolist())):
            x1, y1, x2, y2 = _clip_box(box, width, height)
            crop = frame[y1:y2, x1:x2].copy()
            kp_scores: list[float] = []
            kp_xy = np.zeros((17, 2), dtype=np.float32)
            if result.keypoints is not None and result.keypoints.conf is not None:
                kp_scores = [float(value) for value in result.keypoints.conf[index].cpu().tolist()]
                kp_xy = result.keypoints.xy[index].cpu().numpy().astype(np.float32)
            visibility = _visibility(kp_scores)
            timestamp = frame_index / source_fps
            observation = Observation(
                frame_index=frame_index,
                timestamp=timestamp,
                bbox=box,
                confidence=float(score),
                quality=_quality(crop, float(score), visibility, width * height),
                visibility=visibility,
                keypoints=kp_xy,
                keypoint_scores=np.asarray(kp_scores or [0.0] * 17, dtype=np.float32),
                crop=crop,
            )
            if track_id not in tracks:
                tracks[track_id] = TrackBuffer(track_id, timestamp, timestamp)
            tracks[track_id].add(observation, keep_best)
            current.append((track_id, box))
        if len(current) > len(peak_boxes):
            peak_snapshot = frame.copy()
            peak_boxes = current

    capture.release()
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_root = output_dir / "crops"
    clip_root = output_dir / "clips"
    crop_root.mkdir(parents=True, exist_ok=True)
    clip_root.mkdir(parents=True, exist_ok=True)

    peak_bbox_by_track = {track_id: list(box) for track_id, box in peak_boxes}
    track_records = []
    for track_id in sorted(tracks, key=lambda value: int(value) if value.isdigit() else value):
        track = tracks[track_id]
        crop_paths = []
        for rank, observation in enumerate(track.observations, start=1):
            crop_path = crop_root / f"t{int(track_id):04d}_r{rank:02d}.jpg"
            cv2.imwrite(str(crop_path), observation.crop)
            crop_paths.append(str(crop_path.resolve()))

        clip_path = clip_root / f"track_{int(track_id):04d}.avi"
        usable = [item for item in track.timeline if item.crop.size]
        if usable:
            size = (256, 256)
            writer = cv2.VideoWriter(
                str(clip_path), cv2.VideoWriter_fourcc(*"MJPG"),
                max(1.0, min(detection_fps, source_fps)), size,
            )
            for item in usable:
                canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
                crop = item.crop
                scale = min(256 / crop.shape[1], 256 / crop.shape[0])
                resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
                y = (256 - resized.shape[0]) // 2
                x = (256 - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                writer.write(canvas)
            writer.release()

        best = track.observations[0]
        track_records.append({
            "track_id": track_id,
            "first_seen": round(track.first_seen, 3),
            "last_seen": round(track.last_seen, 3),
            "duration": round(max(0.0, track.last_seen - track.first_seen), 3),
            "observations": len(track.timeline),
            "best_quality": best.quality,
            "best_bbox_xyxy": list(best.bbox),
            "snapshot_bbox_xyxy": peak_bbox_by_track.get(track_id),
            "visibility": best.visibility,
            "crop_paths": crop_paths,
            "clip_path": str(clip_path.resolve()) if clip_path.is_file() else None,
        })

    snapshot_path = output_dir / "peak_snapshot.jpg"
    if peak_snapshot is not None:
        for track_id, box in peak_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(peak_snapshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(peak_snapshot, f"Track {track_id}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imwrite(str(snapshot_path), peak_snapshot)

    # Preserve all simultaneously tracked skeletons in full-frame coordinates.
    # PoseC3D can then recognize interactions that disappear from person crops.
    sampled_indices = sorted({item.frame_index for track in tracks.values() for item in track.timeline})
    frame_to_index = {frame: index for index, frame in enumerate(sampled_indices)}
    ordered_track_ids = sorted(tracks, key=lambda value: int(value) if value.isdigit() else value)
    scene_keypoints = np.zeros((len(ordered_track_ids), len(sampled_indices), 17, 2), dtype=np.float32)
    scene_scores = np.zeros((len(ordered_track_ids), len(sampled_indices), 17), dtype=np.float32)
    for person_index, track_id in enumerate(ordered_track_ids):
        for item in tracks[track_id].timeline:
            time_index = frame_to_index[item.frame_index]
            scene_keypoints[person_index, time_index] = item.keypoints
            scene_scores[person_index, time_index] = item.keypoint_scores
    scene_pose_path = output_dir / "scene_pose.json"
    write_json(scene_pose_path, {
        "img_shape": [height, width],
        "original_shape": [height, width],
        "total_frames": len(sampled_indices),
        "track_ids": ordered_track_ids,
        "source_frame_indices": sampled_indices,
        "keypoint": scene_keypoints.tolist(),
        "keypoint_score": scene_scores.tolist(),
    })

    report = {
        "video": str(video_path.resolve()),
        "source_fps": source_fps,
        "source_frames": total_frames,
        "sample_stride": stride,
        "sampled_frames": sampled_frames,
        "unique_tracks": len(track_records),
        "peak_visible_count": len(peak_boxes),
        "snapshot": str(snapshot_path.resolve()) if snapshot_path.is_file() else None,
        "scene_pose_path": str(scene_pose_path.resolve()),
        "tracks": track_records,
    }
    write_json(output_dir / "tracks.json", report)
    return report
