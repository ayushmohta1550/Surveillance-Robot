"""CPU-only PoseC3D smoke test and adapter.

This module deliberately keeps the upstream PoseC3D/PYSKL repository
read-only.  MediaPipe extracts a single COCO-style skeleton sequence and the
official PYSKL inference API performs temporal action classification.

The supplied NTU-120 checkpoint is a clip classifier.  This adapter is meant
as a CPU proof of concept; multi-person production use will require tracked
per-person pose sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import mmcv
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
POSEC3D_ROOT = WORKSPACE / "PoseC3d"
DEFAULT_CONFIG = (
    POSEC3D_ROOT
    / "configs"
    / "posec3d"
    / "slowonly_r50_ntu120_xsub"
    / "joint.py"
)
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "models"
    / "posec3d"
    / "slowonly_r50_ntu120_xsub_joint.pth"
)
DEFAULT_LABELS = (
    POSEC3D_ROOT / "tools" / "data" / "label_map" / "nturgbd_120.txt"
)
DEFAULT_VIDEO = POSEC3D_ROOT / "demo" / "ntu_sample.avi"
DEFAULT_OUTPUT = WORKSPACE / "output" / "posec3d_cpu_sample.json"

# MediaPipe Pose landmark indices corresponding to COCO's 17-keypoint order.
MEDIAPIPE_TO_COCO = (
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
)

ACTIVITY_PHRASES = {
    "hugging other person": "hugging another person",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official PoseC3D NTU-120 model on one video using CPU."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--pose-json",
        type=Path,
        default=None,
        help="Optional tracked multi-person COCO-17 skeleton sequence.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit; zero processes the complete video.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum top-1 probability required to publish an activity.",
    )
    return parser.parse_args()


def extract_coco_pose(video_path: Path, max_frames: int = 0) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    keypoints: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    detected_frames = 0

    pose_api = mp.solutions.pose
    with pose_api.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as pose:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames and len(keypoints) >= max_frames:
                break

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_points = np.zeros((17, 2), dtype=np.float32)
            frame_scores = np.zeros(17, dtype=np.float32)
            if result.pose_landmarks:
                detected_frames += 1
                landmarks = result.pose_landmarks.landmark
                for coco_index, mp_index in enumerate(MEDIAPIPE_TO_COCO):
                    landmark = landmarks[mp_index]
                    frame_points[coco_index, 0] = landmark.x * width
                    frame_points[coco_index, 1] = landmark.y * height
                    frame_scores[coco_index] = max(
                        0.0, min(1.0, float(landmark.visibility))
                    )
            keypoints.append(frame_points)
            scores.append(frame_scores)

    capture.release()
    if not keypoints:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    if detected_frames == 0:
        raise RuntimeError("MediaPipe did not find a usable person pose in any frame.")

    return {
        "keypoint": np.asarray(keypoints, dtype=np.float32)[None, ...],
        "keypoint_score": np.asarray(scores, dtype=np.float32)[None, ...],
        "img_shape": (height, width),
        "original_shape": (height, width),
        "total_frames": len(keypoints),
        "detected_frames": detected_frames,
        "fps": fps,
    }


def load_pose_sequence(path: Path, video_path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    keypoint = np.asarray(raw["keypoint"], dtype=np.float32)
    keypoint_score = np.asarray(raw["keypoint_score"], dtype=np.float32)
    if keypoint.ndim != 4 or keypoint.shape[-2:] != (17, 2):
        raise ValueError("pose JSON keypoint must have shape [people, frames, 17, 2]")
    if keypoint_score.shape != keypoint.shape[:-1]:
        raise ValueError("pose JSON keypoint_score shape does not match keypoint")
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if capture.isOpened() else 0.0
    capture.release()
    return {
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
        "img_shape": tuple(raw["img_shape"]),
        "original_shape": tuple(raw.get("original_shape", raw["img_shape"])),
        "total_frames": int(raw.get("total_frames", keypoint.shape[1])),
        "detected_frames": int(np.count_nonzero(keypoint_score.max(axis=(0, 2)) > 0)),
        "fps": fps,
        "track_ids": [str(value) for value in raw.get("track_ids", [])],
    }


def cpu_config(config_path: Path) -> mmcv.Config:
    config = mmcv.Config.fromfile(str(config_path))
    pipeline = [
        operation
        for operation in config.data.test.pipeline
        if operation["type"] != "DecompressPose"
    ]
    # The official test uses 10 clips plus horizontal mirroring.  A single
    # unmirrored clip is substantially more practical for a CPU smoke test.
    for operation in pipeline:
        if operation["type"] == "UniformSampleFrames":
            operation["num_clips"] = 1
        if operation["type"] == "GeneratePoseTarget":
            operation["double"] = False
    config.data.test.pipeline = pipeline
    return config


def run(args: argparse.Namespace) -> dict:
    required_paths = [args.video, args.config, args.checkpoint, args.labels]
    if args.pose_json:
        required_paths.append(args.pose_json)
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(POSEC3D_ROOT))
    from pyskl.apis import inference_recognizer, init_recognizer

    started = time.perf_counter()
    pose_data = (
        load_pose_sequence(args.pose_json, args.video)
        if args.pose_json
        else extract_coco_pose(args.video, args.max_frames)
    )
    pose_seconds = time.perf_counter() - started

    annotation = {
        "frame_dir": "",
        "label": -1,
        "start_index": 0,
        "modality": "Pose",
        "total_frames": pose_data["total_frames"],
        "img_shape": pose_data["img_shape"],
        "original_shape": pose_data["original_shape"],
        "keypoint": pose_data["keypoint"],
        "keypoint_score": pose_data["keypoint_score"],
    }

    model_started = time.perf_counter()
    model = init_recognizer(
        cpu_config(args.config), str(args.checkpoint), device="cpu"
    )
    load_seconds = time.perf_counter() - model_started

    inference_started = time.perf_counter()
    predictions = inference_recognizer(model, annotation)
    inference_seconds = time.perf_counter() - inference_started

    labels = args.labels.read_text(encoding="utf-8").splitlines()
    top_predictions = [
        {
            "class_index": int(class_index),
            "label": labels[int(class_index)],
            "score": float(score),
        }
        for class_index, score in predictions[: max(1, args.top_k)]
    ]
    top_prediction = top_predictions[0]
    activity_accepted = top_prediction["score"] >= args.min_confidence
    report = {
        "status": "success",
        "device": "cpu",
        "video": str(args.video.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_vocabulary": "NTU-RGB+D 120",
        "frames_read": pose_data["total_frames"],
        "frames_with_pose": pose_data["detected_frames"],
        "people_in_pose_sequence": int(pose_data["keypoint"].shape[0]),
        "pose_source": str(args.pose_json.resolve()) if args.pose_json else "MediaPipe single-person extraction",
        "track_ids": pose_data.get("track_ids", []),
        "fps": pose_data["fps"],
        "top_predictions": top_predictions,
        "activity": (
            top_prediction["label"] if activity_accepted else "Not determined"
        ),
        "activity_phrase": (
            ACTIVITY_PHRASES.get(top_prediction["label"], top_prediction["label"])
            if activity_accepted
            else "performing an undetermined activity"
        ),
        "activity_confidence": top_prediction["score"],
        "activity_min_confidence": args.min_confidence,
        "activity_accepted": activity_accepted,
        "timing_seconds": {
            "pose_extraction": round(pose_seconds, 3),
            "model_loading": round(load_seconds, 3),
            "action_inference": round(inference_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "limitations": [
            "CPU proof of concept using tracked COCO-17 skeletons.",
            "The checkpoint predicts NTU-120 clip classes, not arbitrary activities.",
            "Interaction labels require a multi-person scene pose sequence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as error:  # CLI boundary: preserve a useful failure report.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
