"""End-to-end tracked-video surveillance workflow.

The image workflow remains unchanged.  This command adds temporal tracking,
quality-based crop selection, per-track model aggregation, PoseC3D activity,
and deterministic dry-run alert generation for prerecorded video.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ..alerts.service import render_alert
from ..core.config import PATHS, TIMEZONE_NAME
from ..core.io import read_json, write_json
from ..core.models import IdentityStatus, PersonRecord, SurveillanceEvent
from ..vision.merge import ITEMS, LOWER, UPPER
from ..vision.video import scan_video


TRACK_PATTERN = re.compile(r"t(?P<track>\d+)_r\d+", re.IGNORECASE)
INTERACTION_ACTIONS = {
    "hugging other person",
    "support somebody with hand",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "touch other person's pocket",
    "giving something to other person",
    "high-five",
    "grab other person’s stuff",
    "knock over other person (hit with body)",
    "hit other person with something",
    "wield knife towards other person",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracked SapiensID + OpenPAR + PoseC3D video workflow")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--location", default="Not determined")
    parser.add_argument("--source-id", default="Not determined")
    parser.add_argument("--event-type", default="People observed")
    parser.add_argument("--detection-fps", type=float, default=5.0)
    parser.add_argument("--detection-confidence", type=float, default=0.30)
    parser.add_argument("--match-threshold", type=float, default=0.33)
    parser.add_argument("--attribute-threshold", type=float, default=0.45)
    parser.add_argument("--activity-threshold", type=float, default=0.35)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--best-crops", type=int, default=3)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--skip-attributes", action="store_true")
    parser.add_argument("--skip-action", action="store_true")
    return parser.parse_args()


def _run(command: list[str], cwd: Path = PATHS.workspace) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _track_from_path(value: str | Path) -> str | None:
    match = TRACK_PATTERN.search(Path(value).stem)
    return str(int(match.group("track"))) if match else None


def _identity_evidence(directory: Path) -> dict[str, dict]:
    evidence: dict[str, list[dict]] = defaultdict(list)
    if not directory.is_dir():
        return {}
    for summary_path in directory.glob("batch/*/summary.json"):
        summary = read_json(summary_path)
        track_id = _track_from_path(summary.get("input_image", ""))
        if not track_id:
            continue
        matches = summary.get("step_7_database_match", {}).get("matches", [])
        if matches:
            best = matches[0].get("best_match") or {}
            evidence[track_id].append({
                "label": best.get("label"),
                "score": float(best.get("score") or 0.0),
                "evaluated": True,
            })
        else:
            evidence[track_id].append({"label": None, "score": 0.0, "evaluated": False})

    aggregated = {}
    for track_id, observations in evidence.items():
        by_label: dict[str, list[float]] = defaultdict(list)
        evaluated = 0
        for item in observations:
            evaluated += int(item["evaluated"])
            if item["label"]:
                by_label[str(item["label"])].append(item["score"])
        candidates = []
        for label, scores in by_label.items():
            strongest = sorted(scores, reverse=True)[:3]
            candidates.append((sum(strongest) / len(strongest), max(scores), label))
        mean_score, max_score, label = max(candidates, default=(0.0, 0.0, None))
        aggregated[track_id] = {
            "label": label,
            "aggregate_score": round(mean_score, 6),
            "maximum_score": round(max_score, 6),
            "evaluated_observations": evaluated,
            "total_observations": len(observations),
        }
    return aggregated


def _attribute_evidence(path: Path, threshold: float) -> dict[str, dict]:
    if not path.is_file():
        return {}
    raw = read_json(path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for image_path, attributes in raw.items():
        track_id = _track_from_path(image_path)
        if track_id:
            grouped[track_id].extend(attributes)

    def best(items: list[dict], vocabulary: dict[str, str], fallback: str) -> str:
        scores: dict[str, list[float]] = defaultdict(list)
        for item in items:
            attribute = item.get("attribute")
            score = float(item.get("score", 0.0))
            if attribute in vocabulary and score >= threshold:
                scores[vocabulary[attribute]].append(score)
        ranked = [
            (sum(values) / len(values), max(values), label)
            for label, values in scores.items()
        ]
        return max(ranked, default=(0.0, 0.0, fallback))[2]

    return {
        track_id: {
            "upper": best(items, UPPER, "upper clothing not determined"),
            "lower": best(items, LOWER, "lower clothing not determined"),
            "items": best(items, ITEMS, "items not determined"),
        }
        for track_id, items in grouped.items()
    }


def _run_actions(track_report: dict, output_dir: Path, threshold: float) -> dict[str, dict]:
    python = PATHS.workspace / ".venv_posec3d" / "Scripts" / "python.exe"
    script = PATHS.source / "vision" / "action.py"
    if not python.is_file():
        return {}
    action_dir = output_dir / "actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    scene_pose = track_report.get("scene_pose_path")
    if scene_pose and Path(scene_pose).is_file():
        scene_path = action_dir / "scene.json"
        _run([
            str(python), str(script), "--video", track_report["video"],
            "--pose-json", scene_pose, "--output", str(scene_path),
            "--min-confidence", str(threshold),
        ])
        results["_scene"] = read_json(scene_path)
    for track in track_report["tracks"]:
        clip = track.get("clip_path")
        if not clip or not Path(clip).is_file():
            continue
        result_path = action_dir / f"track_{int(track['track_id']):04d}.json"
        _run([
            str(python), str(script), "--video", clip,
            "--output", str(result_path),
            "--min-confidence", str(threshold),
        ])
        results[str(track["track_id"])] = read_json(result_path)
    return results


def _people(track_report: dict, identities: dict, attributes: dict, actions: dict, match_threshold: float) -> tuple[PersonRecord, ...]:
    people = []
    scene_action = actions.get("_scene", {})
    scene_label = scene_action.get("activity")
    scene_tracks = set(scene_action.get("track_ids", []))
    use_scene_interaction = bool(
        scene_action.get("activity_accepted") and scene_label in INTERACTION_ACTIONS
    )
    for track in track_report["tracks"]:
        track_id = str(track["track_id"])
        identity = identities.get(track_id)
        if not identity or not identity.get("evaluated_observations"):
            status, name = IdentityStatus.UNRESOLVED, "Not identified"
        elif identity.get("aggregate_score", 0.0) > match_threshold and identity.get("label"):
            status, name = IdentityStatus.KNOWN, str(identity["label"])
        else:
            status, name = IdentityStatus.UNKNOWN, "Not identified"
        attrs = attributes.get(track_id, {})
        action = actions.get(track_id, {})
        if use_scene_interaction and track_id in scene_tracks:
            action = scene_action
        people.append(PersonRecord(
            track_id=track_id,
            identity_status=status,
            name=name,
            upper_clothing=attrs.get("upper", "upper clothing not determined"),
            lower_clothing=attrs.get("lower", "lower clothing not determined"),
            carried_items=attrs.get("items", "items not determined"),
            activity=action.get("activity_phrase", "performing an undetermined activity"),
            bbox_xyxy=tuple(track.get("snapshot_bbox_xyxy") or track["best_bbox_xyxy"]),
        ))
    return tuple(people)


def main() -> int:
    args = arguments()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    timestamp = datetime.now(ZoneInfo(TIMEZONE_NAME))
    event_id = f"VID-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    output_dir = (args.output_dir or PATHS.output / "video_runs" / event_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    yolo = PATHS.sapiensid / "tasks" / "WebBody" / "c_detect" / "yolov8n-pose.pt"

    tracks = scan_video(
        video, output_dir / "tracking", yolo,
        detection_fps=args.detection_fps,
        confidence=args.detection_confidence,
        max_detections=args.max_detections,
        keep_best=args.best_crops,
    )
    if args.scan_only or not tracks["tracks"]:
        print(json.dumps(tracks, indent=2))
        return 0

    crop_dir = output_dir / "tracking" / "crops"
    identity_dir = output_dir / "identity"
    if not args.skip_identity:
        _run([
            sys.executable, str(PATHS.sapiens_runner),
            "--image", str(crop_dir), "--output-dir", str(identity_dir),
            "--database-dir", str(PATHS.database), "--mode", "direct",
            "--match-threshold", str(args.match_threshold),
        ])
    identities = _identity_evidence(identity_dir)

    openpar_path = output_dir / "openpar.json"
    if not args.skip_attributes:
        _run([
            sys.executable, str(PATHS.openpar_runner), "PETA",
            "--dir", str(PATHS.openpar_checkpoint), "--image-root", str(crop_dir),
            "--output", str(openpar_path), "--threshold", str(args.attribute_threshold),
            "--use_div", "--use_vismask", "--use_GL", "--use_textprompt", "--use_mm_former",
        ], PATHS.openpar_promptpar)
    attributes = _attribute_evidence(openpar_path, args.attribute_threshold)
    actions = {} if args.skip_action else _run_actions(tracks, output_dir, args.activity_threshold)

    people = _people(tracks, identities, attributes, actions, args.match_threshold)
    duration = (tracks["source_frames"] / tracks["source_fps"]) if tracks["source_fps"] else 0.0
    event = SurveillanceEvent(
        event_id=event_id,
        event_type=args.event_type,
        date_time_with_timezone=timestamp.strftime("%Y-%m-%d %H:%M:%S %z"),
        location=args.location,
        source_id=args.source_id,
        people=people,
        specific_observed_reason=f"{len(people)} unique tracked people observed in video",
        first_detected_time=timestamp.strftime("%Y-%m-%d %H:%M:%S %z"),
        observed_duration=f"{duration:.1f} seconds",
        status="Resolved",
        snapshot="annotated_snapshot.jpg" if tracks.get("snapshot") else "Not available",
        video_clip=video.name,
    )
    write_json(output_dir / "identity_aggregate.json", identities)
    write_json(output_dir / "attribute_aggregate.json", attributes)
    write_json(output_dir / "action_aggregate.json", actions)
    write_json(output_dir / "event.json", event.as_dict())
    render_alert(event, output_dir / "alert", tracks.get("snapshot"))
    print(f"Video workflow complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
