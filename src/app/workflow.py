from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from ..vision.merge import integrate_outputs
from ..alerts.service import render_alert
from ..vision.colors import analyze_person_colors
from ..core.io import read_json, write_json
from ..core.models import SurveillanceEvent
from ..core.config import PATHS, TIMEZONE_NAME


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic person detection, identity matching, attributes, and alert rendering"
    )
    parser.add_argument("image", help="Input surveillance image")
    parser.add_argument("--location", default="Not determined")
    parser.add_argument("--source-id", default="Not determined")
    parser.add_argument("--event-type", default="Person detected")
    parser.add_argument("--position", default="an undetermined position")
    parser.add_argument("--match-threshold", type=float, default=0.33)
    parser.add_argument("--attribute-threshold", type=float, default=0.45)
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.30,
        help="YOLO person-detection confidence; independent of identity matching",
    )
    parser.add_argument("--max-detections", type=int, default=20)
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def make_event_id(image: Path, timestamp: datetime) -> str:
    digest = hashlib.sha256(str(image.resolve()).encode("utf-8")).hexdigest()[:8].upper()
    return f"EV-{timestamp.strftime('%Y%m%d-%H%M%S')}-{digest}"


def build_manifest(summary: dict, openpar: dict, image: Path) -> dict:
    width, height = Image.open(image).size
    tracks = []
    for number, detection in enumerate(summary.get("outputs", {}).get("webbody", []), start=1):
        crop = str(Path(detection["crop_path"]).resolve())
        normalized = detection.get("bbox_xyxyn")
        bbox = None
        if normalized and len(normalized) == 4:
            bbox = [
                normalized[0] * width, normalized[1] * height,
                normalized[2] * width, normalized[3] * height,
            ]
        # Path spelling is taken from OpenPAR's emitted key, not guessed.
        openpar_key = next((key for key in openpar if Path(key).resolve() == Path(crop)), None)
        tracks.append({
            "track_id": str(number),
            "sapiens_query_label": detection["label"],
            "openpar_image_key": openpar_key,
            "face_evaluated": True,
            "bbox_xyxy": bbox,
            "visibility": detection.get("visibility", {}),
            "colors": analyze_person_colors(detection),
        })
    return {"tracks": tracks}


def main() -> int:
    args = arguments()
    image = Path(args.image).resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Input image not found: {image}")
    if not PATHS.database.is_dir():
        raise FileNotFoundError(f"Identity database not found: {PATHS.database}")
    for required in (PATHS.sapiens_runner, PATHS.openpar_runner, PATHS.openpar_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(f"Required model resource not found: {required}")

    now = datetime.now(ZoneInfo(TIMEZONE_NAME))
    event_id = make_event_id(image, now)
    run_dir = PATHS.output / "surveillance_alerts" / event_id
    sapiens_dir = run_dir / "sapiensid"
    run_dir.mkdir(parents=True, exist_ok=False)

    run([
        sys.executable, str(PATHS.sapiens_runner),
        "--image", str(image),
        "--output-dir", str(sapiens_dir),
        "--database-dir", str(PATHS.database),
        "--mode", "webbody",
        "--max-detections", str(args.max_detections),
        "--confidence-threshold", str(args.detection_confidence),
        "--match-threshold", str(args.match_threshold),
    ], PATHS.workspace)

    summary_path = sapiens_dir / "summary.json"
    match_path = sapiens_dir / "database_match_results.json"
    summary = read_json(summary_path)
    crops = sapiens_dir / "webbody" / "person_crops"
    detections = summary.get("outputs", {}).get("webbody", [])

    openpar_path = run_dir / "openpar_predictions.json"
    if not detections:
        result = {
            "event_id": event_id,
            "input_image": str(image),
            "people_detected": 0,
            "action": "No surveillance email created",
        }
        write_json(run_dir / "no_people_detected.json", result)
        print(f"No people detected. No alert created. Result: {run_dir}")
        return 0

    run([
        sys.executable, str(PATHS.openpar_runner), "PETA",
        "--dir", str(PATHS.openpar_checkpoint),
        "--image-root", str(crops),
        "--output", str(openpar_path),
        "--threshold", str(args.attribute_threshold),
        "--use_div", "--use_vismask", "--use_GL",
        "--use_textprompt", "--use_mm_former",
    ], PATHS.openpar_promptpar)

    openpar = read_json(openpar_path)
    manifest = build_manifest(summary, openpar, image)
    for track in manifest["tracks"]:
        track["position"] = args.position
    manifest_path = run_dir / "track_manifest.json"
    write_json(manifest_path, manifest)

    people = integrate_outputs(match_path, openpar_path, manifest_path, args.attribute_threshold)
    total = len(people)
    reason = f"{total} person{'s' if total != 1 else ''} detected in the surveillance image" if total else "No people detected in the surveillance image"
    event = SurveillanceEvent(
        event_id=event_id,
        event_type=args.event_type,
        date_time_with_timezone=now.strftime("%Y-%m-%d %H:%M:%S %z"),
        location=args.location,
        source_id=args.source_id,
        people=people,
        specific_observed_reason=reason,
        first_detected_time=now.strftime("%Y-%m-%d %H:%M:%S %z"),
        observed_duration="Single image",
        status="Resolved",
        snapshot="annotated_snapshot.jpg",
        video_clip="Not available",
    )
    rendered, _ = render_alert(event, run_dir, image)
    print(f"Workflow complete. Dry-run alert: {run_dir}")
    print(rendered.subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
