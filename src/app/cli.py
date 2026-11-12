from __future__ import annotations

import argparse
from pathlib import Path

from ..vision.merge import integrate_outputs
from ..alerts.service import render_alert
from ..alerts.cooldown import CooldownStore
from ..core.io import read_json
from ..alerts.mail import send_message
from ..core.models import SurveillanceEvent
from ..core.config import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render deterministic SapiensID + OpenPAR alerts")
    parser.add_argument("--event", required=True, help="Event metadata JSON")
    parser.add_argument("--sapiens-results", required=True, help="SapiensID database_match_results.json")
    parser.add_argument("--openpar-results", required=True, help="OpenPAR predictions JSON")
    parser.add_argument("--track-manifest", required=True, help="Explicit track-to-model-output association JSON")
    parser.add_argument("--output-dir", default=str(PATHS.output / "manual_alerts"))
    parser.add_argument("--attribute-threshold", type=float, default=0.45)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--send", action="store_true", help="Actually send email; default is safe dry-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = read_json(args.event)
    people = integrate_outputs(
        args.sapiens_results, args.openpar_results, args.track_manifest, args.attribute_threshold
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_source = metadata.pop("snapshot_source", None)
    metadata["snapshot"] = "annotated_snapshot.jpg" if snapshot_source else "Not available"
    metadata.setdefault("video_clip", "Not available")
    event = SurveillanceEvent(people=people, **metadata)
    rendered, artifacts = render_alert(event, output_dir, snapshot_source)

    if not args.send:
        print(f"DRY RUN: rendered email saved to {output_dir.resolve()}")
        return 0

    cooldown = CooldownStore(output_dir / "cooldown.json", args.cooldown_seconds)
    if not cooldown.may_send(event.event_id):
        print(f"SUPPRESSED: event {event.event_id} is inside the cooldown window")
        return 2
    from email import policy
    from email.parser import BytesParser
    message = BytesParser(policy=policy.default).parsebytes(artifacts.email_message.read_bytes())
    send_message(message)
    cooldown.record(event.event_id)
    print("Email sent to ayushmohta2402@gmail.com")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
