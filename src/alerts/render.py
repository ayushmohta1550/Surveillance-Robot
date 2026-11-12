from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from string import Template

from ..core.models import SurveillanceEvent


@dataclass(frozen=True)
class RenderedEmail:
    subject: str
    text: str
    html: str


class EmailRenderer:
    def __init__(self, template_dir: str | Path | None = None):
        root = Path(template_dir) if template_dir else Path(__file__).parents[1] / "templates"
        self.email_template = Template((root / "email.txt").read_text(encoding="utf-8"))
        self.person_template = Template((root / "person.txt").read_text(encoding="utf-8"))
        self.html_template = Template((root / "email.html").read_text(encoding="utf-8"))

    def render(self, event: SurveillanceEvent) -> RenderedEmail:
        blocks = []
        for number, person in enumerate(event.sorted_people, start=1):
            blocks.append(self.person_template.substitute(
                person_number=number,
                track_id=person.track_id,
                identity_status=person.identity_status.value,
                name=person.name,
                upper_clothing=person.upper_clothing,
                lower_clothing=person.lower_clothing,
                carried_items=person.carried_items,
                activity=person.activity,
                position=person.position,
                description=person.description(number),
            ))
        counts = event.counts
        values = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "date_time_with_timezone": event.date_time_with_timezone,
            "location": event.location,
            "source_id": event.source_id,
            "total_count": counts["total"],
            "known_count": counts["known"],
            "unknown_count": counts["unknown"],
            "unresolved_count": counts["unresolved"],
            "person_blocks": "\n\n".join(blocks) if blocks else "None",
            "specific_observed_reason": event.specific_observed_reason,
            "first_detected_time": event.first_detected_time,
            "observed_duration": event.observed_duration,
            "status": event.status,
            "snapshot": event.snapshot,
            "video_clip": event.video_clip,
        }
        text = self.email_template.substitute(values)
        subject = f"[Surveillance Alert] {event.event_type} | {event.location} | {event.event_id}"
        html_body = self.html_template.substitute(body=html.escape(text))
        return RenderedEmail(subject=subject, text=text, html=html_body)
