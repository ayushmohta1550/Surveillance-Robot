"""Single service for producing deterministic alert artifacts."""

from dataclasses import dataclass
from pathlib import Path

from ..vision.evidence import annotate_snapshot
from .mail import build_message
from ..core.models import SurveillanceEvent
from .render import EmailRenderer, RenderedEmail


@dataclass(frozen=True)
class AlertArtifacts:
    directory: Path
    subject: Path
    plain_text: Path
    html: Path
    email_message: Path
    annotated_snapshot: Path | None


def render_alert(
    event: SurveillanceEvent,
    output_dir: str | Path,
    snapshot_source: str | Path | None = None,
) -> tuple[RenderedEmail, AlertArtifacts]:
    """Render and save the one canonical text/HTML/EML representation.

    When a source snapshot is supplied, the event must already refer to the
    filename `annotated_snapshot.jpg`; the generated file is attached to EML.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    annotated = None
    if snapshot_source and Path(snapshot_source).is_file():
        annotated = annotate_snapshot(
            snapshot_source, destination / "annotated_snapshot.jpg", event.sorted_people
        )

    rendered = EmailRenderer().render(event)
    message = build_message(rendered, annotated)
    subject_path = destination / "subject.txt"
    text_path = destination / "email.txt"
    html_path = destination / "email.html"
    eml_path = destination / "email.eml"
    subject_path.write_text(rendered.subject, encoding="utf-8")
    text_path.write_text(rendered.text, encoding="utf-8")
    html_path.write_text(rendered.html, encoding="utf-8")
    eml_path.write_bytes(message.as_bytes())
    return rendered, AlertArtifacts(
        directory=destination,
        subject=subject_path,
        plain_text=text_path,
        html=html_path,
        email_message=eml_path,
        annotated_snapshot=annotated,
    )
