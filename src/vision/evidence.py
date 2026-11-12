from __future__ import annotations

from pathlib import Path


def annotate_snapshot(source: str | Path, destination: str | Path, people) -> Path:
    from PIL import Image, ImageDraw

    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for person in people:
        if person.bbox_xyxy is None:
            continue
        draw.rectangle(person.bbox_xyxy, outline="red", width=3)
        draw.text((person.bbox_xyxy[0] + 3, person.bbox_xyxy[1] + 3), f"Track {person.track_id}", fill="yellow")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
