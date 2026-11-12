from __future__ import annotations

from pathlib import Path

from ..core.io import read_json
from ..core.models import IdentityStatus, PersonRecord


UPPER = {
    "A pedestrian in casual upper wear": "casual upper clothing",
    "A pedestrian in formal upper wear": "formal upper clothing",
    "A pedestrian in a jacket": "a jacket",
    "A pedestrian in a short-sleeved top": "a short-sleeved top",
    "A pedestrian in a t-shirt": "a t-shirt",
    "A pedestrian in other upper wear": "an upper garment",
}
LOWER = {
    "A pedestrian in casual lower wear": "casual lower clothing",
    "A pedestrian in formal lower wear": "formal lower clothing",
    "A pedestrian in jeans": "jeans",
    "A pedestrian in shorts": "shorts",
    "A pedestrian in a short skirt": "a short skirt",
    "A pedestrian in trousers": "trousers",
}
ITEMS = {
    "A pedestrian with a backpack": "a backpack",
    "A pedestrian with other types of attachments": "other supported attachments",
    "A pedestrian with a messenger bag": "a messenger bag",
    "A pedestrian with plastic bags": "plastic bags",
    "A pedestrian with no attachments": "no carried items",
}


def _best(attributes: list[dict], supported: dict[str, str], minimum_score: float) -> str | None:
    candidates = [
        (float(item.get("score", 0.0)), supported[item.get("attribute")])
        for item in attributes if item.get("attribute") in supported
        and float(item.get("score", 0.0)) >= minimum_score
    ]
    return max(candidates, default=(0.0, None))[1]


def _with_color(value: str, color_result: dict | None) -> str:
    color = (color_result or {}).get("value")
    if not color:
        return value
    if value.startswith(("a ", "an ")):
        _, noun = value.split(" ", 1)
        article = "an" if color[0].lower() in "aeiou" else "a"
        return f"{article} {color} {noun}"
    return f"{color} {value}"


def integrate_outputs(
    sapiens_results_path: str | Path,
    openpar_results_path: str | Path,
    track_manifest_path: str | Path,
    attribute_threshold: float = 0.45,
) -> tuple[PersonRecord, ...]:
    """Join model results only through explicit same-person manifest entries.

    Manifest entries contain `track_id`, `sapiens_query_label`, and
    `openpar_image_key`. A missing Sapiens result is Unresolved; a completed
    no-match result is Unknown. OpenPAR currently supports clothing/carried
    items, not activity or position.
    """
    sapiens_raw = read_json(sapiens_results_path)
    openpar = read_json(openpar_results_path)
    manifest = read_json(track_manifest_path)
    sapiens = {item["query_label"]: item for item in sapiens_raw.get("matches", [])}

    people: list[PersonRecord] = []
    seen: set[str] = set()
    for link in manifest.get("tracks", []):
        track_id = str(link["track_id"])
        if track_id in seen:
            raise ValueError(f"Duplicate track ID in manifest: {track_id}")
        seen.add(track_id)
        match = sapiens.get(link.get("sapiens_query_label"))
        face_evaluated = bool(link.get("face_evaluated", match is not None))
        if not face_evaluated or match is None:
            identity, name = IdentityStatus.UNRESOLVED, "Not identified"
        elif match.get("recognition_status", "").upper() == "KNOWN" and match.get("recognized_identity"):
            identity, name = IdentityStatus.KNOWN, str(match["recognized_identity"])
        else:
            identity, name = IdentityStatus.UNKNOWN, "Not identified"

        openpar_key = link.get("openpar_image_key")
        attrs = openpar.get(openpar_key, []) if openpar_key else []
        visibility = link.get("visibility", {})
        upper = (
            _best(attrs, UPPER, attribute_threshold)
            if visibility.get("upper_body", True) else None
        ) or "upper clothing not determined"
        lower = (
            _best(attrs, LOWER, attribute_threshold)
            if visibility.get("lower_body", True) else None
        ) or "lower clothing not determined"
        items = (
            _best(attrs, ITEMS, attribute_threshold)
            if visibility.get("carried_items", True) else None
        ) or "items not determined"
        colors = link.get("colors", {})
        if upper != "upper clothing not determined":
            upper = _with_color(upper, colors.get("upper"))
        if lower != "lower clothing not determined":
            lower = _with_color(lower, colors.get("lower"))
        if items not in ("items not determined", "no carried items"):
            items = _with_color(items, colors.get("carried_item"))
        bbox = link.get("bbox_xyxy")
        people.append(PersonRecord(
            track_id=track_id,
            identity_status=identity,
            name=name,
            upper_clothing=upper,
            lower_clothing=lower,
            carried_items=items,
            activity="performing an undetermined activity",
            position=str(link.get("position") or "an undetermined position"),
            bbox_xyxy=tuple(map(float, bbox)) if bbox else None,
        ))
    return tuple(people)
