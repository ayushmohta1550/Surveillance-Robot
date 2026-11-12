"""Conservative garment-colour extraction for OpenPAR person crops.

This module does not modify OpenPAR. It combines the aligned foreground mask
and pose regions already produced by the SapiensID wrapper, then maps visible
pixels to a small deterministic colour vocabulary.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ColorResult:
    value: str | None
    confidence: float
    valid_pixels: int

    def as_dict(self) -> dict:
        return asdict(self)


def _square_resized(image_path: str | Path, size: int = 384) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    side = max(image.size)
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    right = side - image.width - left
    bottom = side - image.height - top
    return ImageOps.expand(image, (left, top, right, bottom), fill="white").resize((size, size))


def _pixel_name(rgb: np.ndarray) -> str:
    red, green, blue = (float(channel) / 255.0 for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    hue *= 360.0
    if value < 0.14:
        return "black"
    if saturation < 0.12:
        if value > 0.93:
            return "white"
        if value > 0.78:
            return "light grey"
        if value < 0.30:
            return "charcoal"
        return "grey"
    if (hue < 15 or hue >= 345) and value < 0.42:
        return "maroon"
    if 15 <= hue < 48 and value > 0.82 and saturation < 0.35:
        return "beige"
    if 15 <= hue < 48 and value > 0.62 and saturation < 0.60:
        return "tan"
    if 8 <= hue < 45 and (value < 0.72 or saturation < 0.50):
        return "brown"
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        if saturation < 0.30 and value > 0.88:
            return "cream"
        if value < 0.55:
            return "olive"
        if value < 0.78:
            return "gold"
        return "yellow"
    if hue < 95:
        if value < 0.55:
            return "olive"
        return "lime"
    if hue < 155:
        if value < 0.45 and saturation < 0.65:
            return "olive"
        return "green"
    if hue < 180:
        return "teal"
    if hue < 205:
        return "cyan"
    if hue < 255:
        if value < 0.45:
            return "navy"
        if value > 0.82 and saturation < 0.65:
            return "sky blue"
        return "blue"
    if hue < 290:
        return "purple"
    if hue < 345:
        return "pink"
    return "red"


def _dominant_color(
    image: np.ndarray,
    foreground: np.ndarray,
    box: tuple[int, int, int, int] | None,
    minimum_pixels: int = 150,
    minimum_share: float = 0.35,
) -> ColorResult:
    if box is None:
        return ColorResult(None, 0.0, 0)
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return ColorResult(None, 0.0, 0)
    pixels = image[y1:y2, x1:x2][foreground[y1:y2, x1:x2]]
    if len(pixels) < minimum_pixels:
        return ColorResult(None, 0.0, int(len(pixels)))
    # Quantization makes the result less sensitive to JPEG noise and lighting.
    pixels = (pixels // 24) * 24 + 12
    names = [_pixel_name(pixel) for pixel in pixels]
    values, counts = np.unique(names, return_counts=True)
    index = int(np.argmax(counts))
    share = float(counts[index]) / len(names)
    return ColorResult(str(values[index]) if share >= minimum_share else None, round(share, 6), len(names))


def _point(records: dict[str, dict], name: str) -> tuple[float, float] | None:
    record = records.get(name)
    if not record or not record.get("valid", False):
        return None
    return float(record["x_normalized"]), float(record["y_normalized"])


def analyze_person_colors(detection: dict) -> dict:
    """Return colour evidence for visible upper/lower/item regions."""
    crop = detection.get("crop_path")
    mask_path = detection.get("foreground_mask_path")
    if not crop or not mask_path or not Path(crop).is_file() or not Path(mask_path).is_file():
        empty = ColorResult(None, 0.0, 0).as_dict()
        return {"upper": empty, "lower": empty, "carried_item": empty}

    image = np.asarray(_square_resized(crop), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L").resize((384, 384))) > 127
    records = {record["name"]: record for record in detection.get("keypoints", [])}
    visibility = detection.get("visibility", {})
    left_shoulder, right_shoulder = _point(records, "left_shoulder"), _point(records, "right_shoulder")
    left_hip, right_hip = _point(records, "left_hip"), _point(records, "right_hip")
    left_ankle, right_ankle = _point(records, "left_ankle"), _point(records, "right_ankle")

    upper_box = None
    if visibility.get("upper_body", False) and left_shoulder and right_shoulder:
        sx = [left_shoulder[0], right_shoulder[0]]
        shoulder_y = min(left_shoulder[1], right_shoulder[1])
        hip_y = max(left_hip[1], right_hip[1]) if left_hip and right_hip else 0.95
        width = max(sx) - min(sx)
        upper_box = (
            int((min(sx) - 0.25 * width) * 384), int((shoulder_y - 0.03) * 384),
            int((max(sx) + 0.25 * width) * 384), int(min(0.96, hip_y) * 384),
        )

    lower_box = None
    if visibility.get("lower_body", False) and left_hip and right_hip:
        hx = [left_hip[0], right_hip[0]]
        hip_y = min(left_hip[1], right_hip[1])
        ankle_y = max(left_ankle[1], right_ankle[1]) if left_ankle and right_ankle else 0.98
        width = max(hx) - min(hx)
        lower_box = (
            int((min(hx) - 0.35 * width) * 384), int(hip_y * 384),
            int((max(hx) + 0.35 * width) * 384), int(min(0.99, ankle_y) * 384),
        )

    # A reliable bag region requires human parsing. Do not reuse a general body
    # colour because it could incorrectly assign a shirt colour to a bag.
    no_item_color = ColorResult(None, 0.0, 0)
    return {
        "upper": _dominant_color(image, mask, upper_box).as_dict(),
        "lower": _dominant_color(image, mask, lower_box).as_dict(),
        "carried_item": no_item_color.as_dict(),
    }
