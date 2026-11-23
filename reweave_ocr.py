"""Care-label OpenCV preprocessing, OCR execution, and structured parsing."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from reweave_core import extract_composition, prepare_image

_CARE_PATTERNS: list[tuple[str, str]] = [
    ("Do not wash", r"\bdo\s+not\s+wash\b"),
    ("Hand wash", r"\bhand\s+wash(?:\s+only)?\b"),
    ("Machine wash cold", r"\bmachine\s+wash\s+cold\b"),
    ("Machine wash warm", r"\bmachine\s+wash\s+warm\b"),
    ("Machine wash", r"\bmachine\s+wash\b"),
    ("Do not bleach", r"\bdo\s+not\s+bleach\b"),
    ("Non-chlorine bleach only", r"\bnon[- ]?chlorine\s+bleach\b"),
    ("Do not tumble dry", r"\bdo\s+not\s+tumble\s+dry\b"),
    ("Tumble dry low", r"\btumble\s+dry\s+low\b"),
    ("Line dry", r"\bline\s+dry\b"),
    ("Dry flat", r"\bdry\s+flat\b"),
    ("Do not iron", r"\bdo\s+not\s+iron\b"),
    ("Low / warm iron", r"\b(?:cool|low|warm)\s+iron\b|\biron\s+(?:low|warm|cool)\b"),
    ("Do not dry clean", r"\bdo\s+not\s+dry\s+clean\b"),
    ("Dry clean only", r"\bdry\s+clean\s+only\b"),
]


def preprocess_label_image(image: Image.Image) -> np.ndarray:
    """Enhance a care-label photo for OCR without inverting dark labels.

    Steps:
    1. Apply EXIF orientation & convert to RGB.
    2. Upscale small label crops (up to 2x) so tiny tag text is readable.
    3. Apply CLAHE on the LAB lightness channel to handle uneven lighting/wrinkles.
    4. Apply a light bilateral filter to reduce fabric grain while keeping text edges sharp.
    """
    rgb = np.asarray(prepare_image(image))
    height, width = rgb.shape[:2]
    if max(height, width) > 1000:
        down = 1000.0 / float(max(height, width))
        rgb = cv2.resize(rgb, (int(width * down), int(height * down)), interpolation=cv2.INTER_AREA)
        height, width = rgb.shape[:2]
    if min(height, width) < 600:
        scale = min(2.0, 600.0 / max(1, min(height, width)))
        rgb = cv2.resize(
            rgb,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced_rgb = cv2.cvtColor(
        cv2.merge((l_enhanced, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )
    return cv2.bilateralFilter(enhanced_rgb, d=5, sigmaColor=40, sigmaSpace=40)


def extract_care_instructions(label_text: str) -> list[str]:
    """Extract only explicitly written care instructions; never invent missing ones."""
    if not label_text:
        return []

    found: list[str] = []
    lowered = label_text.lower()

    for label, pattern in _CARE_PATTERNS:
        if label == "Machine wash" and any(x.startswith("Machine wash ") for x in found):
            continue
        if re.search(pattern, lowered):
            found.append(label)

    for temp_match in re.finditer(r"\bwash(?:\s+at)?\s*(30|40|60)\s*°?\s*c\b", lowered):
        entry = f"Wash at {temp_match.group(1)}°C"
        if entry not in found:
            found.append(entry)

    return found


def parse_care_label(
    raw_text: str,
    ocr_confidence: float | None = None,
    preprocessing_used: bool = True,
) -> dict[str, Any]:
    """Convert raw OCR text into structured, auditable care-label evidence."""
    cleaned_text = (raw_text or "").strip()
    composition = extract_composition(cleaned_text)
    care_instructions = extract_care_instructions(cleaned_text)

    if composition != "UNKNOWN":
        evidence = "Care-label OCR (explicit percentage + fibre detected)"
    elif cleaned_text:
        evidence = "Care-label text read, but no explicit fibre composition found"
    else:
        evidence = "None (no readable care-label text extracted)"

    return {
        "raw_text": cleaned_text,
        "composition": composition,
        "composition_evidence": evidence,
        "care_instructions": care_instructions,
        "ocr_confidence": ocr_confidence,
        "preprocessing_used": preprocessing_used,
    }


def _read_lines(reader: Any, image_array: np.ndarray) -> tuple[str, float | None]:
    results = reader.readtext(image_array, detail=1, paragraph=False)
    lines: list[str] = []
    confidences: list[float] = []
    for item in results:
        if len(item) >= 3:
            text_value = str(item[1]).strip()
            if text_value:
                lines.append(text_value)
                confidences.append(float(item[2]))
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else None
    return "\n".join(lines), avg_conf


def run_label_ocr(label_image: Image.Image, reader: Any) -> dict[str, Any]:
    """Run EasyOCR on preprocessed label image with safe fallback to raw RGB."""
    preprocessed = preprocess_label_image(label_image)
    text_pre, conf_pre = _read_lines(reader, preprocessed)
    parsed_pre = parse_care_label(text_pre, conf_pre, preprocessing_used=True)

    # Fast path: preprocessing already found explicit fibre composition.
    if parsed_pre["composition"] != "UNKNOWN":
        return parsed_pre

    # Safe fallback: retry on raw RGB so preprocessing never hurts readability.
    raw_rgb = np.asarray(prepare_image(label_image))
    text_raw, conf_raw = _read_lines(reader, raw_rgb)
    parsed_raw = parse_care_label(text_raw, conf_raw, preprocessing_used=False)

    if parsed_raw["composition"] != "UNKNOWN":
        return parsed_raw
    if not parsed_pre["raw_text"] and parsed_raw["raw_text"]:
        return parsed_raw
    return parsed_pre
