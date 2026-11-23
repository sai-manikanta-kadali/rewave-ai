"""Core logic and SQLite persistence for ReWeave AI."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

DB_PATH = Path(__file__).resolve().parent / "reweave.db"

FIBRE_TERMS = (
    "cotton",
    "polyester",
    "viscose",
    "rayon",
    "nylon",
    "wool",
    "silk",
    "acrylic",
    "linen",
    "elastane",
    "spandex",
    "polyamide",
    "modal",
    "lyocell",
)

_FIBRE_PATTERN = "|".join(FIBRE_TERMS)
_COMPOSITION_RE = re.compile(
    rf"(?i)(?:(?<!\d)(?P<pct1>\d{{1,3}})\s*%\s*(?P<fib1>{_FIBRE_PATTERN})\b|"
    rf"\b(?P<fib2>{_FIBRE_PATTERN})\s*[:\-]?\s*(?P<pct2>\d{{1,3}})\s*%)"
)

_MIGRATION_COLUMNS: dict[str, str] = {
    "yolo_detections": "TEXT NOT NULL DEFAULT '[]'",
    "yolo_max_confidence": "REAL",
    "ocr_confidence": "REAL",
    "composition_evidence": "TEXT NOT NULL DEFAULT ''",
    "care_instructions": "TEXT NOT NULL DEFAULT '[]'",
    "confirmation_status": "TEXT NOT NULL DEFAULT 'PENDING'",
    "confirmed_pathway": "TEXT NOT NULL DEFAULT ''",
    "confirmed_at": "TEXT",
}


def prepare_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and return a safe RGB image copy."""
    return ImageOps.exif_transpose(image).convert("RGB")


def image_diagnostics(image: Image.Image) -> dict[str, float | int]:
    """Return objective image diagnostics; these are not defect predictions."""
    rgb = np.asarray(prepare_image(image))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness_proxy = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mean_brightness": round(float(np.mean(gray)), 1),
        "sharpness_proxy": round(sharpness_proxy, 1),
    }


def extract_composition(label_text: str) -> str:
    """Extract only explicit percentage/fibre pairs; otherwise return UNKNOWN."""
    if not label_text:
        return "UNKNOWN"

    parsed: list[str] = []
    seen: set[str] = set()
    for match in _COMPOSITION_RE.finditer(label_text):
        pct_str = match.group("pct1") or match.group("pct2")
        fibre_str = match.group("fib1") or match.group("fib2")
        if not pct_str or not fibre_str:
            continue
        value = int(pct_str)
        if 1 <= value <= 100:
            entry = f"{value}% {fibre_str.lower()}"
            if entry not in seen:
                seen.add(entry)
                parsed.append(entry)

    return " + ".join(parsed) if parsed else "UNKNOWN"


def recommend_pathway(condition: str) -> tuple[str, str]:
    """Rule-based routing from worker-confirmed condition, not model inference."""
    routing = {
        "Ready to wear": (
            "REUSE",
            "Worker reports that the garment is wearable. Confirm hygiene, fit and resale/donation suitability.",
        ),
        "Minor repair needed": (
            "REPAIR",
            "Worker reports repairable damage. Inspect the repair effort and material condition before routing.",
        ),
        "Beyond repair": (
            "MATERIAL TESTING / RECYCLING ASSESSMENT",
            "Worker reports that the garment is not repairable. Verify fibre composition before material-specific recycling.",
        ),
        "Unsure — needs inspection": (
            "MANUAL INSPECTION",
            "Insufficient evidence for automated routing. Ask a trained worker to inspect the garment.",
        ),
    }
    return routing.get(condition, routing["Unsure — needs inspection"])


def init_db(db_path: Path = DB_PATH) -> None:
    """Create local inspection log table and safely migrate new columns in-place."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                image_name TEXT NOT NULL,
                garment_type TEXT NOT NULL,
                condition TEXT NOT NULL,
                observed_damage TEXT NOT NULL,
                care_label_text TEXT NOT NULL,
                composition TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                rationale TEXT NOT NULL,
                worker_notes TEXT NOT NULL
            )
            """
        )
        existing_cols = {
            row[1]
            for row in connection.execute("PRAGMA table_info(inspections)").fetchall()
        }
        for col_name, col_def in _MIGRATION_COLUMNS.items():
            if col_name not in existing_cols:
                connection.execute(
                    f"ALTER TABLE inspections ADD COLUMN {col_name} {col_def}"
                )
        connection.commit()


def save_inspection(record: dict[str, Any], db_path: Path = DB_PATH) -> str:
    """Persist inspection evidence & metadata. Original image files are not stored."""
    init_db(db_path=db_path)
    inspection_id = str(uuid.uuid4())[:8].upper()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    yolo_detections = record.get("yolo_detections", [])
    yolo_max_conf = record.get("yolo_max_confidence")
    if yolo_max_conf is None and yolo_detections:
        yolo_max_conf = max(float(d.get("confidence", 0.0)) for d in yolo_detections)

    confirmation_status = record.get("confirmation_status", "PENDING")
    confirmed_pathway = record.get("confirmed_pathway", "")
    confirmed_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
        if confirmation_status == "CONFIRMED"
        else None
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO inspections (
                id, created_at, image_name, garment_type, condition,
                observed_damage, care_label_text, composition,
                recommendation, rationale, worker_notes,
                yolo_detections, yolo_max_confidence, ocr_confidence,
                composition_evidence, care_instructions,
                confirmation_status, confirmed_pathway, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                created_at,
                record["image_name"],
                record["garment_type"],
                record["condition"],
                json.dumps(record.get("observed_damage", [])),
                record.get("care_label_text", ""),
                record.get("composition", "UNKNOWN"),
                record["recommendation"],
                record.get("rationale", ""),
                record.get("worker_notes", ""),
                json.dumps(yolo_detections),
                yolo_max_conf,
                record.get("ocr_confidence"),
                record.get("composition_evidence", ""),
                json.dumps(record.get("care_instructions", [])),
                confirmation_status,
                confirmed_pathway,
                confirmed_at,
            ),
        )
        connection.commit()
    return inspection_id


def confirm_inspection(
    inspection_id: str,
    confirmed_pathway: str,
    worker_notes: str | None = None,
    db_path: Path = DB_PATH,
) -> bool:
    """Record explicit worker confirmation (or override) of a saved inspection."""
    init_db(db_path=db_path)
    confirmed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as connection:
        if worker_notes is not None:
            cursor = connection.execute(
                """
                UPDATE inspections
                SET confirmation_status = 'CONFIRMED',
                    confirmed_pathway = ?,
                    confirmed_at = ?,
                    worker_notes = ?
                WHERE id = ?
                """,
                (confirmed_pathway, confirmed_at, worker_notes, inspection_id),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE inspections
                SET confirmation_status = 'CONFIRMED',
                    confirmed_pathway = ?,
                    confirmed_at = ?
                WHERE id = ?
                """,
                (confirmed_pathway, confirmed_at, inspection_id),
            )
        connection.commit()
        return cursor.rowcount > 0


def fetch_inspections(db_path: Path = DB_PATH):
    """Return all inspection records as a pandas DataFrame."""
    import pandas as pd

    init_db(db_path=db_path)
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM inspections ORDER BY created_at DESC", connection
        )
