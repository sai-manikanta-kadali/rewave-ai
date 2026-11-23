"""YOLO textile defect inference wrapper for ReWeave AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from reweave_core import prepare_image

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "reweave_defect_yolo.pt"


def load_defect_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Load trained Ultralytics YOLO weights if available, else return None."""
    if not Path(model_path).is_file():
        return None
    from ultralytics import YOLO

    return YOLO(str(model_path))


def detect_defects(
    image: Image.Image,
    model: Any,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Run YOLO inference and return structured detections plus an annotated RGB image."""
    prepared = prepare_image(image)
    if model is None:
        return {
            "model_available": False,
            "detections": [],
            "annotated_image": prepared,
            "summary_text": "YOLO defect model not loaded.",
        }

    rgb = np.asarray(prepared)
    height, width = rgb.shape[:2]
    total_area = float(max(1, height * width))

    results = model.predict(source=rgb, conf=conf_threshold, verbose=False)
    detections: list[dict[str, Any]] = []
    annotated_rgb = rgb.copy()

    if results:
        result = results[0]
        names = result.names or {}
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = round(float(box.conf[0].item()), 3)
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                box_area = max(0.0, (x2 - x1) * (y2 - y1))
                area_ratio = round(box_area / total_area, 4)
                class_name = str(names.get(cls_id, cls_id))
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "area_ratio": area_ratio,
                    }
                )
            # Ultralytics plot() returns BGR; convert cleanly back to RGB for Streamlit/PIL.
            plotted_bgr = result.plot()
            annotated_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    if detections:
        parts = [f"{d['class_name']} ({d['confidence']:.0%})" for d in detections]
        summary_text = f"YOLO suggested {len(detections)} defect(s): " + ", ".join(parts)
    else:
        summary_text = "No supported defects detected above threshold (not proof garment is defect-free)."

    return {
        "model_available": True,
        "detections": detections,
        "annotated_image": Image.fromarray(annotated_rgb),
        "summary_text": summary_text,
    }
