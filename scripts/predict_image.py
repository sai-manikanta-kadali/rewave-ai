from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a trained ReWeave fabric-defect detector on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("runs/predict/reweave"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.weights.is_file():
        parser.error(f"Weights do not exist: {args.weights}")
    if not 0.0 < args.conf < 1.0:
        parser.error("--conf must be between 0 and 1")

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), conf=args.conf, save=True, project=str(args.out.parent), name=args.out.name, exist_ok=True, verbose=False)
    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": result.names.get(class_id, str(class_id)),
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy_pixels": [round(float(x), 2) for x in box.xyxy[0].tolist()],
            })
    output = {
        "image": str(args.image.resolve()),
        "weights": str(args.weights.resolve()),
        "confidence_threshold": args.conf,
        "detections": detections,
        "annotated_output_dir": str(args.out.resolve()),
        "notice": "Predictions are model suggestions. A worker must inspect and confirm before routing.",
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
