from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_audit import audit_dataset, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained ReWeave detector on the held-out test split (or val if no test split exists).")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--out", type=Path, default=Path("reports/model_evaluation.json"))
    args = parser.parse_args()

    report = audit_dataset(args.dataset_dir, args.yaml)
    write_report(report, Path("reports"))
    yaml_path = Path(report["yaml_path"])
    config_split = args.split
    split_images = report["splits"].get(config_split, {}).get("images", 0)
    if args.split == "test" and split_images == 0:
        print("No test split found. Refusing to silently treat validation data as a held-out test set.")
        print("Run with --split val for development evaluation, and label the result validation-only.")
        return 3
    if split_images == 0:
        print(f"No images found in split '{args.split}'.", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Could not import Ultralytics: {exc}", file=sys.stderr)
        return 4
    if not args.weights.is_file():
        print(f"Weights file not found: {args.weights}", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(yaml_path), split=config_split, imgsz=args.imgsz, device=args.device, plots=True)
    box = metrics.box
    output = {
        "weights": str(args.weights.resolve()),
        "dataset_yaml": str(yaml_path.resolve()),
        "evaluated_split": config_split,
        "evaluation_kind": "held-out test" if config_split == "test" else "validation only — not an independent final test",
        "images_in_split": split_images,
        "precision_mean": float(box.mp),
        "recall_mean": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class_map50_95": [float(v) for v in box.maps],
        "class_names": report["class_names"],
        "limitations": [
            "Metrics only apply to this dataset and split.",
            "Fabric-scrap results do not establish performance on worn post-consumer garments.",
            "Do not claim accuracy without a representative, independently labelled garment test set.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved evaluation: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
