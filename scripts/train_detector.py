from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_audit import audit_dataset, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a ReWeave fabric-defect YOLO detector after auditing labels.")
    parser.add_argument("dataset_dir", type=Path, help="Extracted YOLO-format dataset directory")
    parser.add_argument("--yaml", type=Path, default=None, help="Dataset YAML path, if not auto-detected")
    parser.add_argument("--model", default="yolo26n.pt", help="Starting Ultralytics model weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4, help="Start small on a MacBook Air")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"], help="auto uses Apple MPS when available, otherwise CPU")
    parser.add_argument("--workers", type=int, default=0, help="0 is conservative for macOS multiprocessing")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="reweave_fabric_defects")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    try:
        report = audit_dataset(args.dataset_dir, args.yaml)
        write_report(report, Path("reports"))
    except Exception as exc:
        print(f"Could not audit dataset: {exc}", file=sys.stderr)
        return 2

    if not report["detection_ready"]:
        print("\nSTOP: dataset is not structurally ready for object detection.")
        print("Read reports/dataset_audit.md. Do not train on image-level class labels as if they were defect boxes.")
        return 3

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Training dependencies are not available in this Python environment: {exc}", file=sys.stderr)
        return 4

    if args.epochs < 1 or args.imgsz < 128 or args.batch < 1:
        print("epochs, batch must be positive and imgsz must be at least 128.", file=sys.stderr)
        return 2

    device = args.device
    if device == "auto":
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print(f"Starting model: {args.model}")
    print(f"Dataset YAML: {report['yaml_path']}")
    print(f"Device: {device}")
    print("Reminder: training-set metrics are not evaluation. Preserve a separate test split for final reporting.")

    model = YOLO(args.model)
    results = model.train(
        data=report["yaml_path"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        cache=False,
        seed=42,
        deterministic=True,
    )
    save_dir = Path(getattr(results, "save_dir", args.project / args.name)).resolve()
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        print(f"Training finished but best.pt was not found at expected location: {best_weights}", file=sys.stderr)
        return 5

    metadata = {
        "model": args.model,
        "dataset_yaml": str(Path(report["yaml_path"]).resolve()),
        "class_names": report["class_names"],
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "best_weights": str(best_weights),
        "dataset_audit": "reports/dataset_audit.json",
        "note": "This file records training configuration only; use the test split for independent performance claims.",
    }
    (save_dir / "reweave_training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTRAINING COMPLETE")
    print(f"Best weights: {best_weights}")
    print("Next: run scripts/evaluate_detector.py with the same dataset directory and --weights path above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
