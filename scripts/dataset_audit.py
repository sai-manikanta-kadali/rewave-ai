from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - helpful CLI error
    raise SystemExit("PyYAML is required. Install it with: python -m pip install pyyaml") from exc

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_yaml(dataset_dir: Path, requested: Path | None = None) -> Path:
    if requested:
        candidate = requested if requested.is_absolute() else dataset_dir / requested
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {candidate}")
        return candidate

    preferred = [dataset_dir / "data.yaml", dataset_dir / "dataset.yaml", dataset_dir / "data.yml"]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    matches = sorted(p for p in dataset_dir.rglob("*.yaml") if p.is_file())
    matches += sorted(p for p in dataset_dir.rglob("*.yml") if p.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No YOLO data YAML found under {dataset_dir}. The archive may be classification-only or use another annotation format."
        )
    listing = "\n".join(str(p.relative_to(dataset_dir)) for p in matches[:20])
    raise RuntimeError("Multiple YAML files found. Pass --yaml to choose one:\n" + listing)


def load_config(yaml_path: Path) -> tuple[dict[str, Any], Path, dict[str, str]]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("YOLO YAML must contain a mapping of dataset settings.")
    names = data.get("names")
    if isinstance(names, list):
        names_map = {str(i): str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names_map = {str(k): str(v) for k, v in names.items()}
    else:
        raise ValueError("YOLO YAML must define class names in a 'names' list or mapping.")
    if not names_map:
        raise ValueError("No class names found in dataset YAML.")

    root_value = data.get("path", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    else:
        root = root.resolve()
    return data, root, names_map


def resolve_images(root: Path, split_value: Any, yaml_parent: Path) -> list[Path]:
    if split_value is None:
        return []
    entries = split_value if isinstance(split_value, list) else [split_value]
    found: list[Path] = []
    for entry in entries:
        path = Path(str(entry))
        if not path.is_absolute():
            # Ultralytics resolves split paths relative to dataset 'path'; txt file entries are relative to the txt file.
            path = root / path
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                if image_path.suffix.lower() in IMAGE_SUFFIXES and image_path.is_file():
                    found.append(image_path.resolve())
        elif path.is_dir():
            found.extend(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path.resolve())
        else:
            # Permit a path rooted relative to YAML folder as a diagnostic fallback.
            fallback = yaml_parent / str(entry)
            if fallback.is_dir():
                found.extend(p.resolve() for p in fallback.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(set(found))


def label_path_for(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    # Official Ultralytics convention: replace the last /images/ segment with /labels/.
    indices = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if not indices:
        return None
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_dir: Path, yaml_arg: Path | None = None) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    yaml_path = find_yaml(dataset_dir, yaml_arg)
    config, dataset_root, names_map = load_config(yaml_path)
    class_ids = {int(k) for k in names_map}
    if class_ids != set(range(len(class_ids))):
        raise ValueError(f"Class IDs must be contiguous and zero-indexed. Found: {sorted(class_ids)}")

    splits: dict[str, Any] = {}
    hashes_by_split: dict[str, set[str]] = {}
    all_diagnostics: list[str] = []
    total_images = 0
    total_boxes = 0
    all_label_files_found = True

    for split in ("train", "val", "test"):
        image_paths = resolve_images(dataset_root, config.get(split), yaml_path.parent)
        if not image_paths:
            if split in ("train", "val"):
                all_diagnostics.append(f"Required split '{split}' contains no readable images.")
            splits[split] = {"images": 0, "label_files": 0, "empty_label_files": 0, "missing_label_files": 0, "boxes": 0, "class_counts": {}, "invalid_rows": [], "examples": []}
            hashes_by_split[split] = set()
            continue

        counts: Counter[str] = Counter()
        label_files = empty_label_files = missing_label_files = box_count = 0
        invalid_rows: list[str] = []
        example_labels: list[str] = []
        split_hashes: set[str] = set()

        for image_path in image_paths:
            try:
                split_hashes.add(sha256(image_path))
            except OSError as exc:
                all_diagnostics.append(f"Could not hash {image_path}: {exc}")

            label_path = label_path_for(image_path)
            if label_path is None:
                all_label_files_found = False
                missing_label_files += 1
                if len(invalid_rows) < 10:
                    invalid_rows.append(f"No /images/ folder segment to derive label path for: {image_path}")
                continue
            if not label_path.is_file():
                # In detection datasets a missing label file may mean a negative image, but all-missing indicates a bad export.
                missing_label_files += 1
                continue

            label_files += 1
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                empty_label_files += 1
                continue
            for row_number, line in enumerate(content.splitlines(), start=1):
                fields = line.strip().split()
                if len(fields) != 5:
                    if len(invalid_rows) < 30:
                        invalid_rows.append(f"{label_path}:{row_number}: expected 5 fields (class x_center y_center width height), got {len(fields)}")
                    continue
                try:
                    class_id = int(float(fields[0]))
                    coords = [float(v) for v in fields[1:]]
                except ValueError:
                    if len(invalid_rows) < 30:
                        invalid_rows.append(f"{label_path}:{row_number}: non-numeric YOLO values")
                    continue
                if class_id not in class_ids:
                    if len(invalid_rows) < 30:
                        invalid_rows.append(f"{label_path}:{row_number}: class ID {class_id} not declared in names")
                    continue
                x, y, width, height = coords
                if not all(0.0 <= v <= 1.0 for v in coords) or width <= 0 or height <= 0:
                    if len(invalid_rows) < 30:
                        invalid_rows.append(f"{label_path}:{row_number}: coordinates must be normalized 0..1 and width/height > 0")
                    continue
                counts[names_map[str(class_id)]] += 1
                box_count += 1
                if len(example_labels) < 5:
                    example_labels.append(f"{names_map[str(class_id)]}: {' '.join(fields[1:])}")

        total_images += len(image_paths)
        total_boxes += box_count
        hashes_by_split[split] = split_hashes
        splits[split] = {
            "images": len(image_paths),
            "label_files": label_files,
            "empty_label_files": empty_label_files,
            "missing_label_files": missing_label_files,
            "boxes": box_count,
            "class_counts": dict(counts),
            "invalid_rows": invalid_rows,
            "examples": example_labels,
        }
        if split in ("train", "val") and label_files == 0:
            all_diagnostics.append(f"No YOLO label .txt files found for '{split}'. This may be an image-classification export, not an object-detection export.")
        if invalid_rows:
            all_diagnostics.append(f"Split '{split}' contains invalid annotation rows; see report.")

    duplicates: list[str] = []
    split_names = list(hashes_by_split)
    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:
            common = hashes_by_split[first] & hashes_by_split[second]
            if common:
                duplicates.append(f"{len(common)} identical image(s) appear in both {first} and {second} splits.")
    all_diagnostics.extend(duplicates)

    # 'defect-free' as an object class is a semantic warning: backgrounds should normally have empty box labels.
    suspicious_background_classes = [name for name in names_map.values() if name.strip().lower() in {"defect-free", "defect free", "no defect", "no_defect", "background"}]
    semantic_review_required = bool(suspicious_background_classes)
    if suspicious_background_classes:
        all_diagnostics.append(
            "TRAINING BLOCKED: class name(s) " + ", ".join(suspicious_background_classes) +
            " look like image-level/background labels. For object detection, clean images should normally have no defect boxes (empty/missing label file), not a full-image 'defect-free' box. Inspect and correct the annotations/class map before training."
        )

    train_boxes = splits.get("train", {}).get("boxes", 0)
    val_boxes = splits.get("val", {}).get("boxes", 0)
    train_labels = splits.get("train", {}).get("label_files", 0)
    val_labels = splits.get("val", {}).get("label_files", 0)
    has_train_val = splits.get("train", {}).get("images", 0) > 0 and splits.get("val", {}).get("images", 0) > 0
    no_invalid = not any(splits[s].get("invalid_rows") for s in splits)
    no_overlap = not duplicates
    detection_ready = bool(has_train_val and train_labels > 0 and val_labels > 0 and train_boxes > 0 and val_boxes > 0 and no_invalid and no_overlap and not semantic_review_required)

    return {
        "dataset_dir": str(dataset_dir),
        "yaml_path": str(yaml_path),
        "dataset_root": str(dataset_root),
        "class_names": names_map,
        "splits": splits,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "duplicate_warnings": duplicates,
        "diagnostics": all_diagnostics,
        "detection_ready": detection_ready,
        "semantic_review_required": semantic_review_required,
        "note": "detection_ready means basic YOLO structure and the automated background-class guard passed; it does not establish annotation quality, garment-domain suitability, or model accuracy.",
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_audit.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# ReWeave AI — Dataset audit",
        "",
        f"- Dataset directory: `{report['dataset_dir']}`",
        f"- YAML: `{report['yaml_path']}`",
        f"- Total images: **{report['total_images']}**",
        f"- Validated bounding boxes: **{report['total_boxes']}**",
        f"- Structural YOLO detection-ready: **{'YES' if report['detection_ready'] else 'NO'}**",
        "",
        "> Detection-ready is a format check only. It does not certify dataset quality, garment-domain suitability, or accuracy.",
        "",
        "## Classes",
    ]
    for key, value in report["class_names"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Splits"]
    for split, details in report["splits"].items():
        lines += [
            f"### {split}",
            f"- Images: {details['images']}",
            f"- Label files: {details['label_files']}",
            f"- Empty label files: {details['empty_label_files']}",
            f"- Missing label files: {details['missing_label_files']}",
            f"- Bounding boxes: {details['boxes']}",
            f"- Class counts: `{details['class_counts']}`",
        ]
        if details["invalid_rows"]:
            lines.append("- Invalid annotation examples:")
            lines.extend(f"  - {item}" for item in details["invalid_rows"][:15])
        if details["examples"]:
            lines.append("- Example labels:")
            lines.extend(f"  - `{item}`" for item in details["examples"])
    lines += ["", "## Warnings"]
    lines.extend(f"- {warning}" for warning in report["diagnostics"] or ["No structural warnings found."])
    (output_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {output_dir / 'dataset_audit.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a dataset before ReWeave AI YOLO training.")
    parser.add_argument("dataset_dir", type=Path, help="Folder containing the extracted dataset archive")
    parser.add_argument("--yaml", type=Path, default=None, help="Optional path to the dataset YAML (relative to dataset_dir or absolute)")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Report output folder")
    args = parser.parse_args()

    try:
        report = audit_dataset(args.dataset_dir, args.yaml)
        write_report(report, args.out)
    except Exception as exc:
        print(f"DATASET AUDIT FAILED: {exc}", file=sys.stderr)
        return 2

    print("\nDATASET AUDIT SUMMARY")
    print(f"Images: {report['total_images']}")
    print(f"Bounding boxes parsed: {report['total_boxes']}")
    print(f"Classes: {report['class_names']}")
    print(f"YOLO detection format structurally ready: {'YES' if report['detection_ready'] else 'NO'}")
    for warning in report["diagnostics"]:
        print(f"WARNING: {warning}")
    return 0 if report["detection_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
