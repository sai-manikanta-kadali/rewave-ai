# ReWeave AI — YOLO Fabric-Defect Training Toolkit

This toolkit is the next step after the existing Streamlit + OCR + SQLite MVP. It does **not** replace the current app and does not claim a trained model exists yet.

## Dataset candidate and an important caveat

FabriD (Mendeley Data, Version 2, DOI `10.17632/vncdwps24r.2`) is listed as 2,407 fabric images covering needle marks, holes, stains, missing yarn and defect-free samples, under CC BY 4.0:

https://data.mendeley.com/datasets/vncdwps24r/2

The dataset description mentions bounding-box/polygon annotations, but also describes image-level class labels. This is a metadata ambiguity. **Inspect the downloaded archive and the actual annotation files before training.** A classification-only export is not a defect-localisation dataset. Do not invent boxes for it.

FabriD contains fabric images, not necessarily representative worn post-consumer clothes. Any trained model must be separately tested on real garments before claims about post-consumer sorting.

## 1. Add toolkit files to your existing project

Copy the `scripts/` folder from this toolkit into the root of your existing `reweave_ai_mvp` folder. You can keep `app.py`, `reweave_core.py`, SQLite database and tests unchanged.

Install requirements inside the project's activated virtual environment:

```bash
python -m pip install pyyaml ultralytics
```

## 2. Download and extract the dataset

Open the official Mendeley link above, download the versioned dataset, and extract it into:

```text
reweave_ai_mvp/data/FabriD/
```

Do not move or rename the source labels until the archive structure has been inspected.

## 3. Audit before training (mandatory)

From the project root, with `.venv` active:

```bash
python scripts/dataset_audit.py data/FabriD --out reports
```

Read both:

- `reports/dataset_audit.md`
- `reports/dataset_audit.json`

The script checks YAML class definitions, train/val images, YOLO-format bounding-box rows, normalized coordinates, class IDs, missing/empty labels and duplicate image files across splits. `detection_ready: YES` means only structural checks passed; it is not proof that annotations are semantically correct or that the model will generalise.

If the audit fails, stop. Share the audit report or error so the dataset format can be adapted correctly. Do not run training on guessed formats.

## 4. Train a small baseline (only after audit passes)

```bash
python scripts/train_detector.py data/FabriD --epochs 30 --imgsz 640 --batch 4 --device auto
```

- The default starting weights are `yolo26n.pt` and will be downloaded by Ultralytics when needed.
- `--device auto` chooses Apple MPS if PyTorch reports it available, otherwise CPU.
- `--workers 0` is the default for conservative macOS operation.
- First run may download pretrained weights. Training time depends on Mac hardware and dataset.

Expected best weights:

```text
runs/detect/reweave_fabric_defects/weights/best.pt
```

If Ultralytics reports a different save directory, use the path it prints.

## 5. Evaluate honestly

If the dataset includes a real held-out `test` split:

```bash
python scripts/evaluate_detector.py data/FabriD --weights runs/detect/reweave_fabric_defects/weights/best.pt --split test
```

If there is no test split, you may inspect validation performance during development, but it must be labelled validation-only:

```bash
python scripts/evaluate_detector.py data/FabriD --weights runs/detect/reweave_fabric_defects/weights/best.pt --split val
```

The report records precision, recall, mAP50 and mAP50-95. These are detection metrics, not a blanket accuracy percentage. Do not tune repeatedly against the held-out test split.

## 6. Test one image

```bash
python scripts/predict_image.py /path/to/garment_or_fabric_photo.jpg --weights runs/detect/reweave_fabric_defects/weights/best.pt
```

The annotated image is saved by Ultralytics under `runs/predict/reweave/`.

## 7. Before integrating in the web app

Validate on a separate collection of real post-consumer garments, including clear and difficult examples, different lighting, different colours/textures and garment types. Have a person annotate the visible defects and calculate per-class precision/recall. Only then place the selected `best.pt` in the app's configured weights path and display detections as suggestions requiring worker confirmation.

## Safety and scientific honesty

- Ordinary RGB photos cannot chemically identify cotton/polyester blends.
- A defect detector does not decide reuse/repair/recycling by itself.
- No detection is not proof that a garment is defect-free.
- OCR output must be checked against the physical care label.
- Human confirmation remains part of the workflow.
- Keep the dataset citation and CC BY 4.0 attribution with any redistributed dataset-derived materials.

## Official technical references

- Ultralytics object-detection dataset format: https://docs.ultralytics.com/datasets/detect/
- Ultralytics detection training/evaluation: https://docs.ultralytics.com/tasks/detect/
