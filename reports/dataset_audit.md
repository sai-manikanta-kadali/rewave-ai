# ReWeave AI — Dataset audit

- Dataset directory: `/Users/kadalisaimanikanta/Downloads/reweave_ai_mvp/data/textile_defects`
- YAML: `/Users/kadalisaimanikanta/Downloads/reweave_ai_mvp/data/textile_defects/data.yaml`
- Total images: **1248**
- Validated bounding boxes: **2172**
- Structural YOLO detection-ready: **YES**

> Detection-ready is a format check only. It does not certify dataset quality, garment-domain suitability, or accuracy.

## Classes
- `0`: hole
- `1`: stain
- `2`: yarn missing

## Splits
### train
- Images: 963
- Label files: 963
- Empty label files: 96
- Missing label files: 0
- Bounding boxes: 1712
- Class counts: `{'yarn missing': 614, 'hole': 597, 'stain': 501}`
- Example labels:
  - `yarn missing: 0.46484375 0.48984375 0.6296875 0.06484375`
  - `yarn missing: 0.49765625 0.496875 0.8296875 0.05390625`
  - `yarn missing: 0.5 0.509375 1 0.05703125`
  - `yarn missing: 0.5203125 0.5 0.04765625 1`
  - `yarn missing: 0.49921875 0.51015625 0.9984375 0.034375`
### val
- Images: 227
- Label files: 227
- Empty label files: 22
- Missing label files: 0
- Bounding boxes: 380
- Class counts: `{'yarn missing': 151, 'stain': 117, 'hole': 112}`
- Example labels:
  - `yarn missing: 0.5140625 0.50703125 0.04375 0.509375`
  - `yarn missing: 0.33671875 0.49609375 0.6734375 0.06171875`
  - `yarn missing: 0.58046875 0.49921875 0.0328125 0.9984375`
  - `yarn missing: 0.50078125 0.425 0.9984375 0.0484375`
  - `yarn missing: 0.53515625 0.4671875 0.8015625 0.06328125`
### test
- Images: 58
- Label files: 58
- Empty label files: 6
- Missing label files: 0
- Bounding boxes: 80
- Class counts: `{'stain': 23, 'hole': 32, 'yarn missing': 25}`
- Example labels:
  - `stain: 0.51640625 0.7359375 0.07578125 0.1703125`
  - `hole: 0.53828125 0.43671875 0.0796875 0.134375`
  - `hole: 0.13515625 0.68984375 0.0484375 0.096875`
  - `hole: 0.63125 0.546875 0.14375 0.11171875`
  - `hole: 0.54921875 0.37421875 0.053125 0.11875`

## Warnings
- No structural warnings found.
