# peaceofcake
<img width="300" src="https://github.com/user-attachments/assets/1c98acd6-8fb7-476d-919a-57ee82b0dd9b">

A simple Python library for state-of-the-art object detection. Supports [D-FINE](https://github.com/Peterande/D-FINE) and [RF-DETR](https://github.com/roboflow/rf-detr) models with a unified Ultralytics-style API. Pretrained weights are downloaded automatically. Includes an iOS demo app with real-time camera detection.

<img width="200" height="435" alt="Image" src="https://github.com/user-attachments/assets/bde0438e-5c56-4528-a083-2952106e8073" />

## Installation

```bash
pip install peaceofcake
```

Optional dependencies:

```bash
pip install peaceofcake[export]        # ONNX, CoreML export
pip install peaceofcake[rfdetr-train]  # RF-DETR training
```

Requirements: Python >= 3.10, PyTorch >= 2.2

## Quick Start

```python
from peaceofcake import DFINE

model = DFINE("dfine-n-coco")
results = model("image.jpg", conf=0.3)  # __call__ shorthand
results[0].save("output.jpg")
```

## CLI

```bash
poc predict source=image.jpg conf=0.3
poc train   model=dfine-m-coco data=dataset.yaml epochs=50 batch_size=16
poc val     model=dfine-l-coco data=dataset.yaml
poc export  model=dfine-l-coco format=coreml img_size=640 precision=FLOAT16
poc info    model=dfine-l-coco

# RF-DETR
poc predict model=rfdetr-l-coco source=image.jpg conf=0.3
poc train   model=rfdetr-m-coco data=dataset/ epochs=50
```

## Available Models

### D-FINE

| Model | Dataset | Size |
|---|---|---|
| `dfine-n-coco` | COCO | Nano (fastest) |
| `dfine-s-coco` | COCO | Small |
| `dfine-m-coco` | COCO | Medium |
| `dfine-l-coco` | COCO | Large |
| `dfine-x-coco` | COCO | XLarge (best accuracy) |
| `dfine-s-obj2coco` | Objects365+COCO | Small |
| `dfine-m-obj2coco` | Objects365+COCO | Medium |
| `dfine-l-obj2coco` | Objects365+COCO | Large |
| `dfine-x-obj2coco` | Objects365+COCO | XLarge |

### RF-DETR

| Model | Dataset | Size |
|---|---|---|
| `rfdetr-n-coco` | COCO | Nano |
| `rfdetr-s-coco` | COCO | Small |
| `rfdetr-m-coco` | COCO | Medium |
| `rfdetr-l-coco` | COCO | Large |

Weights are cached in `~/.cache/peaceofcake/weights/`.

## API

### Inference

```python
from peaceofcake import DFINE, RFDETR

model = DFINE("dfine-n-coco")   # or RFDETR("rfdetr-l-coco")

# From file path, PIL Image, numpy array, or list of paths
results = model.predict("image.jpg", conf=0.25, device="cpu", img_size=640)
```

| Parameter | Default | Description |
|---|---|---|
| `source` | — | File path, list of paths, PIL Image, or numpy array |
| `conf` | 0.25 | Confidence threshold |
| `device` | auto | `"cpu"` or `"cuda"` |
| `img_size` | 640 | Input resolution |

### Results

```python
r = results[0]
r.boxes       # (N, 4) bounding boxes in xyxy format
r.labels      # (N,) class indices
r.scores      # (N,) confidence scores
len(r)        # number of detections
print(r)      # human-readable summary

r.plot()      # returns PIL Image with drawn boxes
r.save("out.jpg")  # save visualization
```

### Training

```python
model = DFINE("dfine-m-coco")
model.train(data="dataset.yaml", epochs=50, batch_size=16, img_size=640)
```

Supports both YOLO and COCO dataset formats. YOLO format is auto-converted.

| Parameter | Default | Description |
|---|---|---|
| `data` | — | Path to dataset YAML (YOLO or COCO format) |
| `epochs` | model default | Number of training epochs |
| `batch_size` | model default | Batch size |
| `img_size` | 640 | Input resolution |
| `output_dir` | `./runs/detect/train` | Output directory (auto-incremented) |
| `resume` | — | `True` (auto-find) or path to checkpoint |

### Validation

```python
results = model.val(data="dataset.yaml")
print(results)  # mAP50-95, mAP50, mAP75, etc.
```

### Export

```python
model.export("onnx")                # ONNX
model.export("coreml")              # CoreML (.mlpackage)
model.export("coreml", img_size=640, precision="FLOAT16", min_target="iOS17")
model.export("tensorrt")            # TensorRT (requires trtexec)
```

| Parameter | Default | Description |
|---|---|---|
| `img_size` | 640 | Input resolution |
| `min_target` | `"iOS17"` | `"iOS16"`, `"iOS17"`, `"iOS18"` |
| `precision` | `"FLOAT16"` | `"FLOAT16"`, `"FLOAT32"` |
| `compute_units` | `"ALL"` | `"ALL"`, `"CPU_AND_GPU"`, `"CPU_AND_NE"`, `"CPU_ONLY"` |
| `output` | auto | Output path |

CoreML model outputs:
- `confidence` — `[N, 80]` class scores
- `coordinates` — `[N, 4]` bounding boxes (normalized cxcywh)

## iOS Demo App

The `DFINEDemo/` directory contains a SwiftUI iOS app with:

- Real-time camera object detection
- Photo library detection
- Confidence threshold slider
- Model picker (when multiple models are bundled)

### Setup

1. Export a CoreML model:
   ```python
   from peaceofcake import DFINE
   model = DFINE("dfine-n-coco")
   model.export("coreml", output="dfine_n_coco.mlpackage")
   ```

2. Drag the `.mlpackage` into `DFINEDemo/DFINEDemo/` in Xcode

3. Build and run on device (iOS 17+)

To use multiple models, add more `.mlpackage` files with `dfine` prefix. A model picker appears automatically in the toolbar.

## License

Apache 2.0

## Acknowledgments

- [D-FINE](https://github.com/Peterande/D-FINE) by Peterande et al. — D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement.
- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow — RF-DETR: Real-Time, Foundational Object Detection.
