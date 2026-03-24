# peaceofcake

A simple Python wrapper for [D-FINE](https://github.com/Peterande/D-FINE) object detection models. Pretrained weights are downloaded automatically. Includes an iOS demo app with real-time camera detection.

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.9, PyTorch >= 2.0

## Quick Start

```python
from peaceofcake import DFINE

model = DFINE("dfine-n-coco")
results = model.predict("image.jpg", conf=0.3)
results[0].save("output.jpg")
```

## Available Models

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

Weights are cached in `~/.cache/peaceofcake/weights/`.

## API

### Inference

```python
from peaceofcake import DFINE

model = DFINE("dfine-n-coco")

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

### Export

```python
model.export("onnx")                # ONNX
model.export("coreml")              # CoreML (.mlpackage)
model.export("coreml", img_size=640, precision="FLOAT16", min_target="iOS17")
model.export("tensorrt")            # TensorRT (requires trtexec)
```

#### CoreML Export Options

| Parameter | Default | Description |
|---|---|---|
| `img_size` | 640 | Input resolution |
| `min_target` | `"iOS17"` | `"iOS16"`, `"iOS17"`, `"iOS18"` |
| `precision` | `"FLOAT16"` | `"FLOAT16"`, `"FLOAT32"` |
| `compute_units` | `"ALL"` | `"ALL"`, `"CPU_AND_GPU"`, `"CPU_AND_NE"`, `"CPU_ONLY"` |
| `output` | `"model.mlpackage"` | Output path |

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

## Project Structure

```
peaceofcake/          # Python library
  models/dfine.py     # Model loading and registry
  engine/             # Predictor, exporter, trainer
  results/            # Detection results and plotting
  cfg/                # Model configs and defaults
third_party/dfine/    # Bundled D-FINE inference source
DFINEDemo/            # iOS demo app (SwiftUI)
```

## License

Apache 2.0

## Acknowledgments

This project wraps [D-FINE](https://github.com/Peterande/D-FINE) by Peterande et al.

> D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement.
