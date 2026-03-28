<div align="center">

<img width="300" src="https://github.com/user-attachments/assets/1c98acd6-8fb7-476d-919a-57ee82b0dd9b">

**State-of-the-art object detection in 3 lines of Python. Fully Apache 2.0.**

[D-FINE](https://github.com/Peterande/D-FINE) + [RF-DETR](https://github.com/roboflow/rf-detr) with a unified Ultralytics-style API.
Train, export to CoreML/ONNX/TensorRT, and deploy to iPhone — all from one package.

```
pip install peaceofcake
```

[![PyPI](https://img.shields.io/pypi/v/peaceofcake?color=blue)](https://pypi.org/project/peaceofcake/)
[![Downloads](https://img.shields.io/pypi/dm/peaceofcake?color=green)](https://pypi.org/project/peaceofcake/)
[![Python](https://img.shields.io/pypi/pyversions/peaceofcake)](https://pypi.org/project/peaceofcake/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/john-rocky/peaceofcake?style=social)](https://github.com/john-rocky/peaceofcake)

<br>

<table>
<tr>
<td align="center" width="33%">
<b>3 Lines of Code</b><br>
<sub>Predict, train, export, validate<br>with a single unified API</sub>
</td>
<td align="center" width="33%">
<b>100% Apache 2.0</b><br>
<sub>Models, weights, and library<br>all free for commercial use</sub>
</td>
<td align="center" width="33%">
<b>iPhone Ready</b><br>
<sub>CoreML export + SwiftUI demo app<br>for real-time on-device detection</sub>
</td>
</tr>
</table>

<br>

<img width="200" height="435" alt="iOS Demo" src="https://github.com/user-attachments/assets/bde0438e-5c56-4528-a083-2952106e8073" />

</div>

## Quick Start

```python
from peaceofcake import DFINE

model = DFINE("dfine-n-coco")
results = model("image.jpg", conf=0.3)
results[0].save("output.jpg")
```

<details>
<summary><b>CLI</b></summary>

```bash
poc predict source=image.jpg conf=0.3
poc train   model=dfine-m-coco data=dataset.yaml epochs=50 batch_size=16
poc val     model=dfine-l-coco data=dataset.yaml
poc export  model=dfine-l-coco format=coreml img_size=640 precision=FLOAT32
poc info    model=dfine-l-coco

# RF-DETR
poc predict model=rfdetr-l-coco source=image.jpg conf=0.3
poc train   model=rfdetr-m-coco data=dataset/ epochs=50
```

</details>

<details>
<summary><b>Optional dependencies</b></summary>

```bash
pip install peaceofcake[export]        # ONNX, CoreML export
pip install peaceofcake[rfdetr-train]  # RF-DETR training
```

> Requirements: Python >= 3.10, PyTorch >= 2.2

</details>

## Models

All benchmarks on COCO val2017, NVIDIA T4 GPU, TensorRT FP16.

<details open>
<summary><b>D-FINE</b></summary>

| Model | AP | Params | Latency | GFLOPs | Resolution |
|:---|:---:|:---:|:---:|:---:|:---:|
| `dfine-n-coco` | 42.8 | 4M | 2.1ms | 7 | 640 |
| `dfine-s-coco` | 48.5 | 10M | 3.5ms | 25 | 640 |
| `dfine-m-coco` | 52.3 | 19M | 5.6ms | 57 | 640 |
| `dfine-l-coco` | 54.0 | 31M | 8.1ms | 91 | 640 |
| `dfine-x-coco` | 55.8 | 62M | 12.9ms | 202 | 640 |
| `dfine-s-obj2coco` | 50.7 | 10M | 3.5ms | 25 | 640 |
| `dfine-m-obj2coco` | 55.1 | 19M | 5.6ms | 57 | 640 |
| `dfine-l-obj2coco` | 57.1 | 31M | 8.1ms | 91 | 640 |
| `dfine-x-obj2coco` | 59.3 | 62M | 12.9ms | 202 | 640 |

</details>

<details open>
<summary><b>RF-DETR</b></summary>

| Model | AP | Params | Latency | GFLOPs | Resolution |
|:---|:---:|:---:|:---:|:---:|:---:|
| `rfdetr-n-coco` | 48.0 | 31M | 2.3ms | 32 | 384 |
| `rfdetr-s-coco` | 52.9 | 32M | 3.5ms | 60 | 512 |
| `rfdetr-m-coco` | 54.7 | 34M | 4.4ms | 79 | 576 |
| `rfdetr-l-coco` | 56.5 | 34M | 6.8ms | 126 | 704 |

</details>

<sub>AP = mAP@0.5:0.95. Weights are downloaded automatically and cached in <code>~/.cache/peaceofcake/weights/</code>.</sub>

## API

<details open>
<summary><b>Inference</b></summary>

```python
from peaceofcake import DFINE, RFDETR

model = DFINE("dfine-n-coco")   # or RFDETR("rfdetr-l-coco")

# From file path, PIL Image, numpy array, or list of paths
results = model.predict("image.jpg", conf=0.25, device="cpu", img_size=640)
```

| Parameter | Default | Description |
|:---|:---|:---|
| `source` | — | File path, list of paths, PIL Image, or numpy array |
| `conf` | `0.25` | Confidence threshold |
| `device` | auto | `"cpu"` or `"cuda"` |
| `img_size` | `640` | Input resolution |

</details>

<details open>
<summary><b>Results</b></summary>

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

</details>

<details>
<summary><b>Training</b></summary>

```python
model = DFINE("dfine-m-coco")
model.train(data="dataset.yaml", epochs=50, batch_size=16, img_size=640)
```

Supports both YOLO and COCO dataset formats. YOLO format is auto-converted.

| Parameter | Default | Description |
|:---|:---|:---|
| `data` | — | Path to dataset YAML (YOLO or COCO format) |
| `epochs` | model default | Number of training epochs |
| `batch_size` | model default | Batch size |
| `img_size` | `640` | Input resolution |
| `output_dir` | `./runs/detect/train` | Output directory (auto-incremented) |
| `resume` | — | `True` (auto-find) or path to checkpoint |

</details>

<details>
<summary><b>Validation</b></summary>

```python
results = model.val(data="dataset.yaml")
print(results)  # mAP50-95, mAP50, mAP75, etc.
```

</details>

<details>
<summary><b>Export</b></summary>

```python
model.export("onnx")                # ONNX
model.export("coreml")              # CoreML (.mlpackage)
model.export("coreml", img_size=640, precision="FLOAT32", min_target="iOS17")
model.export("tensorrt")            # TensorRT (requires trtexec)
```

| Parameter | Default | Description |
|:---|:---|:---|
| `img_size` | `640` | Input resolution |
| `min_target` | `"iOS17"` | `"iOS16"`, `"iOS17"`, `"iOS18"` |
| `precision` | `"FLOAT32"` | `"FLOAT32"`, `"FLOAT16"` |
| `compute_units` | `"ALL"` | `"ALL"`, `"CPU_AND_GPU"`, `"CPU_AND_NE"`, `"CPU_ONLY"` |
| `output` | auto | Output path |

CoreML model outputs:
- `confidence` — `[N, 80]` class scores
- `coordinates` — `[N, 4]` bounding boxes (normalized cxcywh)

</details>

## iOS Demo App

<div align="center">
<img width="300" src="https://github.com/user-attachments/assets/a9af3b06-dc8b-4384-88f3-765b85414b0f">
</div>

The `DFINEDemo/` directory contains a SwiftUI iOS app with real-time camera detection, photo library detection, video detection, confidence threshold slider, and model picker.

<details>
<summary><b>Setup instructions</b></summary>

1. Export a CoreML model:
   ```python
   from peaceofcake import DFINE
   model = DFINE("dfine-n-coco")
   model.export("coreml", output="dfine_n_coco.mlpackage")
   ```

2. Drag the `.mlpackage` into `DFINEDemo/DFINEDemo/` in Xcode

3. Build and run on device (iOS 17+)

To use multiple models, add more `.mlpackage` files with `dfine` prefix. A model picker appears automatically in the toolbar.

</details>

## Why peaceofcake?

| | Ultralytics (YOLO) | peaceofcake |
|:---|:---|:---|
| **Architecture** | YOLO (CNN-based) | D-FINE / RF-DETR (DETR-based) |
| **License** | AGPL-3.0 (paid for commercial) | Apache 2.0 (free for all) |
| **COCO AP** | up to 54.4 (YOLO11x) | up to 59.3 (D-FINE-X obj2coco) |
| **Install** | `pip install ultralytics` | `pip install peaceofcake` |
| **iOS demo** | - | Included (SwiftUI) |

## License

This project is licensed under [Apache 2.0](LICENSE) — free for personal and commercial use.

## Acknowledgments

- [D-FINE](https://github.com/Peterande/D-FINE) by Peterande et al. — D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement.
- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow — RF-DETR: Real-Time, Foundational Object Detection.
