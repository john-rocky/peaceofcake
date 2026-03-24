import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


def yolo_to_coco(
    image_dir: str,
    label_dir: str,
    output_json: str,
    class_names: Optional[List[str]] = None,
    nc: Optional[int] = None,
) -> str:
    """Convert YOLO format labels to a COCO JSON annotation file.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing YOLO .txt label files.
        output_json: Path to write the output COCO JSON.
        class_names: List of class names. If None, uses generic names.
        nc: Number of classes. Inferred from class_names if not given.

    Returns:
        Path to the written JSON file.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if nc is None:
        nc = len(class_names) if class_names else 0

    if class_names is None:
        class_names = [f"class_{i}" for i in range(nc)]

    categories = [
        {"id": i, "name": name} for i, name in enumerate(class_names)
    ]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    images = []
    annotations = []
    ann_id = 0

    for img_id, img_path in enumerate(image_files):
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        txt_path = label_dir / (img_path.stem + ".txt")
        if not txt_path.exists():
            continue

        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert YOLO normalized cxcywh to COCO absolute xywh
            abs_w = bw * w
            abs_h = bh * h
            abs_x = cx * w - abs_w / 2
            abs_y = cy * h - abs_h / 2

            if nc == 0:
                nc = max(nc, cls_id + 1)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    # Fill in generic class names if nc grew
    while len(categories) < nc:
        categories.append({"id": len(categories), "name": f"class_{len(categories)}"})

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"Converted {len(images)} images, {len(annotations)} annotations -> {output_json}")
    return str(output_json)


def detect_yolo_dataset(cfg: Dict) -> bool:
    """Check if a dataset config dict looks like YOLO format.

    YOLO format: train/val point to image directories with sibling labels/ dirs,
    and no explicit train_ann/val_ann keys.
    """
    if "train_ann" in cfg or "val_ann" in cfg:
        return False

    for key in ("train", "val"):
        path = cfg.get(key)
        if not path:
            continue
        img_dir = Path(path)
        if not img_dir.is_dir():
            continue
        # Check for sibling labels directory
        label_dir = _find_label_dir(img_dir)
        if label_dir and label_dir.is_dir():
            return True

    return False


def _find_label_dir(image_dir: Path) -> Optional[Path]:
    """Find the corresponding labels directory for a YOLO image directory.

    Supports common layouts:
      - images/train/ -> labels/train/
      - train/images/ -> train/labels/
      - images/ -> labels/  (sibling)
    """
    parts = image_dir.parts
    for i, part in enumerate(parts):
        if part == "images":
            candidate = Path(*parts[:i]) / "labels" / Path(*parts[i + 1:])
            if candidate.is_dir():
                return candidate

    # Fallback: sibling directory named "labels"
    sibling = image_dir.parent / "labels"
    if sibling.is_dir():
        return sibling

    return None


def convert_yolo_dataset(cfg: Dict, cache_dir: str = ".peaceofcake_cache") -> Dict:
    """Convert a YOLO-format dataset config to COCO format.

    Reads image dirs, finds label dirs, converts to COCO JSON,
    and returns a new config dict with train_ann/val_ann paths.
    """
    cache_dir = Path(cache_dir)
    class_names = cfg.get("names")
    nc = cfg.get("nc", cfg.get("num_classes"))

    result = dict(cfg)

    for split in ("train", "val", "test"):
        img_path = cfg.get(split)
        if not img_path:
            continue

        img_dir = Path(img_path)
        if not img_dir.is_dir():
            continue

        label_dir = _find_label_dir(img_dir)
        if label_dir is None or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find labels directory for {img_dir}. "
                f"Expected 'labels' directory as sibling of 'images'."
            )

        ann_json = cache_dir / f"{split}_coco.json"
        yolo_to_coco(
            image_dir=str(img_dir),
            label_dir=str(label_dir),
            output_json=str(ann_json),
            class_names=class_names,
            nc=nc,
        )
        result[f"{split}_ann"] = str(ann_json)

    # Remove YOLO-specific keys
    result.pop("names", None)
    return result
