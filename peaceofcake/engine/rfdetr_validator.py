from pathlib import Path
from typing import Any, Dict

import torch
import torchvision.transforms as T
from PIL import Image


class RFDETRValidator:
    """Wraps RF-DETR evaluation for simple model.val() API."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_wrapper, overrides: Dict[str, Any] = None):
        self.model_wrapper = model_wrapper
        self.overrides = overrides or {}

    def validate(self) -> Dict:
        data = self.overrides.get("data")
        if data is None:
            raise ValueError("data= is required for validation.")

        val_images, ann_file = self._resolve_val_data(data)
        if ann_file is None:
            raise ValueError(
                "Cannot find COCO annotation file for validation. "
                "Provide a dataset with val_ann key or COCO-format annotations."
            )

        return self._run_coco_eval(val_images, ann_file)

    def _run_coco_eval(self, img_dir: str, ann_file: str) -> Dict:
        """Run COCO-style evaluation using faster-coco-eval."""
        try:
            from faster_coco_eval import COCO, COCOeval_faster
        except ImportError:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval as COCOeval_faster

        coco_gt = COCO(ann_file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(self.overrides.get("device", device))

        resolution = self.overrides.get("img_size", self.model_wrapper._model_resolution)

        # Build deploy model
        from peaceofcake.engine.rfdetr_predictor import RFDETRPredictor
        predictor = RFDETRPredictor(self.model_wrapper)
        predictor._build(device)

        transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        results_list = []
        img_ids = coco_gt.getImgIds()

        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = Path(img_dir) / img_info["file_name"]
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                labels, boxes, scores = predictor._deploy_model(
                    img_t, torch.tensor([[h, w]], device=device)
                )

            for i in range(labels.shape[1]):
                score = scores[0, i].item()
                if score < 0.001:
                    continue
                box = boxes[0, i].cpu().tolist()
                # Convert xyxy to xywh for COCO
                x1, y1, x2, y2 = box
                coco_box = [x1, y1, x2 - x1, y2 - y1]
                results_list.append({
                    "image_id": img_id,
                    "category_id": int(labels[0, i].item()),
                    "bbox": coco_box,
                    "score": score,
                })

        if not results_list:
            print("No detections produced.")
            return {}

        coco_dt = coco_gt.loadRes(results_list)
        coco_eval = COCOeval_faster(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {
            "mAP50-95": float(stats[0]),
            "mAP50": float(stats[1]),
            "mAP75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
        }

    def _resolve_val_data(self, data):
        """Resolve data argument to (val_images_dir, val_ann_file)."""
        import yaml as _yaml

        if isinstance(data, str) and data.endswith((".yml", ".yaml")):
            data_path = Path(data).resolve()
            with open(data_path) as f:
                cfg = _yaml.safe_load(f)

            base_dir = data_path.parent

            # Resolve 'path' key
            if "path" in cfg:
                path_val = Path(cfg["path"])
                if not path_val.is_absolute():
                    path_val = (base_dir / path_val).resolve()
                base_dir = path_val

            # Normalize valid -> val
            if "valid" in cfg and "val" not in cfg:
                cfg["val"] = cfg.pop("valid")

            # If explicit annotation path
            val_ann = cfg.get("val_ann")
            if val_ann:
                if not Path(val_ann).is_absolute():
                    val_ann = str((base_dir / val_ann).resolve())
                val_dir = cfg.get("val", "")
                if val_dir and not Path(val_dir).is_absolute():
                    val_dir = str((base_dir / val_dir).resolve())
                return val_dir, val_ann

            # COCO-format auto-detection
            val_dir = cfg.get("val", "")
            if val_dir and not Path(val_dir).is_absolute():
                val_dir = str((base_dir / val_dir).resolve())

            if val_dir:
                # Look for annotation file in common locations
                for ann_candidate in [
                    Path(val_dir) / "_annotations.coco.json",
                    Path(val_dir).parent / "annotations" / "instances_val2017.json",
                    Path(val_dir).parent / "val.json",
                ]:
                    if ann_candidate.exists():
                        return val_dir, str(ann_candidate)

            return val_dir or str(base_dir), None

        if isinstance(data, str) and Path(data).is_dir():
            data_dir = Path(data).resolve()
            for ann_candidate in [
                data_dir / "_annotations.coco.json",
                data_dir / "annotations" / "instances_val2017.json",
                data_dir / "val.json",
            ]:
                if ann_candidate.exists():
                    return str(data_dir), str(ann_candidate)
            return str(data_dir), None

        raise ValueError(f"Unsupported data argument: {data}")
