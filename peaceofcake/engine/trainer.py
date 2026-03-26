import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch


class DFINETrainer:
    """Wraps D-FINE's DetSolver for simple model.train() API."""

    def __init__(self, model_wrapper, overrides: Dict[str, Any] = None):
        self.model_wrapper = model_wrapper
        self.overrides = overrides or {}
        self.metrics = {}
        self.best_model = None

    def train(self):
        from src.core import YAMLConfig
        from src.solver import TASKS
        from src.misc import dist_utils
        from src import data, optim  # noqa: F401 — register training components
        from src.nn import criterion  # noqa: F401
        from src.zoo.dfine import _register_training_modules
        _register_training_modules()

        data_cfg = self._parse_data(self.overrides.get("data"))
        self._class_names = data_cfg.pop("class_names", None)
        yaml_overrides = self._build_overrides(data_cfg)

        cfg = YAMLConfig(self.model_wrapper._dfine_config_path, **yaml_overrides)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Scale two-stage training schedule to custom epoch count
        if "epochs" in self.overrides:
            self._scale_training_schedule(cfg)

        # Fine-tune from pretrained if weights were loaded
        if self.model_wrapper.ckpt_path:
            cfg.tuning = self.model_wrapper.ckpt_path

        dist_utils.setup_distributed(
            print_rank=0,
            print_method="builtin",
            seed=self.overrides.get("seed"),
        )

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.fit()

        # Save metadata (class names) for later checkpoint loading
        output_dir = Path(yaml_overrides.get("output_dir", "./runs/detect/train"))
        self._save_metadata(output_dir, self._class_names)

        # Store class names on model wrapper
        self.model_wrapper.class_names = self._class_names

        # Load best model back — rebuild with training config (num_classes may differ)
        for name in ["best_stg2.pth", "best_stg1.pth", "last.pth"]:
            best = output_dir / name
            if best.exists():
                train_cfg = YAMLConfig(
                    self.model_wrapper._dfine_config_path, **yaml_overrides
                )
                if "HGNetv2" in train_cfg.yaml_cfg:
                    train_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
                self.model_wrapper.model = train_cfg.model
                self.model_wrapper.cfg_obj = train_cfg

                ckpt = torch.load(best, map_location="cpu")
                state = ckpt.get("ema", {}).get("module", ckpt.get("model"))
                if state:
                    self.model_wrapper.model.load_state_dict(state)
                    self.best_model = self.model_wrapper.model
                break

        dist_utils.cleanup()

    @staticmethod
    def _scale_training_schedule(cfg):
        """Scale stop_epoch and augmentation policy to match custom epoch count."""
        epochs = cfg.yaml_cfg.get("epochs")
        if epochs is None:
            return

        train_dl = cfg.yaml_cfg.get("train_dataloader", {})
        collate_cfg = train_dl.get("collate_fn", {})
        default_stop = collate_cfg.get("stop_epoch")

        if default_stop is None or default_stop <= epochs:
            return

        scaled_stop = max(1, int(epochs * 0.9))
        collate_cfg["stop_epoch"] = scaled_stop

        policy = (
            train_dl
            .get("dataset", {})
            .get("transforms", {})
            .get("policy")
        )
        if policy and "epoch" in policy:
            policy["epoch"] = scaled_stop

    def _parse_data(self, data) -> Dict:
        if data is None:
            return {}
        if isinstance(data, dict):
            return self._handle_simple_or_yolo(data)
        if isinstance(data, str) and data.endswith((".yml", ".yaml")):
            yaml_path = Path(data).resolve()
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            cfg = self._resolve_data_paths(cfg, yaml_path.parent)
            if "train" in cfg or "train_images" in cfg:
                return self._handle_simple_or_yolo(cfg)
            return cfg
        raise ValueError(f"Unsupported data argument: {data}")

    @staticmethod
    def _resolve_data_paths(cfg: Dict, base_dir: Path) -> Dict:
        """Resolve relative paths and normalize Roboflow/Ultralytics-style keys."""
        cfg = dict(cfg)

        # Normalize 'valid' -> 'val'
        if "valid" in cfg and "val" not in cfg:
            cfg["val"] = cfg.pop("valid")

        # Drop Roboflow metadata
        cfg.pop("roboflow", None)

        # Ultralytics 'path' key overrides base_dir
        if "path" in cfg:
            path_val = Path(cfg.pop("path"))
            if not path_val.is_absolute():
                path_val = (base_dir / path_val).resolve()
            base_dir = path_val

        # Resolve relative paths for split directories
        for key in ("train", "val", "test"):
            path = cfg.get(key)
            if path and not Path(path).is_absolute():
                cfg[key] = str((base_dir / path).resolve())

        # Resolve annotation paths too
        for key in ("train_ann", "val_ann", "test_ann"):
            path = cfg.get(key)
            if path and not Path(path).is_absolute():
                cfg[key] = str((base_dir / path).resolve())

        return cfg

    def _handle_simple_or_yolo(self, cfg: Dict) -> Dict:
        from peaceofcake.utils.converters import detect_yolo_dataset, convert_yolo_dataset

        # Extract class names before YOLO conversion (which pops 'names')
        class_names = self._extract_class_names(cfg)

        if detect_yolo_dataset(cfg):
            output_dir = self.overrides.get("output_dir", "./runs/detect/train")
            cache_dir = str(Path(output_dir) / ".yolo_cache")
            cfg = convert_yolo_dataset(cfg, cache_dir=cache_dir)

        # Validate: must have annotation files after conversion
        has_train_ann = "train_ann" in cfg
        has_val_ann = "val_ann" in cfg
        if not has_train_ann and not has_val_ann:
            train_path = cfg.get("train", "?")
            raise ValueError(
                f"Cannot find annotation files for dataset.\n"
                f"  train path: {train_path}\n"
                f"  Expected YOLO format (labels/ directory alongside images/) "
                f"or COCO format (train_ann/val_ann keys in YAML).\n"
                f"  If using YOLO format, ensure directory structure is:\n"
                f"    dataset/train/images/  and  dataset/train/labels/\n"
                f"    or  dataset/images/train/  and  dataset/labels/train/"
            )

        result = self._convert_simple_format(cfg)
        if class_names:
            result["class_names"] = class_names
        return result

    def _convert_simple_format(self, cfg: Dict) -> Dict:
        """Convert simple dataset YAML to D-FINE overrides."""
        result = {
            "num_classes": cfg.get("nc", cfg.get("num_classes", 80)),
            "remap_mscoco_category": False,
        }
        if "train" in cfg:
            result.setdefault("train_dataloader", {}).setdefault("dataset", {})["img_folder"] = cfg["train"]
        if "train_ann" in cfg:
            result.setdefault("train_dataloader", {}).setdefault("dataset", {})["ann_file"] = cfg["train_ann"]
        if "val" in cfg:
            result.setdefault("val_dataloader", {}).setdefault("dataset", {})["img_folder"] = cfg["val"]
        if "val_ann" in cfg:
            result.setdefault("val_dataloader", {}).setdefault("dataset", {})["ann_file"] = cfg["val_ann"]
        return result

    def _build_overrides(self, data_cfg: Dict) -> Dict:
        ov = self.overrides
        result = {}

        if "epochs" in ov:
            result["epochs"] = ov["epochs"]
        result["output_dir"] = ov.get("output_dir", "./runs/detect/train")

        if "batch_size" in ov:
            bs = ov["batch_size"]
            result.setdefault("train_dataloader", {})["total_batch_size"] = bs
            result.setdefault("val_dataloader", {})["total_batch_size"] = bs * 2

        if "num_workers" in ov:
            result.setdefault("train_dataloader", {})["num_workers"] = ov["num_workers"]
            result.setdefault("val_dataloader", {})["num_workers"] = ov["num_workers"]

        if "img_size" in ov:
            sz = ov["img_size"]
            result["eval_spatial_size"] = [sz, sz]

        for key in ["use_amp", "use_ema", "use_wandb", "seed"]:
            if key in ov:
                result[key] = ov[key]

        # Merge dataset config (skip non-override keys)
        for k, v in data_cfg.items():
            if k == "class_names":
                continue
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k].update(v)
            else:
                result[k] = v

        return result

    @staticmethod
    def _extract_class_names(cfg: Dict) -> Optional[List[str]]:
        """Extract class names from dataset config (YOLO or COCO style)."""
        names = cfg.get("names")
        if names is None:
            return None
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        if isinstance(names, list):
            return list(names)
        return None

    @staticmethod
    def _save_metadata(output_dir: Path, class_names: Optional[List[str]]):
        """Save training metadata alongside checkpoints."""
        if class_names is None:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {"class_names": class_names}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

    @staticmethod
    def load_metadata(ckpt_path: str) -> Optional[List[str]]:
        """Load class names from metadata.json next to a checkpoint file."""
        meta_path = Path(ckpt_path).parent / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("class_names")
