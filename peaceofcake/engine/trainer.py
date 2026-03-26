from pathlib import Path
from typing import Any, Dict

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
        yaml_overrides = self._build_overrides(data_cfg)

        cfg = YAMLConfig(self.model_wrapper._dfine_config_path, **yaml_overrides)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

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

        # Load best model back
        output_dir = Path(yaml_overrides.get("output_dir", "./runs/detect/train"))
        for name in ["best_stg2.pth", "best_stg1.pth", "last.pth"]:
            best = output_dir / name
            if best.exists():
                ckpt = torch.load(best, map_location="cpu")
                state = ckpt.get("ema", {}).get("module", ckpt.get("model"))
                if state:
                    self.model_wrapper.model.load_state_dict(state)
                    self.best_model = self.model_wrapper.model
                break

        dist_utils.cleanup()

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
        """Resolve relative paths and normalize Roboflow-style keys."""
        cfg = dict(cfg)

        # Normalize 'valid' -> 'val'
        if "valid" in cfg and "val" not in cfg:
            cfg["val"] = cfg.pop("valid")

        # Drop Roboflow metadata
        cfg.pop("roboflow", None)

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

        if detect_yolo_dataset(cfg):
            output_dir = self.overrides.get("output_dir", "./runs/detect/train")
            cache_dir = str(Path(output_dir) / ".yolo_cache")
            cfg = convert_yolo_dataset(cfg, cache_dir=cache_dir)
        return self._convert_simple_format(cfg)

    def _convert_simple_format(self, cfg: Dict) -> Dict:
        """Convert simple dataset YAML to D-FINE overrides.

        Simple format:
            train: /path/to/train/images
            val: /path/to/val/images
            train_ann: /path/to/train.json
            val_ann: /path/to/val.json
            nc: 10
            names: [class1, class2, ...]
        """
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

        # Merge dataset config
        for k, v in data_cfg.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k].update(v)
            else:
                result[k] = v

        return result
