from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch


class RFDETRTrainer:
    """Wraps RF-DETR training for simple model.train() API.

    Delegates to rfdetr's PyTorch Lightning training pipeline.
    Requires: pip install 'rfdetr[train]'
    """

    def __init__(self, model_wrapper, overrides: Dict[str, Any] = None):
        self.model_wrapper = model_wrapper
        self.overrides = overrides or {}
        self.metrics = {}
        self.best_model = None

    def train(self):
        data = self.overrides.get("data")
        if data is None:
            raise ValueError("data= is required. Provide a dataset directory or YAML file.")

        dataset_dir = self._resolve_dataset_dir(data)
        train_kwargs = self._build_train_kwargs(dataset_dir)

        rfdetr_obj = self.model_wrapper._rfdetr_obj
        rfdetr_obj.train(**train_kwargs)

        # Sync updated weights back
        self.model_wrapper.model = rfdetr_obj.model.model
        self.model_wrapper._postprocessor = rfdetr_obj.model.postprocess
        self.model_wrapper._num_classes = rfdetr_obj.model.args.num_classes
        if rfdetr_obj.model.class_names:
            self.model_wrapper.class_names = rfdetr_obj.model.class_names

        self.best_model = self.model_wrapper.model
        self.model_wrapper._predictor = None  # reset cached predictor

        # Embed class_names into checkpoint files
        output_dir = Path(train_kwargs.get("output_dir", "output"))
        self._embed_class_names(output_dir)

    def _resolve_dataset_dir(self, data) -> str:
        """Resolve data argument to a dataset directory path."""
        if isinstance(data, dict):
            # Direct dict with paths
            if "path" in data:
                return str(Path(data["path"]).resolve())
            if "train" in data:
                return str(Path(data["train"]).resolve().parent)
            raise ValueError("Cannot determine dataset directory from dict config.")

        data_path = Path(data)

        # If it's a YAML file, read it to find dataset paths
        if data_path.suffix in (".yml", ".yaml") and data_path.exists():
            with open(data_path) as f:
                cfg = yaml.safe_load(f)
            cfg = self._resolve_yaml_paths(cfg, data_path.parent)

            # If cfg has 'path' key (Ultralytics/Roboflow format), use it
            if "path" in cfg:
                return str(Path(cfg["path"]).resolve())

            # Try to find the common parent of train/val dirs
            for key in ("train", "val"):
                p = cfg.get(key)
                if p and Path(p).exists():
                    # Go up to find the dataset root
                    pp = Path(p).resolve()
                    # Typical structure: dataset/train/images -> dataset
                    if pp.name in ("images", "train"):
                        return str(pp.parent)
                    return str(pp.parent)

            # Fallback: use YAML file's parent as dataset dir
            return str(data_path.parent.resolve())

        # If it's a directory, use it directly
        if data_path.is_dir():
            return str(data_path.resolve())

        raise ValueError(
            f"Cannot resolve dataset from '{data}'. "
            f"Provide a dataset directory or a YAML file with train/val paths."
        )

    @staticmethod
    def _resolve_yaml_paths(cfg: Dict, base_dir: Path) -> Dict:
        """Resolve relative paths in dataset YAML."""
        cfg = dict(cfg)
        if "valid" in cfg and "val" not in cfg:
            cfg["val"] = cfg.pop("valid")
        cfg.pop("roboflow", None)

        if "path" in cfg:
            path_val = Path(cfg["path"])
            if not path_val.is_absolute():
                cfg["path"] = str((base_dir / path_val).resolve())

        for key in ("train", "val", "test"):
            path = cfg.get(key)
            if path and not Path(path).is_absolute():
                base = Path(cfg["path"]) if "path" in cfg else base_dir
                cfg[key] = str((base / path).resolve())

        return cfg

    def _build_train_kwargs(self, dataset_dir: str) -> Dict:
        """Build kwargs for rfdetr's train() method."""
        ov = self.overrides
        kwargs = {"dataset_dir": dataset_dir}

        param_map = {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "lr": "lr",
            "output_dir": "output_dir",
            "num_workers": "num_workers",
            "seed": "seed",
            "grad_accum_steps": "grad_accum_steps",
        }
        for ov_key, train_key in param_map.items():
            if ov_key in ov:
                kwargs[train_key] = ov[ov_key]

        # Handle resume
        resume = ov.get("resume")
        if resume:
            if isinstance(resume, (str, Path)) and Path(resume).exists():
                kwargs["resume"] = str(resume)
            elif resume is True:
                base = Path(kwargs.get("output_dir", "output"))
                candidates = sorted(base.glob("**/last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    kwargs["resume"] = str(candidates[0])
                else:
                    raise FileNotFoundError(f"Cannot find checkpoint for resume in {base}")

        # Handle img_size -> resolution override
        if "img_size" in ov:
            kwargs["resolution"] = ov["img_size"]

        # Pass through device
        if "device" in ov:
            kwargs["device"] = ov["device"]

        # Auto-increment output_dir
        if "output_dir" not in kwargs:
            kwargs["output_dir"] = self._auto_increment_dir("./runs/detect/train")
        elif not ov.get("resume"):
            kwargs["output_dir"] = self._auto_increment_dir(kwargs["output_dir"])

        return kwargs

    @staticmethod
    def _auto_increment_dir(base: str) -> str:
        base_path = Path(base)
        if not base_path.exists():
            return base
        i = 2
        while True:
            candidate = Path(f"{base}{i}")
            if not candidate.exists():
                return str(candidate)
            i += 1

    def _embed_class_names(self, output_dir: Path):
        """Embed class_names into checkpoint files for later loading."""
        class_names = self.model_wrapper.class_names
        if not class_names or not output_dir.exists():
            return

        for pattern in ["*.pth", "*.ckpt"]:
            for ckpt_path in output_dir.rglob(pattern):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if "class_names" not in ckpt:
                        ckpt["class_names"] = class_names
                        torch.save(ckpt, ckpt_path)
                except Exception:
                    continue

    @staticmethod
    def _extract_class_names(cfg: Dict) -> Optional[List[str]]:
        """Extract class names from dataset config."""
        names = cfg.get("names")
        if names is None:
            return None
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        if isinstance(names, list):
            return list(names)
        return None
