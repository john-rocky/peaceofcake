import sys
from pathlib import Path
from typing import Any, Dict

import torch

from peaceofcake.engine.model import BaseModel
from peaceofcake.cfg.defaults import (
    DFINE_MODEL_REGISTRY,
    DFINE_SIZES,
    get_dfine_config_path,
    get_dfine_root,
)
from peaceofcake.utils.downloads import download_pretrained


class DFINE(BaseModel):
    """D-FINE object detection model.

    Usage:
        model = DFINE("dfine-l-coco")         # pretrained
        model = DFINE("path/to/weights.pth")   # local checkpoint
        model = DFINE("dfine-n")               # random init
    """

    def __init__(self, model_name_or_path: str = "dfine-l-coco"):
        self._model_size = None
        self._dfine_config_path = None
        super().__init__(model_name_or_path, task="detect")

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        from peaceofcake.engine.trainer import DFINETrainer
        from peaceofcake.engine.predictor import DFINEPredictor
        from peaceofcake.engine.exporter import DFINEExporter
        from peaceofcake.engine.validator import DFINEValidator

        return {
            "detect": {
                "trainer": DFINETrainer,
                "predictor": DFINEPredictor,
                "exporter": DFINEExporter,
                "validator": DFINEValidator,
            }
        }

    def _setup(self, model_name_or_path: str):
        self._ensure_dfine_importable()

        if model_name_or_path in DFINE_MODEL_REGISTRY:
            entry = DFINE_MODEL_REGISTRY[model_name_or_path]
            self._model_size = entry["size"]
            self._dfine_config_path = get_dfine_config_path(entry["size"])

            if entry.get("url"):
                self.ckpt_path = download_pretrained(entry["url"], entry.get("filename"))
                self._load_model(self.ckpt_path)
            else:
                self._load_model(None)

        elif Path(model_name_or_path).exists() and model_name_or_path.endswith(".pth"):
            self.ckpt_path = model_name_or_path
            self._model_size = self._detect_size(model_name_or_path)
            self._dfine_config_path = get_dfine_config_path(self._model_size)
            self._load_model(model_name_or_path)

        elif model_name_or_path.endswith(".pth"):
            entry = self._resolve_filename(model_name_or_path)
            if entry:
                self._model_size = entry["size"]
                self._dfine_config_path = get_dfine_config_path(entry["size"])
                self.ckpt_path = download_pretrained(entry["url"], entry.get("filename"))
                self._load_model(self.ckpt_path)
            else:
                raise ValueError(
                    f"'{model_name_or_path}' not found locally and not a known official weight."
                )

        else:
            available = list(DFINE_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Cannot resolve '{model_name_or_path}'. "
                f"Expected one of {available} or a path to a .pth file."
            )

        print(f"D-FINE-{self._model_size.upper()} loaded"
              f"{' (pretrained)' if self.ckpt_path else ' (random init)'}")

    def _ensure_dfine_importable(self):
        if "src" in sys.modules:
            return
        dfine_root = get_dfine_root()
        sys.path.insert(0, str(dfine_root))
        import src  # noqa: F401
        import src.nn  # noqa: F401
        import src.zoo  # noqa: F401

    def _load_model(self, ckpt_path):
        from src.core import YAMLConfig

        overrides = {}

        # Detect num_classes and class_names from checkpoint
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint.get("model", checkpoint)

            nc = self._detect_num_classes(state)
            if nc is not None and nc != 80:
                overrides["num_classes"] = nc

            names = checkpoint.get("class_names")
            if names:
                self.class_names = names

        cfg = YAMLConfig(self._dfine_config_path, **overrides)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        self.model = cfg.model
        self.cfg_obj = cfg

        if ckpt_path:
            self.model.load_state_dict(state, strict=False)

    @staticmethod
    def _detect_num_classes(state_dict: dict) -> int | None:
        """Infer num_classes from checkpoint state dict."""
        key = "decoder.enc_score_head.weight"
        if key in state_dict:
            return state_dict[key].shape[0]
        return None

    @staticmethod
    def _resolve_filename(name: str):
        """Match a .pth filename to a known official model entry."""
        basename = Path(name).name
        for entry in DFINE_MODEL_REGISTRY.values():
            if entry.get("filename") and entry["filename"] == basename and entry.get("url"):
                return entry
        return None

    def _detect_size(self, path: str) -> str:
        stem = Path(path).stem.lower()
        for size in ["n", "s", "m", "l", "x"]:
            if f"_{size}_" in stem or stem.endswith(f"_{size}"):
                return size
        # Fallback: guess from param count
        state = torch.load(path, map_location="cpu")
        model_state = state.get("model", state.get("ema", {}).get("module", state))
        n = sum(v.numel() for v in model_state.values())
        if n < 6_000_000: return "n"
        elif n < 15_000_000: return "s"
        elif n < 25_000_000: return "m"
        elif n < 50_000_000: return "l"
        else: return "x"
