from pathlib import Path
from typing import Any, Dict

import torch

from peaceofcake.engine.model import BaseModel
from peaceofcake.cfg.defaults import RFDETR_MODEL_REGISTRY, RFDETR_SIZES


class RFDETR(BaseModel):
    """RF-DETR object detection model.

    Usage:
        model = RFDETR("rfdetr-l-coco")          # pretrained
        model = RFDETR("path/to/weights.pth")     # local checkpoint
        model = RFDETR("rfdetr-n")                # random init
    """

    def __init__(self, model_name_or_path: str = "rfdetr-l-coco"):
        self._model_size = None
        self._model_resolution = None
        self._rfdetr_obj = None
        self._postprocessor = None
        self._num_classes = None
        super().__init__(model_name_or_path, task="detect")

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        from peaceofcake.engine.rfdetr_predictor import RFDETRPredictor
        from peaceofcake.engine.rfdetr_exporter import RFDETRExporter
        from peaceofcake.engine.rfdetr_trainer import RFDETRTrainer
        from peaceofcake.engine.rfdetr_validator import RFDETRValidator

        return {
            "detect": {
                "trainer": RFDETRTrainer,
                "predictor": RFDETRPredictor,
                "exporter": RFDETRExporter,
                "validator": RFDETRValidator,
            }
        }

    def _setup(self, model_name_or_path: str):
        self._ensure_rfdetr_importable()

        if model_name_or_path in RFDETR_MODEL_REGISTRY:
            entry = RFDETR_MODEL_REGISTRY[model_name_or_path]
            self._model_size = entry["size"]
            self._model_resolution = RFDETR_SIZES[entry["size"]]["resolution"]
            variant_cls = self._get_variant_class(entry["size"])

            if entry.get("pretrained", True):
                self._rfdetr_obj = variant_cls()
                self.ckpt_path = model_name_or_path
            else:
                self._rfdetr_obj = variant_cls(pretrain_weights=None)

            self._sync_from_rfdetr()

        elif Path(model_name_or_path).exists() and model_name_or_path.endswith(".pth"):
            self.ckpt_path = model_name_or_path
            self._model_size = self._detect_size(model_name_or_path)
            self._model_resolution = RFDETR_SIZES[self._model_size]["resolution"]
            variant_cls = self._get_variant_class(self._model_size)

            nc = self._detect_num_classes(model_name_or_path)
            kwargs = {"pretrain_weights": model_name_or_path}
            if nc is not None:
                kwargs["num_classes"] = nc
            self._rfdetr_obj = variant_cls(**kwargs)
            self._sync_from_rfdetr()

        else:
            available = list(RFDETR_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Cannot resolve '{model_name_or_path}'. "
                f"Expected one of {available} or a path to a .pth file."
            )

        print(f"RF-DETR-{self._model_size.upper()} loaded"
              f"{' (pretrained)' if self.ckpt_path else ' (random init)'}"
              f" [resolution={self._model_resolution}]")

    def _sync_from_rfdetr(self):
        """Sync nn.Module and metadata from rfdetr object."""
        ctx = self._rfdetr_obj.model
        self.model = ctx.model
        self._postprocessor = ctx.postprocess
        self._num_classes = ctx.args.num_classes
        if ctx.class_names:
            self.class_names = ctx.class_names

    @staticmethod
    def _ensure_rfdetr_importable():
        try:
            import rfdetr  # noqa: F401
        except ImportError:
            raise ImportError(
                "RF-DETR is not installed. Install it with:\n"
                "  pip install rfdetr\n"
                "For training support:\n"
                "  pip install 'rfdetr[train]'"
            )

    @staticmethod
    def _get_variant_class(size: str):
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        return {"n": RFDETRNano, "s": RFDETRSmall, "m": RFDETRMedium, "l": RFDETRLarge}[size]

    @staticmethod
    def _detect_num_classes(path: str):
        """Infer num_classes from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        key = "class_embed.weight"
        if key in state:
            return state[key].shape[0] - 1  # RF-DETR uses num_classes + 1
        return None

    def _detect_size(self, path: str) -> str:
        stem = Path(path).stem.lower()
        for name, letter in [("nano", "n"), ("small", "s"), ("medium", "m"), ("large", "l")]:
            if name in stem:
                return letter
        for s in ["n", "s", "m", "l"]:
            if f"-{s}-" in stem or f"_{s}_" in stem or stem.endswith(f"-{s}") or stem.endswith(f"_{s}"):
                return s
        # Fallback: param count
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        n = sum(v.numel() for v in model_state.values())
        if n < 15_000_000:
            return "n"
        elif n < 30_000_000:
            return "s"
        elif n < 50_000_000:
            return "m"
        else:
            return "l"
