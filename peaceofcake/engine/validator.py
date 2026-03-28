from pathlib import Path
from typing import Any, Dict


class DFINEValidator:
    """Wraps D-FINE evaluation for simple model.val() API."""

    def __init__(self, model_wrapper, overrides: Dict[str, Any] = None):
        self.model_wrapper = model_wrapper
        self.overrides = overrides or {}

    def validate(self) -> Dict:
        from src.core import YAMLConfig
        from src.solver import TASKS
        from src.misc import dist_utils
        from src import data  # noqa: F401
        from src.nn import criterion  # noqa: F401
        from src.zoo.dfine import _register_training_modules
        _register_training_modules()

        data_cfg = self._parse_data(self.overrides.get("data"))
        yaml_overrides = self._build_overrides(data_cfg)

        cfg = YAMLConfig(self.model_wrapper._dfine_config_path, **yaml_overrides)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Load current model weights
        if self.model_wrapper.ckpt_path:
            cfg.tuning = self.model_wrapper.ckpt_path

        dist_utils.setup_distributed(
            print_rank=0,
            print_method="builtin",
            seed=self.overrides.get("seed"),
        )

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.val()

        dist_utils.cleanup()

        # Extract mAP from COCO evaluator if available
        results = {}
        if hasattr(solver, "evaluator") and solver.evaluator is not None:
            coco_eval = solver.evaluator
            if hasattr(coco_eval, "coco_eval") and "bbox" in coco_eval.coco_eval:
                stats = coco_eval.coco_eval["bbox"].stats
                results = {
                    "mAP50-95": float(stats[0]),
                    "mAP50": float(stats[1]),
                    "mAP75": float(stats[2]),
                    "mAP_small": float(stats[3]),
                    "mAP_medium": float(stats[4]),
                    "mAP_large": float(stats[5]),
                }
        return results

    def _parse_data(self, data_arg) -> Dict:
        if data_arg is None:
            return {}
        # Reuse trainer's data parsing logic
        from peaceofcake.engine.trainer import DFINETrainer
        trainer = DFINETrainer(self.model_wrapper)
        return trainer._parse_data(data_arg)

    def _build_overrides(self, data_cfg: Dict) -> Dict:
        ov = self.overrides
        result = {}
        result["output_dir"] = ov.get("output_dir", "./runs/detect/val")

        if "batch_size" in ov:
            result.setdefault("val_dataloader", {})["total_batch_size"] = ov["batch_size"]

        if "num_workers" in ov:
            result.setdefault("val_dataloader", {})["num_workers"] = ov["num_workers"]

        if "img_size" in ov:
            sz = ov["img_size"]
            result["eval_spatial_size"] = [sz, sz]

        # Merge dataset config
        for k, v in data_cfg.items():
            if k == "class_names":
                continue
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k].update(v)
            else:
                result[k] = v

        return result
