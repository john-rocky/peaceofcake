from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseModel(ABC):
    """Base model class. Subclasses implement task_map and _setup."""

    def __init__(self, model_name_or_path: str = "", task: str = "detect"):
        self.task = task
        self.model = None
        self.cfg_obj = None
        self.ckpt_path = None
        self.overrides = {}
        self.class_names = None
        self._predictor = None
        self._setup(model_name_or_path)

    @property
    @abstractmethod
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def _setup(self, model_name_or_path: str):
        ...

    def train(self, **kwargs):
        """Train the model."""
        trainer_cls = self.task_map[self.task]["trainer"]
        trainer = trainer_cls(self, kwargs)
        trainer.train()
        self.model = trainer.best_model or self.model
        return trainer.metrics

    def predict(
        self,
        source,
        conf: float = 0.25,
        device: Optional[str] = None,
        **kwargs,
    ) -> list:
        """Run inference."""
        predictor_cls = self.task_map[self.task]["predictor"]
        if self._predictor is None:
            self._predictor = predictor_cls(self)
        return self._predictor(source, conf=conf, device=device, **kwargs)

    def val(self, **kwargs) -> Dict:
        """Validate on a dataset."""
        validator_cls = self.task_map[self.task]["validator"]
        return validator_cls(self, kwargs).validate()

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export to format. Returns path to exported file."""
        exporter_cls = self.task_map[self.task]["exporter"]
        return exporter_cls(self, kwargs).export(format, **kwargs)

    def info(self):
        """Print model information."""
        if self.model is None:
            print("Model not loaded")
            return
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {self.__class__.__name__}")
        print(f"Parameters: {n_params:,}")
        print(f"Task: {self.task}")
