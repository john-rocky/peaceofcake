import copy
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from peaceofcake.results.detection import DetectionResults, COCO_NAMES


class DFINEPredictor:
    """Runs D-FINE inference with deploy-mode model."""

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self._deploy_model = None
        self._device = None

    def __call__(
        self,
        source: Union[str, List[str], Image.Image, np.ndarray],
        conf: float = 0.25,
        device: Optional[str] = None,
        img_size: int = 640,
        **kwargs,
    ) -> List[DetectionResults]:
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if self._deploy_model is None or self._device != device:
            self._build(device)

        images, orig_sizes = self._load_images(source)
        transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

        results = []
        for img, (w, h) in zip(images, orig_sizes):
            img_t = transform(img).unsqueeze(0).to(device)
            size_t = torch.tensor([[w, h]], dtype=torch.float32).to(device)

            with torch.no_grad():
                labels, boxes, scores = self._deploy_model(img_t, size_t)

            mask = scores[0] > conf
            results.append(DetectionResults(
                boxes=boxes[0][mask].cpu(),
                labels=labels[0][mask].long().cpu(),
                scores=scores[0][mask].cpu(),
                orig_img=img,
                names=COCO_NAMES,
            ))
        return results

    def _build(self, device: torch.device):
        model_copy = copy.deepcopy(self.model_wrapper.model)
        cfg = self.model_wrapper.cfg_obj

        class _Deploy(nn.Module):
            def __init__(self, m, pp):
                super().__init__()
                self.model = m.deploy()
                self.postprocessor = pp.deploy()

            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        self._deploy_model = _Deploy(model_copy, cfg.postprocessor).to(device).eval()
        self._device = device

    def _load_images(self, source):
        if isinstance(source, (str, Path)):
            source = [source]
        elif isinstance(source, (Image.Image, np.ndarray)):
            source = [source]

        images, sizes = [], []
        for s in source:
            if isinstance(s, (str, Path)):
                img = Image.open(s).convert("RGB")
            elif isinstance(s, np.ndarray):
                img = Image.fromarray(s).convert("RGB")
            elif isinstance(s, Image.Image):
                img = s.convert("RGB")
            else:
                raise ValueError(f"Unsupported source type: {type(s)}")
            images.append(img)
            sizes.append(img.size)  # (w, h)
        return images, sizes
