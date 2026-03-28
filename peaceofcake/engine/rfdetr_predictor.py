import copy
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from peaceofcake.results.detection import DetectionResults, COCO_NAMES


class RFDETRPredictor:
    """Runs RF-DETR inference with export-mode model."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self._deploy_model = None
        self._device = None

    def __call__(
        self,
        source: Union[str, List[str], Image.Image, np.ndarray],
        conf: float = 0.25,
        device: Optional[str] = None,
        img_size: Optional[int] = None,
        **kwargs,
    ) -> List[DetectionResults]:
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        resolution = img_size or self.model_wrapper._model_resolution
        if self._deploy_model is None or self._device != device:
            self._build(device)

        images, orig_sizes = self._load_images(source)
        transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        results = []
        for img, (w, h) in zip(images, orig_sizes):
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                labels, boxes, scores = self._deploy_model(img_t, torch.tensor([[h, w]], device=device))

            mask = scores[0] > conf
            names = self.model_wrapper.class_names or COCO_NAMES
            results.append(DetectionResults(
                boxes=boxes[0][mask].cpu(),
                labels=labels[0][mask].long().cpu(),
                scores=scores[0][mask].cpu(),
                orig_img=img,
                names=names,
            ))
        return results

    def _build(self, device: torch.device):
        model_copy = copy.deepcopy(self.model_wrapper.model)
        postprocessor = copy.deepcopy(self.model_wrapper._postprocessor)

        class _Deploy(nn.Module):
            def __init__(self, m, pp):
                super().__init__()
                self.model = m
                self.model.eval()
                self.model.export()
                self.postprocessor = pp

            def forward(self, images, target_sizes):
                pred_boxes, pred_logits = self.model(images)
                outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                results = self.postprocessor(outputs, target_sizes=target_sizes)
                # PostProcess returns list of dicts; unpack to tensors
                labels = torch.stack([r["labels"] for r in results])
                boxes = torch.stack([r["boxes"] for r in results])
                scores = torch.stack([r["scores"] for r in results])
                return labels, boxes, scores

        self._deploy_model = _Deploy(model_copy, postprocessor).to(device).eval()
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
