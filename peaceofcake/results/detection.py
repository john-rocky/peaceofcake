from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from PIL import Image


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class DetectionResults:
    """Detection results for a single image."""
    boxes: torch.Tensor         # (N, 4) xyxy in original image coordinates
    labels: torch.Tensor        # (N,) class indices
    scores: torch.Tensor        # (N,) confidence scores
    orig_img: Image.Image = field(repr=False)
    names: Optional[List[str]] = field(default=None, repr=False)

    def __len__(self):
        return len(self.scores)

    def __repr__(self):
        if len(self) == 0:
            return "DetectionResults(0 detections)"
        names = self.names or COCO_NAMES
        items = []
        for i in range(min(5, len(self))):
            idx = int(self.labels[i])
            name = names[idx] if idx < len(names) else f"class_{idx}"
            items.append(f"{name} {self.scores[i]:.2f}")
        suffix = f" +{len(self) - 5} more" if len(self) > 5 else ""
        return f"DetectionResults({len(self)} detections: {', '.join(items)}{suffix})"

    def plot(self, line_width: int = 2, font_size: int = 12) -> Image.Image:
        """Draw bounding boxes on the image. Returns a new PIL Image."""
        from peaceofcake.utils.plotting import draw_detections
        return draw_detections(
            self.orig_img, self.boxes, self.labels, self.scores,
            names=self.names, line_width=line_width, font_size=font_size,
        )

    def save(self, path: str, **kwargs):
        """Save plotted image to file."""
        self.plot(**kwargs).save(path)
        print(f"Saved to {path}")
