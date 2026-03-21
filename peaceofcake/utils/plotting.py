from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from peaceofcake.results.detection import COCO_NAMES

COLORS = [
    (255, 56, 56), (0, 157, 255), (255, 178, 29), (72, 249, 10),
    (180, 0, 255), (255, 0, 128), (0, 255, 183), (255, 244, 79),
    (52, 209, 255), (128, 0, 64),
]


def draw_detections(
    img: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    names: Optional[List[str]] = None,
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    names = names or COCO_NAMES

    try:
        font = ImageFont.truetype("Arial", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].tolist()
        idx = int(labels[i])
        color = COLORS[idx % len(COLORS)]
        name = names[idx] if idx < len(names) else f"class_{idx}"
        text = f"{name} {scores[i]:.0%}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        # Label background
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), text, fill=(255, 255, 255), font=font)

    return img
