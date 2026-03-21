"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
CoreML export script for D-FINE models.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import YAMLConfig


class CoreMLModel(nn.Module):
    """Wrapper that includes postprocessing with hard-coded orig_target_sizes.

    Reimplements postprocessor logic to ensure gather indices are int32,
    which is required by CoreML's gather_along_axis op.
    """

    def __init__(self, model, postprocessor, input_size=640):
        super().__init__()
        self.model = model.deploy()
        postprocessor.deploy()
        self.num_classes = postprocessor.num_classes
        self.num_top_queries = postprocessor.num_top_queries
        self.use_focal_loss = postprocessor.use_focal_loss
        self.register_buffer(
            "orig_target_sizes",
            torch.tensor([[input_size, input_size]], dtype=torch.float32),
        )

    def forward(self, images):
        outputs = self.model(images)
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]

        # box_convert cxcywh -> xyxy (inline to avoid torchvision tracing issues)
        cx, cy, w, h = boxes.unbind(-1)
        bbox_pred = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
        bbox_pred = bbox_pred * self.orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # mod(index, num_classes) - avoid Python % which may not trace cleanly
            labels = index - index // self.num_classes * self.num_classes
            index = index // self.num_classes
            # CoreML gather requires int32 indices
            gather_idx = index.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]).to(torch.int32)
            boxes = bbox_pred.gather(dim=1, index=gather_idx)
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                gather_idx_1d = index.to(torch.int32)
                labels = torch.gather(labels, dim=1, index=gather_idx_1d)
                gather_idx = index.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]).to(torch.int32)
                boxes = torch.gather(bbox_pred, dim=1, index=gather_idx)

        return labels, boxes, scores


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        cfg.model.load_state_dict(state)
    else:
        print("Not loading model.state_dict, using default init state dict...")

    model = CoreMLModel(cfg.model, cfg.postprocessor, input_size=args.input_size)
    model.eval()

    # Fix: CoreML's linear op requires 2D weight. The decoder's project tensor
    # (from weighting_function) is 1D, which F.linear accepts in PyTorch but
    # CoreML does not. Reshape it to 2D.
    decoder = model.model.decoder.decoder
    if hasattr(decoder, "project") and decoder.project.dim() == 1:
        decoder.project = nn.Parameter(decoder.project.unsqueeze(0), requires_grad=False)

    # Warm-up forward pass
    example_input = torch.rand(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        _ = model(example_input)

    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    import coremltools as ct

    target_map = {
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    min_target = target_map.get(args.min_target)
    if min_target is None:
        raise ValueError(
            f"Unsupported deployment target: {args.min_target}. "
            f"Choose from: {list(target_map.keys())}. "
            f"iOS16 is the minimum for grid_sample/resample op support."
        )

    precision_map = {
        "FLOAT16": ct.precision.FLOAT16,
        "FLOAT32": ct.precision.FLOAT32,
    }
    compute_precision = precision_map.get(args.precision.upper())
    if compute_precision is None:
        raise ValueError(f"Unsupported precision: {args.precision}. Choose from: FLOAT16, FLOAT32")

    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    compute_units = compute_units_map.get(args.compute_units.upper())
    if compute_units is None:
        raise ValueError(f"Unsupported compute units: {args.compute_units}")

    print("Converting to CoreML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, args.input_size, args.input_size),
                scale=1.0 / 255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="labels"),
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
        ],
        minimum_deployment_target=min_target,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        compute_units=compute_units,
    )

    coreml_model.author = "D-FINE"
    coreml_model.short_description = "D-FINE real-time object detection model"
    coreml_model.version = "1.0"
    coreml_model.output_description["labels"] = (
        "Predicted class labels, shape [1, num_top_queries]"
    )
    coreml_model.output_description["boxes"] = (
        "Predicted bounding boxes in xyxy format, shape [1, num_top_queries, 4]"
    )
    coreml_model.output_description["scores"] = (
        "Confidence scores, shape [1, num_top_queries]"
    )

    output_file = args.output
    if output_file is None:
        if args.resume:
            output_file = args.resume.replace(".pth", ".mlpackage")
        else:
            output_file = "model.mlpackage"

    coreml_model.save(output_file)
    print(f"CoreML model saved to {output_file}")

    if args.verify:
        print("Verifying CoreML model...")
        from PIL import Image as PILImage

        loaded_model = ct.models.MLModel(output_file)
        dummy_image = PILImage.new("RGB", (args.input_size, args.input_size))
        prediction = loaded_model.predict({"image": dummy_image})
        print("Output keys:", list(prediction.keys()))
        for k, v in prediction.items():
            arr = np.array(v)
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        print("Verification passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export D-FINE model to CoreML format")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/dfine/dfine_hgnetv2_l_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path (default: derived from --resume with .mlpackage extension)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--compute-units",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
        help="CoreML compute units (default: ALL)",
    )
    parser.add_argument(
        "--min-target",
        type=str,
        default="iOS17",
        choices=["iOS16", "iOS17", "iOS18"],
        help="Minimum deployment target (default: iOS17)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="FLOAT16",
        choices=["FLOAT16", "FLOAT32"],
        help="Compute precision (default: FLOAT16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify the exported CoreML model with a dummy input",
    )
    args = parser.parse_args()
    main(args)
