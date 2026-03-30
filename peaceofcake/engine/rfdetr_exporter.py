import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_rfdetr_coreml_patches():
    """Apply runtime patches to coremltools and rfdetr for CoreML conversion.

    The DINOv2 backbone and deformable attention in RF-DETR produce traced ops
    that coremltools cannot convert. This function patches:
    1. torch_int – force Python int so trace bakes in constants
    2. DINOv2 channel check – remove traced shape assertion
    3. Deformable attention – rank-5 only (CoreML limit), no dynamic splits
    4. coremltools _cast – handle multi-element const arrays
    5. coremltools tensor_assign – relax shape check
    6. coremltools meshgrid – allow non-1D inputs
    7. coremltools split – handle list inputs
    """
    import transformers.utils
    transformers.utils.torch_int = lambda x: int(x)

    import rfdetr.models.backbone.dinov2_with_windowed_attn as dwv
    dwv.torch_int = lambda x: int(x)
    dwv.Dinov2WithRegistersPatchEmbeddings.forward = (
        lambda self, pv: self.projection(pv).flatten(2).transpose(1, 2)
    )

    # Deformable attention: avoid rank-6 tensors and dynamic splits
    import rfdetr.models.ops.modules.ms_deform_attn as mda
    from rfdetr.utilities.tensors import _bilinear_grid_sample

    def _patched_deform_forward(
        self, query, reference_points, input_flatten, input_spatial_shapes,
        input_level_start_index, input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        nh, nl, np_ = self.n_heads, self.n_levels, self.n_points
        hd = self.d_model // nh

        offsets = self.sampling_offsets(query).view(N, Len_q, nh, nl * np_, 2)
        attn_w = self.attention_weights(query).view(N, Len_q, nh, nl * np_)

        if reference_points.shape[-1] == 2:
            norm = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            norm = norm.repeat_interleave(np_, dim=0)
            ref = reference_points.repeat_interleave(np_, dim=2)
            sloc = ref[:, :, None, :, :] + offsets / norm[None, None, None, :, :]
        elif reference_points.shape[-1] == 4:
            rxy = reference_points[:, :, :, :2].repeat_interleave(np_, dim=2)
            rwh = reference_points[:, :, :, 2:].repeat_interleave(np_, dim=2)
            sloc = rxy[:, :, None, :, :] + offsets / np_ * rwh[:, :, None, :, :] * 0.5
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        attn_w = F.softmax(attn_w, -1)
        value = value.transpose(1, 2).contiguous().view(N, nh, hd, Len_in)

        sg = 2 * sloc - 1
        svl = []
        offset = 0
        for lid_ in range(nl):
            H = int(input_spatial_shapes[lid_, 0].item())
            W = int(input_spatial_shapes[lid_, 1].item())
            vl = value[:, :, :, offset:offset + H * W].reshape(N * nh, hd, H, W)
            offset += H * W
            gl = sg[:, :, :, lid_ * np_:(lid_ + 1) * np_, :]
            gl = gl.permute(0, 2, 1, 3, 4).reshape(N * nh, Len_q, np_, 2)
            svl.append(_bilinear_grid_sample(vl, gl, padding_mode="zeros", align_corners=False))

        attn_w = attn_w.permute(0, 2, 1, 3).reshape(N * nh, 1, Len_q, nl * np_)
        out = (torch.stack(svl, dim=-2).flatten(-2) * attn_w).sum(-1)
        out = out.reshape(N, nh * hd, Len_q).permute(0, 2, 1).contiguous()
        return self.output_proj(out)

    mda.MSDeformAttn.forward = _patched_deform_forward

    # Apply shared coremltools patches (_cast, tensor_assign, meshgrid, split)
    from peaceofcake.engine.coreml_patches import apply_coreml_patches
    apply_coreml_patches()


class RFDETRExporter:
    """Export RF-DETR model to ONNX, CoreML, or TensorRT."""

    def __init__(self, model_wrapper, overrides=None):
        self.model_wrapper = model_wrapper
        self.overrides = overrides or {}

    def export(self, format: str = "onnx", **kwargs) -> str:
        format = format.lower().strip()
        exporters = {
            "onnx": self._export_onnx,
            "coreml": self._export_coreml,
            "tensorrt": self._export_tensorrt,
            "trt": self._export_tensorrt,
        }
        if format not in exporters:
            raise ValueError(f"Unsupported format '{format}'. Choose from: {list(exporters.keys())}")
        return exporters[format](**{**self.overrides, **kwargs})

    def _get_resolution(self, img_size=None):
        return img_size or self.model_wrapper._model_resolution

    def _export_onnx(self, output=None, img_size=None, simplify=True, opset=17, **kw) -> str:
        resolution = self._get_resolution(img_size)
        model = copy.deepcopy(self.model_wrapper.model).cpu()
        model.eval()
        model.export()

        output = output or "model.onnx"
        data = torch.rand(1, 3, resolution, resolution)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        data = (data - mean) / std

        torch.onnx.export(
            model, (data,), output,
            input_names=["images"],
            output_names=["pred_boxes", "pred_logits"],
            dynamic_axes={"images": {0: "N"}},
            opset_version=opset,
            do_constant_folding=True,
        )

        if simplify:
            try:
                import onnx, onnxsim
                m = onnx.load(output)
                m_sim, ok = onnxsim.simplify(m)
                if ok:
                    onnx.save(m_sim, output)
            except ImportError:
                pass

        print(f"ONNX exported to {output}")
        return output

    def _export_coreml(
        self, output=None, img_size=None, min_target="iOS17",
        precision="FLOAT32", compute_units="ALL", **kw,
    ) -> str:
        """Export to CoreML via torch.jit.trace with runtime patches."""
        _apply_rfdetr_coreml_patches()

        resolution = self._get_resolution(img_size)
        model = copy.deepcopy(self.model_wrapper.model).cpu()
        model.eval()
        model.export()

        class _CoreMLWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, images):
                x = (images - self.mean) / self.std
                pred_boxes, pred_logits = self.model(x)
                return torch.sigmoid(pred_logits).squeeze(0), pred_boxes.squeeze(0)

        wrapper = _CoreMLWrapper(model).cpu().eval()
        example = torch.rand(1, 3, resolution, resolution)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example)

        import coremltools as ct

        targets = {"iOS16": ct.target.iOS16, "iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18}
        precisions = {"FLOAT16": ct.precision.FLOAT16, "FLOAT32": ct.precision.FLOAT32}
        units_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }

        coreml_model = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="image", shape=(1, 3, resolution, resolution),
                scale=1.0 / 255.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB,
            )],
            outputs=[
                ct.TensorType(name="confidence"),
                ct.TensorType(name="coordinates"),
            ],
            minimum_deployment_target=targets.get(min_target, ct.target.iOS17),
            convert_to="mlprogram",
            compute_precision=precisions.get(precision.upper(), ct.precision.FLOAT32),
            compute_units=units_map.get(compute_units.upper(), ct.ComputeUnit.ALL),
        )

        output = output or "model.mlpackage"
        coreml_model.save(output)
        print(f"CoreML exported to {output} (precision={precision})")
        return output

    def _export_tensorrt(self, output=None, img_size=None, **kw) -> str:
        import subprocess
        onnx_path = self._export_onnx(output="__temp.onnx", img_size=img_size, **kw)
        output = output or "model.engine"
        subprocess.run(
            ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={output}", "--fp16"],
            check=True,
        )
        Path(onnx_path).unlink(missing_ok=True)
        print(f"TensorRT exported to {output}")
        return output
