import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFINEExporter:
    """Export D-FINE model to ONNX, CoreML, or TensorRT."""

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

    def _get_model_and_postprocessor(self):
        model = copy.deepcopy(self.model_wrapper.model)
        return model, self.model_wrapper.cfg_obj.postprocessor

    def _export_onnx(self, output=None, img_size=640, simplify=True, opset=16, **kw) -> str:
        model, postprocessor = self._get_model_and_postprocessor()

        class OnnxModel(nn.Module):
            def __init__(self, m, pp):
                super().__init__()
                self.model = m.deploy()
                self.postprocessor = pp.deploy()

            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        export_model = OnnxModel(model, postprocessor).eval()
        output = output or "model.onnx"
        data = torch.rand(1, 3, img_size, img_size)
        size = torch.tensor([[img_size, img_size]])

        torch.onnx.export(
            export_model, (data, size), output,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
            opset_version=opset,
            do_constant_folding=True,
        )

        if simplify:
            try:
                import onnx, onnxsim
                m = onnx.load(output)
                m_sim, ok = onnxsim.simplify(output)
                if ok:
                    onnx.save(m_sim, output)
            except ImportError:
                pass

        print(f"ONNX exported to {output}")
        return output

    def _export_coreml(
        self, output=None, img_size=640, min_target="iOS17",
        precision="FLOAT32", compute_units="ALL", **kw,
    ) -> str:
        model, postprocessor = self._get_model_and_postprocessor()

        class CoreMLModel(nn.Module):
            """Output confidence [N, C] and coordinates [N, 4] (normalized cxcywh)."""
            def __init__(self, m, pp):
                super().__init__()
                self.model = m.deploy()
                pp.deploy()
                self.use_focal_loss = pp.use_focal_loss

            def forward(self, images):
                outputs = self.model(images)
                logits = outputs["pred_logits"]
                boxes = outputs["pred_boxes"]  # [1, N, 4] cxcywh normalized
                if self.use_focal_loss:
                    confidence = F.sigmoid(logits)
                else:
                    confidence = F.softmax(logits, dim=-1)[:, :, :-1]
                return confidence.squeeze(0), boxes.squeeze(0)

        export_model = CoreMLModel(model, postprocessor).eval()

        # Fix project tensor for CoreML linear op
        decoder = export_model.model.decoder.decoder
        if hasattr(decoder, "project") and decoder.project.dim() == 1:
            decoder.project = nn.Parameter(decoder.project.unsqueeze(0), requires_grad=False)

        example = torch.rand(1, 3, img_size, img_size)
        with torch.no_grad():
            eager_conf, eager_boxes = export_model(example)
            traced = torch.jit.trace(export_model, example)
            traced_conf, traced_boxes = traced(example)

        # Verify tracing correctness
        conf_diff = (eager_conf - traced_conf).abs().max().item()
        box_diff = (eager_boxes - traced_boxes).abs().max().item()
        if conf_diff > 1e-4 or box_diff > 1e-4:
            print(f"WARNING: Tracing divergence detected (conf={conf_diff:.6f}, box={box_diff:.6f})")

        try:
            import coremltools as ct
        except ImportError:
            import subprocess, sys
            print("Installing coremltools...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools>=7.0"])
            import coremltools as ct

        targets = {"iOS16": ct.target.iOS16, "iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18}
        precisions = {"FLOAT16": ct.precision.FLOAT16, "FLOAT32": ct.precision.FLOAT32}
        units_map = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
                     "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY}

        coreml_model = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="image", shape=(1, 3, img_size, img_size),
                scale=1.0/255.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB,
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

    def _export_tensorrt(self, output=None, img_size=640, **kw) -> str:
        import subprocess
        onnx_path = self._export_onnx(output="__temp.onnx", img_size=img_size, **kw)
        output = output or "model.engine"
        subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={output}", "--fp16"], check=True)
        Path(onnx_path).unlink(missing_ok=True)
        print(f"TensorRT exported to {output}")
        return output
