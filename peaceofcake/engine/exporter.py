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
        precision="FLOAT16", compute_units="ALL",
        iou_threshold=0.6, conf_threshold=0.25, **kw,
    ) -> str:
        model, postprocessor = self._get_model_and_postprocessor()

        class CoreMLModel(nn.Module):
            """Output raw_confidence [1, N, C] and raw_coordinates [1, N, 4]
            (normalized cxcywh) for NMS pipeline."""
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
                # Keep batch dim for NMS: [1, N, C] and [1, N, 4]
                return confidence, boxes

        export_model = CoreMLModel(model, postprocessor).eval()

        # Fix project tensor for CoreML linear op
        decoder = export_model.model.decoder.decoder
        if hasattr(decoder, "project") and decoder.project.dim() == 1:
            decoder.project = nn.Parameter(decoder.project.unsqueeze(0), requires_grad=False)

        example = torch.rand(1, 3, img_size, img_size)
        with torch.no_grad():
            _ = export_model(example)
            traced = torch.jit.trace(export_model, example)

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

        # Step 1: Convert the detector model
        detector = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="image", shape=(1, 3, img_size, img_size),
                scale=1.0/255.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB,
            )],
            outputs=[
                ct.TensorType(name="raw_confidence"),
                ct.TensorType(name="raw_coordinates"),
            ],
            minimum_deployment_target=targets.get(min_target, ct.target.iOS17),
            convert_to="mlprogram",
            compute_precision=precisions.get(precision.upper(), ct.precision.FLOAT16),
            compute_units=units_map.get(compute_units.upper(), ct.ComputeUnit.ALL),
        )

        # Step 2: Build NMS model spec
        from peaceofcake.results.detection import COCO_NAMES
        num_classes = len(COCO_NAMES)

        detector_spec = detector.get_spec()
        # Get num_queries from detector output shape
        for out in detector_spec.description.output:
            if out.name == "raw_confidence":
                num_queries = out.type.multiArrayType.shape[1]
                break

        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 4

        # NMS inputs
        for name, shape in [
            ("raw_confidence", (num_queries, num_classes)),
            ("raw_coordinates", (num_queries, 4)),
            ("iouThreshold", (1,)),
            ("confidenceThreshold", (1,)),
        ]:
            inp = nms_spec.description.input.add()
            inp.name = name
            inp.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE
            for s in shape:
                inp.type.multiArrayType.shape.append(s)

        # NMS outputs
        for name, shape in [
            ("confidence", (num_queries, num_classes)),
            ("coordinates", (num_queries, 4)),
        ]:
            out = nms_spec.description.output.add()
            out.name = name
            out.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE
            for s in shape:
                out.type.multiArrayType.shape.append(s)

        # Configure NMS
        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = "raw_confidence"
        nms.coordinatesInputFeatureName = "raw_coordinates"
        nms.confidenceOutputFeatureName = "confidence"
        nms.coordinatesOutputFeatureName = "coordinates"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
        nms.iouThreshold = iou_threshold
        nms.confidenceThreshold = conf_threshold
        nms.pickTop.perClass = True
        for label in COCO_NAMES:
            nms.stringClassLabels.vector.append(label)

        # Step 3: Build pipeline
        # Reshape detector outputs: remove batch dim [1, N, C] -> [N, C]
        # by updating the detector spec output shapes
        for out in detector_spec.description.output:
            arr = out.type.multiArrayType
            if out.name == "raw_confidence":
                arr.shape[:] = [num_queries, num_classes]
            elif out.name == "raw_coordinates":
                arr.shape[:] = [num_queries, 4]

        pipeline_spec = ct.proto.Model_pb2.Model()
        pipeline_spec.specificationVersion = 4
        pipeline_spec.isUpdatable = False

        # Pipeline inputs (image + threshold overrides)
        img_input = pipeline_spec.description.input.add()
        img_input.name = "image"
        img_input.type.imageType.width = img_size
        img_input.type.imageType.height = img_size
        img_input.type.imageType.colorSpace = ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB

        for name, default in [("iouThreshold", iou_threshold), ("confidenceThreshold", conf_threshold)]:
            inp = pipeline_spec.description.input.add()
            inp.name = name
            inp.type.doubleType.SetInParent()

        # Pipeline outputs
        for name, shape in [("confidence", (num_queries, num_classes)), ("coordinates", (num_queries, 4))]:
            out = pipeline_spec.description.output.add()
            out.name = name
            out.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE
            for s in shape:
                out.type.multiArrayType.shape.append(s)

        # Add models to pipeline
        pipeline_spec.pipeline.models.add().CopyFrom(detector_spec)
        pipeline_spec.pipeline.models.add().CopyFrom(nms_spec)

        pipeline_model = ct.models.MLModel(pipeline_spec, weights_dir=detector.weights_dir)

        output = output or "model.mlpackage"
        pipeline_model.save(output)
        print(f"CoreML exported to {output} (with NMS pipeline)")
        return output

    def _export_tensorrt(self, output=None, img_size=640, **kw) -> str:
        import subprocess
        onnx_path = self._export_onnx(output="__temp.onnx", img_size=img_size, **kw)
        output = output or "model.engine"
        subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={output}", "--fp16"], check=True)
        Path(onnx_path).unlink(missing_ok=True)
        print(f"TensorRT exported to {output}")
        return output
