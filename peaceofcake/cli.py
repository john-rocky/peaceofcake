"""
CLI entry point for peaceofcake.

Usage:
    poc train  data=dataset.yaml epochs=50 batch_size=16 img_size=640
    poc predict source=image.jpg conf=0.3
    poc export format=coreml img_size=640 precision=FLOAT16
    poc val    data=dataset.yaml batch_size=16
    poc info

    # Specify model (default: dfine-n-coco)
    poc train model=dfine-l-coco data=dataset.yaml epochs=100

    # RF-DETR models
    poc predict model=rfdetr-l-coco source=image.jpg conf=0.3
    poc train model=rfdetr-m-coco data=dataset/ epochs=50
    poc export model=rfdetr-l-coco format=coreml precision=FLOAT16
"""
import sys


def _parse_args(args: list) -> dict:
    """Parse key=value arguments into a dict with type inference."""
    result = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        # Type inference
        if value.lower() in ("true", "yes"):
            value = True
        elif value.lower() in ("false", "no"):
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        result[key] = value
    return result


def _get_model_class(model_name: str):
    """Determine model class from model name."""
    if model_name.startswith("rfdetr"):
        from peaceofcake import RFDETR
        return RFDETR
    from peaceofcake import DFINE
    return DFINE


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return

    command = sys.argv[1]
    kwargs = _parse_args(sys.argv[2:])

    model_name = kwargs.pop("model", "dfine-n-coco")
    ModelClass = _get_model_class(model_name)

    if command == "info":
        model = ModelClass(model_name)
        model.info()

    elif command == "train":
        model = ModelClass(model_name)
        model.train(**kwargs)

    elif command == "val":
        model = ModelClass(model_name)
        results = model.val(**kwargs)
        if results:
            print("\nValidation Results:")
            for k, v in results.items():
                print(f"  {k}: {v:.4f}")

    elif command == "predict":
        source = kwargs.pop("source", None)
        if source is None:
            print("Error: source= is required. Example: poc predict source=image.jpg")
            sys.exit(1)
        output = kwargs.pop("output", None)
        model = ModelClass(model_name)
        results = model.predict(source, **kwargs)
        for i, det in enumerate(results):
            print(det)
            if output:
                det.save(output)

    elif command == "export":
        fmt = kwargs.pop("format", "onnx")
        model = ModelClass(model_name)
        path = model.export(fmt, **kwargs)
        print(f"Exported to {path}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, val, predict, export, info")
        sys.exit(1)


if __name__ == "__main__":
    main()
