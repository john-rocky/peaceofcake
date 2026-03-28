__version__ = "0.2.0"

from peaceofcake.models.dfine import DFINE

__all__ = ["DFINE", "RFDETR", "__version__"]


def __getattr__(name):
    """Lazy import RFDETR to avoid loading rfdetr/transformers at startup."""
    if name == "RFDETR":
        from peaceofcake.models.rfdetr import RFDETR
        globals()["RFDETR"] = RFDETR
        return RFDETR
    raise AttributeError(f"module 'peaceofcake' has no attribute {name}")
