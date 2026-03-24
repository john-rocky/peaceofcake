from pathlib import Path


def _find_dfine_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "third_party" / "dfine"


def _cfg(rel: str) -> str:
    return str(_find_dfine_root() / rel)


_BASE_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

DFINE_SIZES = {
    "n": {"config": "configs/dfine/dfine_hgnetv2_n_coco.yml", "epochs": 160, "batch_size": 16},
    "s": {"config": "configs/dfine/dfine_hgnetv2_s_coco.yml", "epochs": 132, "batch_size": 8},
    "m": {"config": "configs/dfine/dfine_hgnetv2_m_coco.yml", "epochs": 132, "batch_size": 4},
    "l": {"config": "configs/dfine/dfine_hgnetv2_l_coco.yml", "epochs": 80, "batch_size": 4},
    "x": {"config": "configs/dfine/dfine_hgnetv2_x_coco.yml", "epochs": 80, "batch_size": 4},
}

DFINE_MODEL_REGISTRY = {
    # Pretrained on COCO
    "dfine-n-coco": {"size": "n", "url": f"{_BASE_URL}/dfine_n_coco.pth", "filename": "dfine_n_coco.pth"},
    "dfine-s-coco": {"size": "s", "url": f"{_BASE_URL}/dfine_s_coco.pth", "filename": "dfine_s_coco.pth"},
    "dfine-m-coco": {"size": "m", "url": f"{_BASE_URL}/dfine_m_coco.pth", "filename": "dfine_m_coco.pth"},
    "dfine-l-coco": {"size": "l", "url": f"{_BASE_URL}/dfine_l_coco.pth", "filename": "dfine_l_coco.pth"},
    "dfine-x-coco": {"size": "x", "url": f"{_BASE_URL}/dfine_x_coco.pth", "filename": "dfine_x_coco.pth"},
    # Objects365+COCO
    "dfine-s-obj2coco": {"size": "s", "url": f"{_BASE_URL}/dfine_s_obj2coco.pth", "filename": "dfine_s_obj2coco.pth"},
    "dfine-m-obj2coco": {"size": "m", "url": f"{_BASE_URL}/dfine_m_obj2coco.pth", "filename": "dfine_m_obj2coco.pth"},
    "dfine-l-obj2coco": {"size": "l", "url": f"{_BASE_URL}/dfine_l_obj2coco_e25.pth", "filename": "dfine_l_obj2coco_e25.pth"},
    "dfine-x-obj2coco": {"size": "x", "url": f"{_BASE_URL}/dfine_x_obj2coco.pth", "filename": "dfine_x_obj2coco.pth"},
    # Size-only (no pretrained weights)
    "dfine-n": {"size": "n", "url": None},
    "dfine-s": {"size": "s", "url": None},
    "dfine-m": {"size": "m", "url": None},
    "dfine-l": {"size": "l", "url": None},
    "dfine-x": {"size": "x", "url": None},
}


def get_dfine_config_path(size: str) -> str:
    return _cfg(DFINE_SIZES[size]["config"])


def get_dfine_root() -> Path:
    return _find_dfine_root()
