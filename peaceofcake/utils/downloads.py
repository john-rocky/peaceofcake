import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("POC_CACHE", Path.home() / ".cache" / "peaceofcake" / "weights"))


def download_pretrained(url: str, filename: str = None) -> str:
    """Download pretrained weights to cache dir. Returns local path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = filename or url.split("/")[-1]
    local_path = CACHE_DIR / filename

    if local_path.exists():
        print(f"Using cached weights: {local_path}")
        return str(local_path)

    print(f"Downloading {url} ...")
    import urllib.request
    urllib.request.urlretrieve(url, str(local_path))
    print(f"Saved to {local_path}")
    return str(local_path)
