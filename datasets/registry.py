from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

DATASET_REGISTRY = {
    "m3fd": {
        "raw_root": "data/raw/M3FD",
        "image_subdir": "Ir",
        "extensions": DEFAULT_IMAGE_EXTENSIONS,
    },
    "kaist": {
        "raw_root": "data/raw/KAIST",
        "image_subdir": "",
        "extensions": DEFAULT_IMAGE_EXTENSIONS,
    },
}


def get_dataset_config(dataset_name: Optional[str]) -> Dict[str, object]:
    if not dataset_name:
        return {}
    key = dataset_name.lower()
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset_name: {dataset_name}. Known datasets: {known}")
    return dict(DATASET_REGISTRY[key])


def resolve_dataset_raw_root(dataset_name: Optional[str], project_root: Path) -> Optional[Path]:
    cfg = get_dataset_config(dataset_name)
    raw_root = cfg.get("raw_root")
    if not raw_root:
        return None
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = project_root / root
    return root.resolve()


def resolve_image_subdir(dataset_name: Optional[str]) -> str:
    cfg = get_dataset_config(dataset_name)
    return str(cfg.get("image_subdir", ""))


def resolve_extensions(dataset_name: Optional[str]) -> List[str]:
    cfg = get_dataset_config(dataset_name)
    extensions = cfg.get("extensions", DEFAULT_IMAGE_EXTENSIONS)
    return [str(ext).lower() for ext in extensions]
