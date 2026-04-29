import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.registry import (  # noqa: E402
    DEFAULT_IMAGE_EXTENSIONS,
    resolve_dataset_raw_root,
    resolve_extensions,
    resolve_image_subdir,
)


IMG_EXTENSIONS = set(DEFAULT_IMAGE_EXTENSIONS)


def is_image_file(path: Path, extensions: Optional[Sequence[str]] = None) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("._"):
        return False
    allowed = IMG_EXTENSIONS if extensions is None else {ext.lower() for ext in extensions}
    return path.suffix.lower() in allowed


def find_dataset_images(
    raw_root: Path,
    image_subdir: str = "",
    recursive: bool = False,
    extensions: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Collect dataset images from raw_root or raw_root/image_subdir."""
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_root}")

    image_dir = raw_root / image_subdir if image_subdir else raw_root
    if not image_dir.exists() and image_subdir:
        alt_subdir = image_subdir.swapcase()
        alt_dir = raw_root / alt_subdir
        if alt_dir.exists():
            image_dir = alt_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Cannot find image directory: {image_dir}")

    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    image_paths = [p.resolve() for p in iterator if is_image_file(p, extensions=extensions)]
    image_paths.sort()

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under: {image_dir}")

    return image_paths


def group_images_by_stem(image_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Treat each image stem as one group key.
    This keeps split outputs compatible with train_groups/val_groups/test_groups.
    """
    grouped: Dict[str, List[Path]] = {}
    for p in image_paths:
        grouped[p.stem] = [p]
    return grouped


def split_groups(
    grouped_paths: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path], List[str], List[str], List[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got: {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got: {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(f"train_ratio + val_ratio must be < 1, got: {train_ratio + val_ratio}")

    group_keys = sorted(grouped_paths.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    total_groups = len(group_keys)
    if total_groups == 0:
        return [], [], [], [], [], []

    n_train = int(total_groups * train_ratio)
    n_val = int(total_groups * val_ratio)

    if total_groups >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= total_groups:
            n_val = max(1, total_groups - n_train - 1)
            if n_train + n_val >= total_groups:
                n_train = total_groups - 2
                n_val = 1
    elif total_groups == 2:
        n_train = 1
        n_val = 0
    else:
        n_train = 1
        n_val = 0

    train_groups = group_keys[:n_train]
    val_groups = group_keys[n_train:n_train + n_val]
    test_groups = group_keys[n_train + n_val:]

    train_paths: List[Path] = []
    val_paths: List[Path] = []
    test_paths: List[Path] = []

    for key in train_groups:
        train_paths.extend(grouped_paths[key])
    for key in val_groups:
        val_paths.extend(grouped_paths[key])
    for key in test_groups:
        test_paths.extend(grouped_paths[key])

    return train_paths, val_paths, test_paths, train_groups, val_groups, test_groups


def save_split_file(paths: List[Path], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8", newline="\n") as f:
        for p in paths:
            f.write(str(p).replace("\\", "/") + "\n")


def save_group_file(group_keys: List[str], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8", newline="\n") as f:
        for key in group_keys:
            f.write(key + "\n")


def build_default_raw_root(project_root: Path, dataset_name: str) -> Path:
    resolved = resolve_dataset_raw_root(dataset_name, project_root)
    if resolved is None:
        raise ValueError(f"Cannot resolve raw root for dataset_name: {dataset_name}")
    return resolved


def build_default_processed_root(project_root: Path) -> Path:
    return project_root / "data" / "processed"


def print_split_summary(
    total_images: int,
    total_groups: int,
    train_paths: List[Path],
    val_paths: List[Path],
    test_paths: List[Path],
) -> None:
    print("=" * 60)
    print("Dataset split complete")
    print(f"Total images : {total_images}")
    print(f"Total groups : {total_groups}")
    print(f"Train images : {len(train_paths)}")
    print(f"Val images   : {len(val_paths)}")
    print(f"Test images  : {len(test_paths)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image datasets and generate split files."
    )
    parser.add_argument(
        "--dataset_name",
        "--dataset-name",
        type=str,
        default="m3fd",
        help="Registered dataset name used for defaults. Default: m3fd (compatibility).",
    )
    parser.add_argument(
        "--raw-root",
        "--raw_root",
        type=str,
        default=None,
        help="Root of raw dataset. Overrides registry defaults.",
    )
    parser.add_argument(
        "--image-subdir",
        "--image_subdir",
        type=str,
        default=None,
        help="Image subdirectory under raw root. Overrides registry defaults.",
    )
    parser.add_argument(
        "--processed-root",
        "--processed_root",
        type=str,
        default=None,
        help="Directory to save split files. Default: data/processed/",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan image directory recursively.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if exist.")

    args = parser.parse_args()
    project_root = PROJECT_ROOT

    raw_root = Path(args.raw_root).resolve() if args.raw_root else build_default_raw_root(project_root, args.dataset_name)
    image_subdir = args.image_subdir if args.image_subdir is not None else resolve_image_subdir(args.dataset_name)
    extensions = resolve_extensions(args.dataset_name)
    processed_root = (
        Path(args.processed_root).resolve() if args.processed_root else build_default_processed_root(project_root)
    )

    train_file = processed_root / "train.txt"
    val_file = processed_root / "val.txt"
    test_file = processed_root / "test.txt"
    train_groups_file = processed_root / "train_groups.txt"
    val_groups_file = processed_root / "val_groups.txt"
    test_groups_file = processed_root / "test_groups.txt"

    output_files = [
        train_file,
        val_file,
        test_file,
        train_groups_file,
        val_groups_file,
        test_groups_file,
    ]

    if not args.overwrite:
        existed = [p for p in output_files if p.exists()]
        if existed:
            existed_str = "\n".join(str(p) for p in existed)
            raise FileExistsError(
                "Output files already exist. Use --overwrite to replace:\n"
                f"{existed_str}"
            )

    print(f"[INFO] Dataset : {args.dataset_name}")
    print(f"[INFO] Raw root: {raw_root}")
    print(f"[INFO] Subdir  : {image_subdir or '.'}")
    print(f"[INFO] Recursive: {'yes' if args.recursive else 'no'}")
    image_paths = find_dataset_images(
        raw_root=raw_root,
        image_subdir=image_subdir,
        recursive=args.recursive,
        extensions=extensions,
    )
    grouped_paths = group_images_by_stem(image_paths)

    (
        train_paths,
        val_paths,
        test_paths,
        train_groups,
        val_groups,
        test_groups,
    ) = split_groups(
        grouped_paths=grouped_paths,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_split_file(train_paths, train_file)
    save_split_file(val_paths, val_file)
    save_split_file(test_paths, test_file)
    save_group_file(train_groups, train_groups_file)
    save_group_file(val_groups, val_groups_file)
    save_group_file(test_groups, test_groups_file)

    print_split_summary(
        total_images=len(image_paths),
        total_groups=len(grouped_paths),
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
    )

    print(f"[INFO] Saved: {train_file}")
    print(f"[INFO] Saved: {val_file}")
    print(f"[INFO] Saved: {test_file}")
    print(f"[INFO] Saved: {train_groups_file}")
    print(f"[INFO] Saved: {val_groups_file}")
    print(f"[INFO] Saved: {test_groups_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
