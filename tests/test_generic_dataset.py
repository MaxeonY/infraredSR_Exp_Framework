from pathlib import Path

import cv2
import numpy as np

from datasets.builder import build_m3fd_dataset, build_sr_dataset
from datasets.preprocess import find_dataset_images, group_images_by_stem, split_groups
from datasets.sr_dataset import GenericSRDataset


def _write_gray_image(path: Path, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = ((np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) + offset) % 256).astype(np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_generic_sr_dataset_loads_paths_relative_to_raw_root(tmp_path):
    raw_root = tmp_path / "raw"
    image_path = raw_root / "images" / "sample.png"
    _write_gray_image(image_path)

    split_file = tmp_path / "train.txt"
    split_file.write_text("images/sample.png\n", encoding="utf-8")

    dataset = build_sr_dataset(
        split_file=str(split_file),
        scale=2,
        patch_size=16,
        mode="val",
        cache_in_memory=False,
        degradation_cfg={"noise_type": "none", "compression_prob": 0.0},
        raw_root=raw_root,
    )

    lr, hr, path = dataset[0]
    assert len(dataset) == 1
    assert lr.shape == (1, 16, 16)
    assert hr.shape == (1, 32, 32)
    assert Path(path) == image_path


def test_m3fd_builder_alias_returns_generic_dataset(tmp_path):
    raw_root = tmp_path / "raw"
    image_path = raw_root / "sample.png"
    _write_gray_image(image_path)

    split_file = tmp_path / "train.txt"
    split_file.write_text("sample.png\n", encoding="utf-8")

    dataset = build_m3fd_dataset(
        split_file=str(split_file),
        scale=2,
        patch_size=16,
        mode="val",
        cache_in_memory=False,
        degradation_cfg={"noise_type": "none", "compression_prob": 0.0},
        raw_root=raw_root,
    )

    assert isinstance(dataset, GenericSRDataset)
    assert len(dataset) == 1


def test_generic_dataset_keeps_historical_m3fd_path_remap(tmp_path):
    raw_root = tmp_path / "raw"
    image_path = raw_root / "Ir" / "sample.png"
    _write_gray_image(image_path)

    split_file = tmp_path / "test.txt"
    split_file.write_text("/old/project/data/raw/M3FD/Ir/sample.png\n", encoding="utf-8")

    dataset = build_sr_dataset(
        split_file=str(split_file),
        scale=2,
        patch_size=16,
        mode="test",
        cache_in_memory=False,
        degradation_cfg={"noise_type": "none", "compression_prob": 0.0},
        raw_root=raw_root,
    )

    assert len(dataset) == 1
    assert dataset.image_paths[0] == image_path


def test_preprocess_helpers_are_dataset_agnostic(tmp_path):
    raw_root = tmp_path / "raw"
    _write_gray_image(raw_root / "nested" / "a.png", offset=1)
    _write_gray_image(raw_root / "nested" / "b.png", offset=2)
    _write_gray_image(raw_root / "nested" / "c.png", offset=3)

    images = find_dataset_images(raw_root=raw_root, image_subdir="nested", recursive=False)
    grouped = group_images_by_stem(images)
    train, val, test, train_groups, val_groups, test_groups = split_groups(grouped, 0.6, 0.2, seed=1)

    assert len(images) == 3
    assert len(grouped) == 3
    assert len(train) + len(val) + len(test) == 3
    assert len(train_groups) + len(val_groups) + len(test_groups) == 3
