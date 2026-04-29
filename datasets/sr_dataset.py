from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.degrade import (
    check_lr_hr_shapes,
    generate_lr_hr_pair,
    mod_crop,
    normalize_to_float32,
    read_grayscale_image,
    to_tensor_like_input,
)


PathLike = Union[str, Path]


class GenericSRDataset(Dataset):
    """Generic grayscale image super-resolution dataset."""

    def __init__(
        self,
        split_file: str,
        scale: int = 2,
        patch_size: int = 64,
        mode: str = "train",
        augment: bool = False,
        cache_in_memory: bool = True,
        degradation_cfg: Optional[Dict[str, Any]] = None,
        raw_root: Optional[PathLike] = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be train/val/test, got: {mode}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got: {scale}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got: {patch_size}")

        self.split_file = Path(split_file)
        self.scale = scale
        self.patch_size = patch_size
        self.mode = mode
        self.augment = augment if mode == "train" else False
        self.degradation_cfg = dict(degradation_cfg or {})
        self.deterministic_eval = bool(self.degradation_cfg.get("deterministic_eval", True))
        self.eval_seed = int(self.degradation_cfg.get("eval_seed", 3407))

        if not self.split_file.exists():
            raise FileNotFoundError(f"split file does not exist: {self.split_file}")

        project_root = Path(__file__).resolve().parent.parent
        resolved_raw_root = self._resolve_root(raw_root, project_root)
        self.image_paths = self._load_split_file(self.split_file, project_root, resolved_raw_root)
        if len(self.image_paths) == 0:
            raise RuntimeError("No valid samples in split file.")

        self._cache: List[np.ndarray] = []
        if cache_in_memory:
            for p in self.image_paths:
                self._cache.append(read_grayscale_image(p))

    @staticmethod
    def _resolve_root(raw_root: Optional[PathLike], project_root: Path) -> Optional[Path]:
        if raw_root is None or str(raw_root).strip() == "":
            return None
        root = Path(raw_root)
        if not root.is_absolute():
            root = project_root / root
        return root.resolve()

    @staticmethod
    def _append_if_exists(image_paths: List[Path], candidate: Path) -> bool:
        try:
            if candidate.exists() and candidate.is_file():
                image_paths.append(candidate)
                return True
        except OSError:
            pass
        return False

    @staticmethod
    def _build_candidate_paths(raw_line: str, project_root: Path, raw_root: Optional[Path]) -> List[Path]:
        candidates: List[Path] = []
        image_path = Path(raw_line)

        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            candidates.append(project_root / image_path)
            if raw_root is not None:
                candidates.append(raw_root / image_path)
                if len(image_path.parts) == 1:
                    candidates.append(raw_root / image_path.name)

        normalized = raw_line.replace("\\", "/")
        normalized_lower = normalized.lower()
        historical_markers = [
            "/data/raw/m3fd/",
            "data/raw/m3fd/",
        ]
        fallback_root = raw_root or (project_root / "data" / "raw" / "M3FD")
        for marker in historical_markers:
            idx = normalized_lower.find(marker)
            if idx >= 0:
                tail = normalized[idx + len(marker) :].lstrip("/")
                candidates.append(fallback_root / tail)
                break

        return candidates

    @classmethod
    def _load_split_file(cls, split_file: Path, project_root: Path, raw_root: Optional[Path]) -> List[Path]:
        image_paths: List[Path] = []
        skipped_meta_files = 0
        skipped_missing_files = 0
        remapped_paths = 0
        first_missing_path = None

        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                raw_line = line.strip().lstrip("\ufeff")
                if not raw_line:
                    continue

                if Path(raw_line).name.startswith("._"):
                    skipped_meta_files += 1
                    continue

                candidates = cls._build_candidate_paths(raw_line, project_root, raw_root)
                for candidate_idx, candidate in enumerate(candidates):
                    if cls._append_if_exists(image_paths, candidate):
                        if candidate_idx > 0:
                            remapped_paths += 1
                        break
                else:
                    skipped_missing_files += 1
                    if first_missing_path is None:
                        first_missing_path = raw_line

        if skipped_meta_files > 0:
            warnings.warn(
                f"Skipped {skipped_meta_files} metadata files in split: {split_file}",
                RuntimeWarning,
                stacklevel=2,
            )
        if remapped_paths > 0:
            warnings.warn(
                f"Remapped {remapped_paths} paths while loading split: {split_file}",
                RuntimeWarning,
                stacklevel=2,
            )
        if skipped_missing_files > 0:
            warnings.warn(
                f"Skipped {skipped_missing_files} missing paths in split: {split_file}. "
                f"First missing: {first_missing_path}",
                RuntimeWarning,
                stacklevel=2,
            )
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def _random_crop_hr_patch(self, hr: np.ndarray) -> np.ndarray:
        h, w = hr.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError(f"image is smaller than patch_size: ({h}, {w}) < {ps}")
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        return hr[top : top + ps, left : left + ps]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.5:
            img = np.flipud(img)
        if np.random.rand() < 0.5:
            img = np.rot90(img)
        return img.copy()

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).float()

    def _build_eval_rng(self, image_path: Path) -> np.random.Generator:
        key = f"{image_path.resolve()}|scale={self.scale}|seed={self.eval_seed}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2 ** 32)
        return np.random.default_rng(seed)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        if self._cache:
            hr_img = self._cache[index]
        else:
            if not image_path.exists():
                raise FileNotFoundError(f"image does not exist: {image_path}")
            hr_img = read_grayscale_image(image_path)

        if self.mode == "train":
            hr_img = mod_crop(hr_img, self.scale)
            hr_patch = self._random_crop_hr_patch(hr_img)
            if self.augment:
                hr_patch = self._augment(hr_patch)
            lr_img, hr_img = generate_lr_hr_pair(
                hr_patch,
                self.scale,
                degradation_cfg=self.degradation_cfg,
                rng=None,
            )
        else:
            hr_img = mod_crop(hr_img, self.scale)
            eval_rng = self._build_eval_rng(image_path) if self.deterministic_eval else None
            lr_img, hr_img = generate_lr_hr_pair(
                hr_img,
                self.scale,
                degradation_cfg=self.degradation_cfg,
                rng=eval_rng,
            )

        check_lr_hr_shapes(lr_img, hr_img, self.scale)
        lr_img = to_tensor_like_input(normalize_to_float32(lr_img))
        hr_img = to_tensor_like_input(normalize_to_float32(hr_img))

        lr_tensor = self._to_tensor(lr_img)
        hr_tensor = self._to_tensor(hr_img)

        if self.mode == "train":
            return lr_tensor, hr_tensor
        return lr_tensor, hr_tensor, str(image_path)
