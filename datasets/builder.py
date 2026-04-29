from typing import Any, Dict, Optional, Union
from pathlib import Path

from datasets.sr_dataset import GenericSRDataset


def build_sr_dataset(
    split_file: str,
    scale: int,
    patch_size: int,
    mode: str,
    augment: bool = False,
    cache_in_memory: bool = True,
    degradation_cfg: Optional[Dict[str, Any]] = None,
    raw_root: Optional[Union[str, Path]] = None,
) -> GenericSRDataset:
    return GenericSRDataset(
        split_file=split_file,
        scale=scale,
        patch_size=patch_size,
        mode=mode,
        augment=augment,
        cache_in_memory=cache_in_memory,
        degradation_cfg=degradation_cfg,
        raw_root=raw_root,
    )


build_m3fd_dataset = build_sr_dataset
