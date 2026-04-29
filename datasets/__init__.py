from datasets.builder import build_m3fd_dataset, build_sr_dataset
from datasets.sr_dataset import GenericSRDataset

M3FDSRDataset = GenericSRDataset

__all__ = ["GenericSRDataset", "M3FDSRDataset", "build_sr_dataset", "build_m3fd_dataset"]
