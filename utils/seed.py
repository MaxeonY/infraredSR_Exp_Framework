import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: Whether to force deterministic behavior in cuDNN.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int，当前类型为：{type(seed)}")

    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 保证尽量可复现，但可能会影响速度
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 某些 PyTorch 算子会要求这一设置
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # 更快，但可能不完全可复现
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers.

    Args:
        worker_id: Worker ID assigned by PyTorch DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)