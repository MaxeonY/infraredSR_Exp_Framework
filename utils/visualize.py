from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


TensorOrArray = Union[torch.Tensor, np.ndarray]


def _to_numpy_image(img: TensorOrArray) -> np.ndarray:
    """
    Convert image to numpy array with shape [H, W].

    Supported input:
        - torch.Tensor: [H, W], [1, H, W]
        - np.ndarray  : [H, W], [1, H, W]

    Returns:
        np.ndarray, shape [H, W], dtype float32
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float().numpy()

    if not isinstance(img, np.ndarray):
        raise TypeError(f"输入必须是 torch.Tensor 或 np.ndarray，当前类型为：{type(img)}")

    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"当前只支持单通道图像 [1, H, W]，收到 shape={img.shape}")
        img = img[0]
    elif img.ndim != 2:
        raise ValueError(f"当前只支持 [H, W] 或 [1, H, W]，收到 shape={img.shape}")

    return img.astype(np.float32)


def _clip_image(img: np.ndarray) -> np.ndarray:
    """
    Clip image to [0, 1].
    """
    return np.clip(img, 0.0, 1.0)


def save_image(
    img: TensorOrArray,
    save_path: str,
    cmap: str = "gray",
) -> None:
    """
    Save a single grayscale image.

    Args:
        img: Input image
        save_path: Output image path
        cmap: Matplotlib colormap, default 'gray'
    """
    img_np = _clip_image(_to_numpy_image(img))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_comparison_figure(
    lr: TensorOrArray,
    sr: TensorOrArray,
    hr: TensorOrArray,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "gray",
) -> None:
    """
    Save a side-by-side comparison figure: LR / SR / HR

    Args:
        lr: Low-resolution image
        sr: Super-resolved image
        hr: Ground-truth high-resolution image
        save_path: Output figure path
        title: Optional figure title
        cmap: Matplotlib colormap, default 'gray'
    """
    lr_np = _clip_image(_to_numpy_image(lr))
    sr_np = _clip_image(_to_numpy_image(sr))
    hr_np = _clip_image(_to_numpy_image(hr))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    if title is not None:
        plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(lr_np, cmap=cmap)
    plt.title("LR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sr_np, cmap=cmap)
    plt.title("SR")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hr_np, cmap=cmap)
    plt.title("HR")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_difference_map(
    sr: TensorOrArray,
    hr: TensorOrArray,
    save_path: str,
    cmap: str = "hot",
) -> None:
    """
    Save absolute difference map between SR and HR.

    Args:
        sr: Super-resolved image
        hr: Ground-truth image
        save_path: Output image path
        cmap: Colormap for difference map, default 'hot'
    """
    sr_np = _clip_image(_to_numpy_image(sr))
    hr_np = _clip_image(_to_numpy_image(hr))

    if sr_np.shape != hr_np.shape:
        raise ValueError(f"SR 和 HR 尺寸不一致：SR={sr_np.shape}, HR={hr_np.shape}")

    diff = np.abs(sr_np - hr_np)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(diff, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Absolute Difference")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()