from typing import Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn


TensorOrArray = Union[torch.Tensor, np.ndarray]


def _to_numpy_image(img: TensorOrArray) -> np.ndarray:
    """
    Convert input image to numpy array with shape [H, W].

    Supported input shapes:
        - torch.Tensor: [H, W], [1, H, W]
        - np.ndarray  : [H, W], [1, H, W]

    Returns:
        np.ndarray with shape [H, W], dtype float32
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float().numpy()

    if not isinstance(img, np.ndarray):
        raise TypeError(f"输入必须是 torch.Tensor 或 np.ndarray，当前类型为：{type(img)}")

    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(
                f"当前只支持单通道图像 [1, H, W]，收到 shape={img.shape}"
            )
        img = img[0]
    elif img.ndim != 2:
        raise ValueError(
            f"当前只支持 [H, W] 或 [1, H, W]，收到 shape={img.shape}"
        )

    return img.astype(np.float32)


def calculate_psnr(
    pred: TensorOrArray,
    target: TensorOrArray,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Calculate PSNR between two single-channel images.

    Args:
        pred: Predicted image, shape [H, W] or [1, H, W]
        target: Ground truth image, shape [H, W] or [1, H, W]
        data_range: Value range of input images. Default is 1.0 for normalized images.
        eps: Small value to avoid log(0)

    Returns:
        PSNR value as float
    """
    pred_np = _to_numpy_image(pred)
    target_np = _to_numpy_image(target)

    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"pred 和 target 尺寸不一致：pred={pred_np.shape}, target={target_np.shape}"
        )

    mse = np.mean((pred_np - target_np) ** 2)

    if mse < eps:
        return float("inf")

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def calculate_ssim(
    pred: TensorOrArray,
    target: TensorOrArray,
    data_range: float = 1.0,
) -> float:
    """
    Calculate SSIM between two single-channel images.

    Args:
        pred: Predicted image, shape [H, W] or [1, H, W]
        target: Ground truth image, shape [H, W] or [1, H, W]
        data_range: Value range of input images. Default is 1.0 for normalized images.

    Returns:
        SSIM value as float
    """
    pred_np = _to_numpy_image(pred)
    target_np = _to_numpy_image(target)

    if pred_np.shape != target_np.shape:
        raise ValueError(
            f"pred 和 target 尺寸不一致：pred={pred_np.shape}, target={target_np.shape}"
        )

    value = ssim_fn(
        target_np,
        pred_np,
        data_range=data_range,
    )
    return float(value)