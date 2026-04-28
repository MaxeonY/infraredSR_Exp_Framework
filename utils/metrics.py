from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn


TensorOrArray = Union[torch.Tensor, np.ndarray]


SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=torch.float32,
)
SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=torch.float32,
)
LAPLACIAN_4N = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)


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
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img)}")

    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"Expected [1, H, W] for 3D input, got shape={img.shape}")
        img = img[0]
    elif img.ndim != 2:
        raise ValueError(f"Expected [H, W] or [1, H, W], got shape={img.shape}")

    return img.astype(np.float32)


def _to_torch_4d(img: TensorOrArray) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        tensor = torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        tensor = img
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img)}")

    tensor = tensor.detach().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected 2D/3D/4D input, got shape={tuple(tensor.shape)}")

    if tensor.shape[1] <= 0:
        raise ValueError(f"Invalid channel dimension: {tuple(tensor.shape)}")
    return tensor


def _assert_same_shape(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")


def _depthwise_conv3x3(x: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    kernel = kernel_2d.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=c)


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
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, target={target_np.shape}")

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
        raise ValueError(f"Shape mismatch: pred={pred_np.shape}, target={target_np.shape}")

    value = ssim_fn(
        target_np,
        pred_np,
        data_range=data_range,
    )
    return float(value)


def calculate_mse(sr: TensorOrArray, hr: TensorOrArray) -> float:
    sr_t = _to_torch_4d(sr)
    hr_t = _to_torch_4d(hr)
    _assert_same_shape(sr_t, hr_t)
    return float(torch.mean((sr_t - hr_t) ** 2).item())


def calculate_rmse(sr: TensorOrArray, hr: TensorOrArray) -> float:
    mse = calculate_mse(sr, hr)
    return float(np.sqrt(mse))


def calculate_gradient_mae(sr: TensorOrArray, hr: TensorOrArray, eps: float = 1e-12) -> float:
    sr_t = _to_torch_4d(sr)
    hr_t = _to_torch_4d(hr)
    _assert_same_shape(sr_t, hr_t)

    sr_gx = _depthwise_conv3x3(sr_t, SOBEL_X)
    sr_gy = _depthwise_conv3x3(sr_t, SOBEL_Y)
    hr_gx = _depthwise_conv3x3(hr_t, SOBEL_X)
    hr_gy = _depthwise_conv3x3(hr_t, SOBEL_Y)

    sr_g = torch.sqrt(sr_gx ** 2 + sr_gy ** 2 + eps)
    hr_g = torch.sqrt(hr_gx ** 2 + hr_gy ** 2 + eps)
    return float(torch.mean(torch.abs(sr_g - hr_g)).item())


def calculate_laplacian_mae(sr: TensorOrArray, hr: TensorOrArray) -> float:
    sr_t = _to_torch_4d(sr)
    hr_t = _to_torch_4d(hr)
    _assert_same_shape(sr_t, hr_t)

    sr_lap = _depthwise_conv3x3(sr_t, LAPLACIAN_4N)
    hr_lap = _depthwise_conv3x3(hr_t, LAPLACIAN_4N)
    return float(torch.mean(torch.abs(sr_lap - hr_lap)).item())


def calculate_fft_l1(sr: TensorOrArray, hr: TensorOrArray) -> float:
    sr_t = _to_torch_4d(sr)
    hr_t = _to_torch_4d(hr)
    _assert_same_shape(sr_t, hr_t)

    sr_fft = torch.fft.rfft2(sr_t, dim=(-2, -1))
    hr_fft = torch.fft.rfft2(hr_t, dim=(-2, -1))
    return float(torch.mean(torch.abs(sr_fft - hr_fft)).item())


def calculate_hfen(sr: TensorOrArray, hr: TensorOrArray, eps: float = 1e-12) -> float:
    sr_t = _to_torch_4d(sr)
    hr_t = _to_torch_4d(hr)
    _assert_same_shape(sr_t, hr_t)

    sr_lap = _depthwise_conv3x3(sr_t, LAPLACIAN_4N)
    hr_lap = _depthwise_conv3x3(hr_t, LAPLACIAN_4N)
    numerator = torch.linalg.norm(sr_lap - hr_lap)
    denominator = torch.linalg.norm(hr_lap)
    return float((numerator / (denominator + eps)).item())


def calculate_extended_metrics(sr: TensorOrArray, hr: TensorOrArray) -> Dict[str, float]:
    mse = calculate_mse(sr, hr)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "gradient_mae": calculate_gradient_mae(sr, hr),
        "laplacian_mae": calculate_laplacian_mae(sr, hr),
        "fft_l1": calculate_fft_l1(sr, hr),
        "hfen": calculate_hfen(sr, hr),
    }
