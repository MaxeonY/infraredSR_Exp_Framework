from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np


ArrayLike = np.ndarray
PathLike = Union[str, Path]
RngLike = Optional[np.random.Generator]


def mod_crop(img: ArrayLike, scale: int) -> ArrayLike:
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got: {scale}")
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected 2D/3D image, got shape={img.shape}")

    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    return img[:h, :w] if img.ndim == 2 else img[:h, :w, :]


def bicubic_degrade(img: ArrayLike, scale: int) -> ArrayLike:
    img = mod_crop(img, scale)
    h, w = img.shape[:2]
    lr_h = h // scale
    lr_w = w // scale
    if lr_h <= 0 or lr_w <= 0:
        raise ValueError(f"Invalid LR shape from ({h}, {w}) with scale={scale}")
    return cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)


def bicubic_upscale(img: ArrayLike, scale: int) -> ArrayLike:
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got: {scale}")
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def add_random_gaussian_noise(
    img: ArrayLike,
    noise_prob: float = 0.5,
    sigma_range: Tuple[float, float] = (1.0, 10.0),
    rng: RngLike = None,
) -> ArrayLike:
    if not (0.0 <= noise_prob <= 1.0):
        raise ValueError(f"noise_prob must be in [0, 1], got: {noise_prob}")
    if sigma_range[0] < 0 or sigma_range[0] > sigma_range[1]:
        raise ValueError(f"Invalid sigma_range: {sigma_range}")

    if rng is None:
        rng = np.random.default_rng()

    if float(rng.random()) >= noise_prob:
        return img

    sigma = float(rng.uniform(sigma_range[0], sigma_range[1]))
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def add_random_jpeg_compression(
    img: ArrayLike,
    compression_prob: float = 0.5,
    quality_range: Tuple[int, int] = (30, 95),
    rng: RngLike = None,
) -> ArrayLike:
    if not (0.0 <= compression_prob <= 1.0):
        raise ValueError(f"compression_prob must be in [0, 1], got: {compression_prob}")
    if quality_range[0] <= 0 or quality_range[1] > 100 or quality_range[0] > quality_range[1]:
        raise ValueError(f"Invalid quality_range: {quality_range}")

    if rng is None:
        rng = np.random.default_rng()

    if float(rng.random()) >= compression_prob:
        return img

    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), encode_param)
    if not ok:
        raise RuntimeError("Failed to JPEG encode image.")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("Failed to JPEG decode image.")

    if img.ndim == 2 and decoded.ndim == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    return decoded.astype(np.uint8)


def _resolve_degradation_config(
    degradation_cfg: Optional[Dict[str, Any]] = None,
    noise_prob: float = 0.5,
    noise_sigma_range: Tuple[float, float] = (1.0, 10.0),
    compression_prob: float = 0.5,
    jpeg_quality_range: Tuple[int, int] = (30, 95),
) -> Dict[str, Any]:
    cfg = {
        "downsample_mode": "bicubic",
        "noise_type": "gaussian",
        "noise_prob": noise_prob,
        "noise_sigma_range": noise_sigma_range,
        "compression_prob": compression_prob,
        "jpeg_quality_range": jpeg_quality_range,
    }
    if degradation_cfg:
        cfg.update(degradation_cfg)
    return cfg


def generate_lr_hr_pair(
    img: ArrayLike,
    scale: int,
    noise_prob: float = 0.5,
    noise_sigma_range: Tuple[float, float] = (1.0, 10.0),
    compression_prob: float = 0.5,
    jpeg_quality_range: Tuple[int, int] = (30, 95),
    degradation_cfg: Optional[Dict[str, Any]] = None,
    rng: RngLike = None,
) -> Tuple[ArrayLike, ArrayLike]:
    cfg = _resolve_degradation_config(
        degradation_cfg=degradation_cfg,
        noise_prob=noise_prob,
        noise_sigma_range=noise_sigma_range,
        compression_prob=compression_prob,
        jpeg_quality_range=jpeg_quality_range,
    )

    hr = mod_crop(img, scale)
    downsample_mode = str(cfg.get("downsample_mode", "bicubic")).lower()
    if downsample_mode != "bicubic":
        raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
    lr = bicubic_degrade(hr, scale)

    noise_type = str(cfg.get("noise_type", "gaussian")).lower()
    if noise_type == "gaussian":
        lr = add_random_gaussian_noise(
            lr,
            noise_prob=float(cfg.get("noise_prob", 0.5)),
            sigma_range=tuple(cfg.get("noise_sigma_range", (1.0, 10.0))),
            rng=rng,
        )
    elif noise_type in ("none", "off", "disabled"):
        pass
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    lr = add_random_jpeg_compression(
        lr,
        compression_prob=float(cfg.get("compression_prob", 0.5)),
        quality_range=tuple(cfg.get("jpeg_quality_range", (30, 95))),
        rng=rng,
    )
    return lr, hr


def read_grayscale_image(image_path: PathLike) -> ArrayLike:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return img


def normalize_to_float32(img: ArrayLike) -> ArrayLike:
    return img.astype(np.float32) / 255.0


def to_tensor_like_input(img: ArrayLike) -> ArrayLike:
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale [H, W], got shape={img.shape}")
    return np.expand_dims(img, axis=0)


def check_lr_hr_shapes(lr: ArrayLike, hr: ArrayLike, scale: int) -> None:
    lr_h, lr_w = (lr.shape[-2], lr.shape[-1]) if lr.ndim == 3 else lr.shape[:2]
    hr_h, hr_w = (hr.shape[-2], hr.shape[-1]) if hr.ndim == 3 else hr.shape[:2]
    if hr_h != lr_h * scale or hr_w != lr_w * scale:
        raise ValueError(
            f"LR/HR shape mismatch: LR=({lr_h}, {lr_w}), HR=({hr_h}, {hr_w}), scale={scale}"
        )
