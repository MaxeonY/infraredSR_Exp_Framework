import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.degrade import normalize_to_float32, read_grayscale_image
from models import SUPPORTED_MODELS, build_model, merge_model_kwargs
from utils.checkpoint import load_checkpoint, read_checkpoint
from utils.metrics import calculate_extended_metrics, calculate_psnr, calculate_ssim
from utils.visualize import save_comparison_figure, save_difference_map


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXTENDED_METRIC_KEYS = ["mse", "rmse", "gradient_mae", "laplacian_mae", "fft_l1", "hfen"]


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_model_scale_from_checkpoint(ckpt_path: Path) -> Tuple[Optional[str], Optional[int]]:
    m = re.match(r"^(?P<model>[a-zA-Z0-9_]+)_x(?P<scale>\d+)_(best|latest)\.pth$", ckpt_path.name)
    if not m:
        return None, None
    return m.group("model").lower(), int(m.group("scale"))


def resolve_run_config(args) -> Tuple[str, int, Path]:
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    model_name = args.model
    scale = args.scale

    if checkpoint_path is not None:
        inferred_model, inferred_scale = infer_model_scale_from_checkpoint(checkpoint_path)
        if model_name is None and inferred_model in SUPPORTED_MODELS:
            model_name = inferred_model
        if scale is None and inferred_scale is not None:
            scale = inferred_scale

    if model_name is None or scale is None:
        raise ValueError("Please provide --model and --scale, or provide a parseable --checkpoint.")
    if checkpoint_path is None:
        checkpoint_path = (Path("outputs/checkpoints") / f"{model_name}_x{scale}_best.pth").resolve()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {SUPPORTED_MODELS}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got: {scale}")
    return model_name, scale, checkpoint_path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_input_images(input_path: Path, recursive: bool) -> Tuple[List[Path], Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path], input_path.parent

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    pattern = "**/*" if recursive else "*"
    images = [p for p in sorted(input_path.glob(pattern)) if is_image_file(p)]
    if len(images) == 0:
        raise RuntimeError(f"No image files found in directory: {input_path}")
    return images, input_path


def prepare_model_input(model_name: str, lr: torch.Tensor, scale: int) -> torch.Tensor:
    if model_name.lower() in ("srcnn", "srcnn_arf"):
        return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    return lr


def to_model_input_tensor(image_path: Path) -> torch.Tensor:
    img_u8 = read_grayscale_image(image_path)
    img = normalize_to_float32(img_u8)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()


def tensor_to_numpy_2d(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [H, W] or [1, H, W], got: {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def save_grayscale_float_image(img: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(save_path), img_u8)
    if not ok:
        raise RuntimeError(f"Failed to save image: {save_path}")


def build_sr_output_path(image_path: Path, input_root: Path, output_dir: Path, model_name: str, scale: int) -> Path:
    try:
        rel = image_path.resolve().relative_to(input_root.resolve())
        rel_parent = rel.parent
    except Exception:
        rel_parent = Path()
    return output_dir / rel_parent / f"{image_path.stem}_sr_{model_name}_x{scale}.png"


def resolve_gt_path(image_path: Path, gt_ref: Path, input_root: Path) -> Optional[Path]:
    gt_ref = gt_ref.resolve()
    if gt_ref.is_file():
        return gt_ref
    if not gt_ref.is_dir():
        return None

    candidates: List[Path] = []
    try:
        rel = image_path.resolve().relative_to(input_root.resolve())
        candidates.append(gt_ref / rel)
        candidates.append((gt_ref / rel).with_suffix(".png"))
    except Exception:
        pass
    candidates.append(gt_ref / image_path.name)
    for ext in IMAGE_EXTENSIONS:
        candidates.append(gt_ref / f"{image_path.stem}{ext}")

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def align_sr_gt(sr: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    if sr.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"SR and GT must be 2D, got {sr.shape} and {gt.shape}")
    mismatch = sr.shape != gt.shape
    h = min(sr.shape[0], gt.shape[0])
    w = min(sr.shape[1], gt.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid aligned size from SR {sr.shape} and GT {gt.shape}")
    return sr[:h, :w], gt[:h, :w], mismatch


def save_metrics_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    if len(rows) == 0:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    extended_enabled = any(any(k in row and row[k] != "" for k in EXTENDED_METRIC_KEYS) for row in rows)
    fieldnames = [
        "filename",
        "input_path",
        "sr_path",
        "gt_path",
        "l1",
        "psnr",
        "ssim",
    ]
    if extended_enabled:
        fieldnames.extend(EXTENDED_METRIC_KEYS)
    fieldnames.extend(["sr_shape", "gt_shape", "shape_aligned"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_model_from_checkpoint(model_name: str, scale: int, checkpoint_path: Path, device: torch.device):
    raw_ckpt = read_checkpoint(str(checkpoint_path), map_location=device.type)
    meta = raw_ckpt.get("model_meta", {}) if isinstance(raw_ckpt, dict) else {}
    meta_kwargs = meta.get("model_kwargs", {}) if isinstance(meta, dict) else {}
    model_kwargs = merge_model_kwargs(model_name=model_name, config_kwargs=meta_kwargs if isinstance(meta_kwargs, dict) else None)
    model = build_model(model_name=model_name, scale=scale, **model_kwargs).to(device)
    checkpoint = load_checkpoint(str(checkpoint_path), model=model, optimizer=None, map_location=device.type)
    return model, checkpoint


def run_inference(args, logger) -> None:
    model_name, scale, checkpoint_path = resolve_run_config(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_images, input_root = collect_input_images(input_path, recursive=args.recursive)
    gt_ref = None
    if args.gt is not None:
        gt_ref = Path(args.gt).resolve()
        if not gt_ref.exists():
            raise FileNotFoundError(f"--gt path does not exist: {gt_ref}")

    model, checkpoint = _build_model_from_checkpoint(model_name, scale, checkpoint_path, device)
    model.eval()

    logger.info("=" * 60)
    logger.info("Inference started")
    logger.info(f"Model       : {model_name}")
    logger.info(f"Scale       : x{scale}")
    logger.info(f"Device      : {device}")
    logger.info(f"Checkpoint  : {checkpoint_path}")
    logger.info(f"Input       : {input_path.resolve()}")
    logger.info(f"Num images  : {len(input_images)}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Extended metrics: {'on' if args.extended_metrics else 'off'}")
    if "epoch" in checkpoint:
        logger.info(f"Ckpt epoch  : {checkpoint['epoch']}")
    if "best_metric" in checkpoint:
        logger.info(f"Ckpt metric : {checkpoint['best_metric']}")
    logger.info("=" * 60)

    metric_rows: List[Dict[str, object]] = []
    metric_count = 0
    metric_psnr_sum = 0.0
    metric_ssim_sum = 0.0
    metric_l1_sum = 0.0
    ext_sums: Dict[str, float] = {k: 0.0 for k in EXTENDED_METRIC_KEYS}

    with torch.no_grad():
        for idx, image_path in enumerate(input_images, start=1):
            lr = to_model_input_tensor(image_path)
            lr_dev = lr.to(device, non_blocking=True)
            sr = model(prepare_model_input(model_name, lr_dev, scale)).clamp(0.0, 1.0)

            sr_np = tensor_to_numpy_2d(sr[0])
            sr_path = build_sr_output_path(image_path, input_root, output_dir, model_name, scale)
            save_grayscale_float_image(sr_np, sr_path)

            if args.save_bicubic:
                bicubic = F.interpolate(lr_dev, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
                bicubic_np = tensor_to_numpy_2d(bicubic[0])
                bicubic_path = sr_path.with_name(sr_path.stem.replace("_sr_", "_bicubic_") + ".png")
                save_grayscale_float_image(bicubic_np, bicubic_path)

            row: Dict[str, object] = {
                "filename": image_path.name,
                "input_path": str(image_path.resolve()),
                "sr_path": str(sr_path.resolve()),
                "gt_path": "",
                "l1": "",
                "psnr": "",
                "ssim": "",
                "sr_shape": f"{sr_np.shape[0]}x{sr_np.shape[1]}",
                "gt_shape": "",
                "shape_aligned": "",
            }
            for key in EXTENDED_METRIC_KEYS:
                row[key] = ""

            if gt_ref is not None:
                gt_path = resolve_gt_path(image_path=image_path, gt_ref=gt_ref, input_root=input_root)
                if gt_path is None:
                    logger.warning(f"[{idx}/{len(input_images)}] GT not found for: {image_path.name}")
                else:
                    gt_np = normalize_to_float32(read_grayscale_image(gt_path))
                    sr_aligned, gt_aligned, mismatch = align_sr_gt(sr_np, gt_np)
                    sr_tensor = torch.from_numpy(sr_aligned).unsqueeze(0)
                    gt_tensor = torch.from_numpy(gt_aligned).unsqueeze(0)
                    l1 = float(np.mean(np.abs(sr_aligned - gt_aligned)))
                    psnr = calculate_psnr(sr_tensor, gt_tensor)
                    ssim = calculate_ssim(sr_tensor, gt_tensor)

                    metric_count += 1
                    metric_l1_sum += l1
                    metric_psnr_sum += psnr
                    metric_ssim_sum += ssim
                    row.update(
                        {
                            "gt_path": str(gt_path.resolve()),
                            "l1": f"{l1:.6f}",
                            "psnr": f"{psnr:.6f}",
                            "ssim": f"{ssim:.6f}",
                            "gt_shape": f"{gt_np.shape[0]}x{gt_np.shape[1]}",
                            "shape_aligned": "yes" if mismatch else "no_need",
                        }
                    )

                    if args.extended_metrics:
                        ext_vals = calculate_extended_metrics(sr_tensor, gt_tensor)
                        for key in EXTENDED_METRIC_KEYS:
                            ext_sums[key] += float(ext_vals[key])
                            row[key] = f"{float(ext_vals[key]):.6f}"

                    if args.save_visuals:
                        vis_root = output_dir / "visuals"
                        try:
                            rel = image_path.resolve().relative_to(input_root.resolve())
                            rel_parent = rel.parent
                        except Exception:
                            rel_parent = Path()
                        vis_dir = vis_root / rel_parent
                        vis_dir.mkdir(parents=True, exist_ok=True)

                        lr_for_vis = F.interpolate(
                            lr,
                            size=(sr_aligned.shape[0], sr_aligned.shape[1]),
                            mode="bicubic",
                            align_corners=False,
                        )[0]
                        sr_vis = torch.from_numpy(sr_aligned).unsqueeze(0)
                        gt_vis = torch.from_numpy(gt_aligned).unsqueeze(0)
                        save_comparison_figure(
                            lr=lr_for_vis,
                            sr=sr_vis,
                            hr=gt_vis,
                            save_path=str(vis_dir / f"{image_path.stem}_cmp_{model_name}_x{scale}.png"),
                            title=f"{model_name.upper()} x{scale} | {image_path.stem}",
                        )
                        save_difference_map(
                            sr=sr_vis,
                            hr=gt_vis,
                            save_path=str(vis_dir / f"{image_path.stem}_diff_{model_name}_x{scale}.png"),
                        )

            metric_rows.append(row)
            if idx % 20 == 0 or idx == len(input_images):
                logger.info(f"Processed {idx}/{len(input_images)} images")

    metrics_csv_path = output_dir / f"infer_{model_name}_x{scale}_metrics.csv"
    save_metrics_csv(metric_rows, metrics_csv_path)
    logger.info(f"Saved inference table: {metrics_csv_path}")
    if metric_count > 0:
        logger.info(
            f"GT-evaluable samples: {metric_count} | "
            f"L1={metric_l1_sum / metric_count:.6f} | "
            f"PSNR={metric_psnr_sum / metric_count:.4f} | "
            f"SSIM={metric_ssim_sum / metric_count:.6f}"
        )
        if args.extended_metrics:
            logger.info(
                "Extended means | "
                + " | ".join(
                    [
                        f"MSE={ext_sums['mse'] / metric_count:.6f}",
                        f"RMSE={ext_sums['rmse'] / metric_count:.6f}",
                        f"GradMAE={ext_sums['gradient_mae'] / metric_count:.6f}",
                        f"LapMAE={ext_sums['laplacian_mae'] / metric_count:.6f}",
                        f"FFT_L1={ext_sums['fft_l1'] / metric_count:.6f}",
                        f"HFEN={ext_sums['hfen'] / metric_count:.6f}",
                    ]
                )
            )
    else:
        logger.info("No GT metrics computed. Provide --gt for quantitative evaluation.")
    logger.info("Inference finished.")
