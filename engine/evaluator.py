import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.m3fd_dataset import M3FDSRDataset
from models import SUPPORTED_MODELS, build_model, get_model_default_kwargs, merge_model_kwargs
from utils.checkpoint import load_checkpoint, read_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualize import save_comparison_figure, save_difference_map


def infer_model_scale_from_checkpoint(ckpt_path: Path) -> Tuple[Optional[str], Optional[int]]:
    m = re.match(r"^(?P<model>[a-zA-Z0-9_]+)_x(?P<scale>\d+)_(best|latest)\.pth$", ckpt_path.name)
    if not m:
        return None, None
    return m.group("model").lower(), int(m.group("scale"))


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_model_input(model_name: str, lr: torch.Tensor, scale: int) -> torch.Tensor:
    if model_name.lower() in ("srcnn", "srcnn_arf"):
        return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    return lr


def _normalize_path_for_match(path_like: str) -> str:
    return str(path_like).replace("\\", "/").lower()


def _canonicalize_path(path_like: str) -> str:
    p = Path(path_like)
    try:
        p = p.resolve()
    except Exception:
        pass
    return _normalize_path_for_match(str(p))


def resolve_sample_indices(
    dataset: M3FDSRDataset,
    sample_index: Optional[int],
    sample_path: Optional[str],
    max_test_samples: Optional[int],
) -> List[int]:
    total = len(dataset)
    indices = list(range(total))
    selected_from_index: Optional[int] = None
    selected_from_path: Optional[int] = None

    if sample_index is not None:
        idx = sample_index if sample_index >= 0 else total + sample_index
        if idx < 0 or idx >= total:
            raise IndexError(
                f"sample_index out of range: {sample_index}. valid [0, {total - 1}] or [-{total}, -1]."
            )
        selected_from_index = idx

    if sample_path is not None:
        needle_raw = sample_path.strip()
        if not needle_raw:
            raise ValueError("sample_path is empty.")
        needle_norm = _normalize_path_for_match(needle_raw)
        needle_name = Path(needle_raw).name.lower()
        needle_stem = Path(needle_raw).stem.lower()

        matches: List[int] = []
        for i, p in enumerate(dataset.image_paths):
            path_abs = _canonicalize_path(str(p))
            path_raw = _normalize_path_for_match(str(p))
            if (
                needle_norm == path_abs
                or needle_norm == path_raw
                or path_abs.endswith(needle_norm)
                or path_raw.endswith(needle_norm)
                or needle_name == p.name.lower()
                or needle_stem == p.stem.lower()
            ):
                matches.append(i)
        if len(matches) == 0:
            raise ValueError(f"sample_path not found in split: {sample_path}")
        if len(matches) > 1:
            preview = ", ".join(str(i) for i in matches[:8])
            raise ValueError(f"sample_path is ambiguous ({len(matches)} matches): {preview}")
        selected_from_path = matches[0]

    if (
        selected_from_index is not None
        and selected_from_path is not None
        and selected_from_index != selected_from_path
    ):
        raise ValueError(
            f"sample_index ({sample_index}) and sample_path ({sample_path}) point to different samples."
        )

    if selected_from_index is not None:
        indices = [selected_from_index]
    elif selected_from_path is not None:
        indices = [selected_from_path]

    if max_test_samples is not None and max_test_samples > 0:
        indices = indices[:max_test_samples]
    if len(indices) == 0:
        raise RuntimeError("No samples selected for evaluation.")
    return indices


def save_per_sample_metrics(sample_metrics: List[Dict[str, object]], metrics_dir: Path) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "per_sample_metrics.csv"
    fieldnames = ["index", "dataset_index", "path", "l1", "psnr", "ssim"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sample_metrics:
            writer.writerow(
                {
                    "index": item["index"],
                    "dataset_index": item.get("dataset_index", item["index"]),
                    "path": item["path"],
                    "l1": f"{float(item['l1']):.6f}",
                    "psnr": f"{float(item['psnr']):.6f}",
                    "ssim": f"{float(item['ssim']):.6f}",
                }
            )
    return csv_path


def _build_model_from_checkpoint(model_name: str, scale: int, checkpoint_path: Path, device: torch.device):
    raw_ckpt = read_checkpoint(str(checkpoint_path), map_location=device.type)
    meta = raw_ckpt.get("model_meta", {}) if isinstance(raw_ckpt, dict) else {}
    meta_kwargs = meta.get("model_kwargs", {}) if isinstance(meta, dict) else {}
    model_kwargs = merge_model_kwargs(
        model_name=model_name,
        config_kwargs=meta_kwargs if isinstance(meta_kwargs, dict) else get_model_default_kwargs(model_name),
    )
    model = build_model(model_name=model_name, scale=scale, **model_kwargs).to(device)
    checkpoint = load_checkpoint(str(checkpoint_path), model=model, optimizer=None, map_location=device.type)
    return model, checkpoint


def evaluate_single_model(
    args,
    model_name: str,
    scale: int,
    device: torch.device,
    logger,
    checkpoint_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        checkpoint_path = (Path("outputs/checkpoints") / f"{model_name}_x{scale}_best.pth").resolve()
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path} - skipping {model_name} x{scale}")
        return None

    save_visuals = not args.no_visuals
    logger.info("=" * 60)
    logger.info(f"Model       : {model_name}")
    logger.info(f"Scale       : x{scale}")
    logger.info(f"Device      : {device}")
    logger.info(f"Checkpoint  : {checkpoint_path}")
    logger.info(f"Test split  : {args.test_split}")

    dataset = M3FDSRDataset(
        split_file=args.test_split,
        scale=scale,
        patch_size=args.patch_size,
        mode="test",
        augment=False,
        degradation_cfg=getattr(args, "degradation_cfg", None),
    )
    selected_indices = resolve_sample_indices(
        dataset=dataset,
        sample_index=args.sample_index,
        sample_path=args.sample_path,
        max_test_samples=args.max_test_samples,
    )
    eval_dataset = dataset if len(selected_indices) == len(dataset) else Subset(dataset, selected_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"Test samples: {len(eval_dataset)} / {len(dataset)}")
    if len(selected_indices) == 1:
        one_idx = selected_indices[0]
        logger.info(f"Selected sample: index={one_idx}, path={dataset.image_paths[one_idx]}")

    path_to_dataset_index = {_canonicalize_path(str(p)): i for i, p in enumerate(dataset.image_paths)}
    model, checkpoint = _build_model_from_checkpoint(model_name, scale, checkpoint_path, device)
    model.eval()
    if "epoch" in checkpoint:
        logger.info(f"Checkpoint epoch      : {checkpoint['epoch']}")
    if "best_metric" in checkpoint:
        logger.info(f"Checkpoint best metric: {checkpoint['best_metric']}")

    criterion = nn.L1Loss()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    sample_metrics: List[Dict[str, object]] = []

    save_results_dir = Path(args.save_results_dir)
    run_label = f"{model_name}_x{scale}"
    if len(selected_indices) != len(dataset):
        if len(selected_indices) == 1:
            run_label = f"{run_label}_sample{selected_indices[0]:05d}"
        else:
            run_label = f"{run_label}_subset{len(selected_indices)}"
    run_root = save_results_dir / run_label
    vis_seq_dir = run_root / "figures" / "sequential"
    vis_rank_dir = run_root / "figures" / "ranked"
    metrics_dir = run_root / "metrics"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            lr, hr, paths = batch
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            model_input = prepare_model_input(model_name, lr, scale)
            sr = model(model_input)
            total_loss += criterion(sr, hr).item() * lr.size(0)

            for i in range(lr.size(0)):
                psnr = calculate_psnr(sr[i], hr[i])
                ssim = calculate_ssim(sr[i], hr[i])
                l1 = torch.mean(torch.abs(sr[i] - hr[i])).item()
                sample_path = str(paths[i])
                dataset_index = path_to_dataset_index.get(_canonicalize_path(sample_path), -1)
                sample_metrics.append(
                    {
                        "index": total_samples,
                        "dataset_index": dataset_index,
                        "path": sample_path,
                        "l1": l1,
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )
                total_psnr += psnr
                total_ssim += ssim
                total_samples += 1

                if save_visuals and total_samples <= args.max_visuals:
                    stem = Path(sample_path).stem
                    save_comparison_figure(
                        lr=lr[i].cpu(),
                        sr=sr[i].cpu(),
                        hr=hr[i].cpu(),
                        save_path=str(vis_seq_dir / f"{total_samples - 1:04d}_{stem}_cmp.png"),
                        title=f"{model_name.upper()} x{scale} | {stem}",
                    )
                    save_difference_map(
                        sr=sr[i].cpu(),
                        hr=hr[i].cpu(),
                        save_path=str(vis_seq_dir / f"{total_samples - 1:04d}_{stem}_diff.png"),
                    )

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Processed {total_samples}/{len(eval_dataset)} samples...")

    avg_loss = total_loss / max(total_samples, 1)
    avg_psnr = total_psnr / max(total_samples, 1)
    avg_ssim = total_ssim / max(total_samples, 1)
    logger.info(f"  Result - L1: {avg_loss:.6f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.6f}")

    run_root.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = save_per_sample_metrics(sample_metrics, metrics_dir)
    logger.info(f"  Saved per-sample metrics: {metrics_csv_path}")

    if save_visuals and not args.no_rank_visuals:
        top_k = max(0, min(args.rank_visuals_k, len(sample_metrics)))
        sorted_by_psnr = sorted(sample_metrics, key=lambda x: float(x["psnr"]), reverse=True)
        groups = {
            "best_psnr": sorted_by_psnr[:top_k],
            "worst_psnr": list(reversed(sorted_by_psnr[-top_k:])),
        }
        with torch.inference_mode():
            for group_name, group_samples in groups.items():
                group_dir = vis_rank_dir / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                for rank, item in enumerate(group_samples, start=1):
                    ds_idx = int(item.get("dataset_index", -1))
                    if ds_idx < 0 or ds_idx >= len(dataset):
                        continue
                    lr, hr, path = dataset[ds_idx]
                    lr_batch = lr.unsqueeze(0).to(device, non_blocking=True)
                    hr_batch = hr.unsqueeze(0).to(device, non_blocking=True)
                    sr_batch = model(prepare_model_input(model_name, lr_batch, scale))
                    stem = Path(path).stem
                    prefix = f"{rank:02d}_idx{ds_idx:04d}_{stem}"
                    save_comparison_figure(
                        lr=lr_batch[0].cpu(),
                        sr=sr_batch[0].cpu(),
                        hr=hr_batch[0].cpu(),
                        save_path=str(group_dir / f"{prefix}_cmp.png"),
                        title=(
                            f"{group_name} | {model_name.upper()} x{scale} | "
                            f"PSNR {float(item['psnr']):.2f} | SSIM {float(item['ssim']):.4f}"
                        ),
                    )
                    save_difference_map(
                        sr=sr_batch[0].cpu(),
                        hr=hr_batch[0].cpu(),
                        save_path=str(group_dir / f"{prefix}_diff.png"),
                    )

                    # Keep peak GPU memory flat when ranking visuals are enabled.
                    del lr_batch, hr_batch, sr_batch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        logger.info(f"  Saved ranked visualizations: {vis_rank_dir}")

    best_sample = max(sample_metrics, key=lambda x: float(x["psnr"])) if sample_metrics else None
    worst_sample = min(sample_metrics, key=lambda x: float(x["psnr"])) if sample_metrics else None
    selected_indices_text = ",".join(str(i) for i in selected_indices[:100])
    if len(selected_indices) > 100:
        selected_indices_text += ",..."

    report_path = run_root / f"{run_label}_test_report.txt"
    report_lines = [
        f"model={model_name}",
        f"scale={scale}",
        f"checkpoint={checkpoint_path}",
        f"test_split={args.test_split}",
        f"num_samples={total_samples}",
        f"num_total_samples={len(dataset)}",
        f"selected_indices={selected_indices_text}",
        f"avg_l1_loss={avg_loss:.6f}",
        f"avg_psnr={avg_psnr:.4f}",
        f"avg_ssim={avg_ssim:.6f}",
        f"device={device}",
        f"metrics_csv={metrics_csv_path}",
        "metric_charts=disabled",
        f"sequential_visuals_dir={vis_seq_dir if save_visuals else 'disabled'}",
        f"ranked_visuals_dir={vis_rank_dir if (save_visuals and not args.no_rank_visuals) else 'disabled'}",
    ]
    if best_sample is not None:
        report_lines.extend(
            [
                f"best_sample_index={best_sample['index']}",
                f"best_sample_dataset_index={best_sample.get('dataset_index', -1)}",
                f"best_sample_path={best_sample['path']}",
                f"best_sample_psnr={float(best_sample['psnr']):.6f}",
                f"best_sample_ssim={float(best_sample['ssim']):.6f}",
                f"best_sample_l1={float(best_sample['l1']):.6f}",
            ]
        )
    if worst_sample is not None:
        report_lines.extend(
            [
                f"worst_sample_index={worst_sample['index']}",
                f"worst_sample_dataset_index={worst_sample.get('dataset_index', -1)}",
                f"worst_sample_path={worst_sample['path']}",
                f"worst_sample_psnr={float(worst_sample['psnr']):.6f}",
                f"worst_sample_ssim={float(worst_sample['ssim']):.6f}",
                f"worst_sample_l1={float(worst_sample['l1']):.6f}",
            ]
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info(f"  Saved test report: {report_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "scale": scale,
        "avg_loss": avg_loss,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "report_path": report_path,
    }


def evaluate_models(args, logger) -> None:
    device = resolve_device(args.device)
    if args.all_models:
        scales = args.scales or ([args.scale] if args.scale else [2])
        logger.info("=" * 70)
        logger.info("  MULTI-MODEL EVALUATION MODE")
        logger.info(f"  Models : {', '.join(m.upper() for m in SUPPORTED_MODELS)}")
        logger.info(f"  Scales : {', '.join(f'x{s}' for s in scales)}")
        logger.info(f"  Device : {device}")
        logger.info("=" * 70)

        all_results = []
        for scale in scales:
            for model_name in SUPPORTED_MODELS:
                result = evaluate_single_model(
                    args=args,
                    model_name=model_name,
                    scale=scale,
                    device=device,
                    logger=logger,
                )
                if result is not None:
                    all_results.append(result)

        if not all_results:
            logger.error("No models were evaluated successfully. Check checkpoint availability.")
            return
        logger.info("")
        logger.info("All done! Per-model test reports:")
        for result in sorted(
            all_results,
            key=lambda kv: (kv["scale"], SUPPORTED_MODELS.index(kv["model_name"])),
        ):
            logger.info(
                f"  {result['model_name'].upper()} x{result['scale']} | "
                f"PSNR={result['avg_psnr']:.4f}, SSIM={result['avg_ssim']:.6f}, L1={result['avg_loss']:.6f}"
            )
            logger.info(f"    report: {result['report_path']}")
        logger.info(f"  Output root: {Path(args.save_results_dir).resolve()}")
        return

    model_name = args.model
    scale = args.scale
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint_path is not None:
        inferred_model, inferred_scale = infer_model_scale_from_checkpoint(checkpoint_path)
        if model_name is None and inferred_model in SUPPORTED_MODELS:
            model_name = inferred_model
        if scale is None and inferred_scale is not None:
            scale = inferred_scale

    if model_name is None or scale is None:
        raise ValueError("Please provide --model and --scale, or provide parseable --checkpoint.")
    if checkpoint_path is None:
        checkpoint_path = (Path("outputs/checkpoints") / f"{model_name}_x{scale}_best.pth").resolve()

    logger.info(f"Single-model evaluation: {model_name.upper()} x{scale}")
    result = evaluate_single_model(
        args=args,
        model_name=model_name,
        scale=scale,
        device=device,
        logger=logger,
        checkpoint_path=checkpoint_path,
    )
    if result is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
