from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.checkpoint import save_best_checkpoint, save_latest_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim


def prepare_model_input(model_name: str, lr: torch.Tensor, scale: int) -> torch.Tensor:
    if model_name.lower() in ("srcnn", "srcnn_arf"):
        return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    return lr


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def estimate_model_macs_flops(model: nn.Module, sample_input: torch.Tensor) -> Tuple[int, int]:
    total_macs = 0

    def conv2d_hook(module: nn.Conv2d, _, outputs):
        nonlocal total_macs
        output_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if not torch.is_tensor(output_tensor):
            return
        kernel_h, kernel_w = module.kernel_size
        in_channels_per_group = module.in_channels // module.groups
        kernel_mul = kernel_h * kernel_w * in_channels_per_group
        total_macs += int(output_tensor.numel() * kernel_mul)

    def linear_hook(module: nn.Linear, _, outputs):
        nonlocal total_macs
        output_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if not torch.is_tensor(output_tensor):
            return
        total_macs += int(output_tensor.numel() * module.in_features)

    handles = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(conv2d_hook))
        elif isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))

    was_training = model.training
    try:
        model.eval()
        _ = model(sample_input)
    finally:
        if was_training:
            model.train()
        for h in handles:
            h.remove()

    return total_macs, int(total_macs * 2)


def build_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def estimate_optimizer_flops_per_step(
    num_params: int,
    weight_decay: float,
) -> int:
    # Adam: ~14 fused ops per parameter per step; +2 when weight decay is on.
    ops_per_param = 14
    if weight_decay > 0:
        ops_per_param += 2
    return int(num_params * ops_per_param)


def train_one_epoch_gradient(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scale: int,
    flops_per_sample_forward: int,
    optimizer_flops_per_step: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = 0
    total_forward_evals = 0
    total_backward_evals = 0
    est_model_flops = 0.0
    est_optimizer_flops = 0.0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        batch_size = lr.size(0)
        model_input = prepare_model_input(model_name, lr, scale)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            sr = model(model_input)
            loss = criterion(sr, hr)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        num_batches += 1
        total_forward_evals += 1
        total_backward_evals += 1
        est_model_flops += float(flops_per_sample_forward * batch_size * 3)
        est_optimizer_flops += float(optimizer_flops_per_step)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, {
        "num_batches": float(num_batches),
        "num_samples": float(total_samples),
        "forward_evals": float(total_forward_evals),
        "backward_evals": float(total_backward_evals),
        "est_model_flops": float(est_model_flops),
        "est_optimizer_flops": float(est_optimizer_flops),
    }


@torch.no_grad()
def validate_one_epoch(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scale: int,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    for lr, hr, _ in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        model_input = prepare_model_input(model_name, lr, scale)
        with torch.amp.autocast("cuda", enabled=use_amp):
            sr = model(model_input)
            loss = criterion(sr, hr)

        total_loss += loss.item()
        psnr = calculate_psnr(sr.float().squeeze(0), hr.float().squeeze(0))
        ssim = calculate_ssim(sr.float().squeeze(0), hr.float().squeeze(0))
        total_psnr += psnr
        total_ssim += ssim
        total_samples += 1

    return (
        total_loss / max(total_samples, 1),
        total_psnr / max(total_samples, 1),
        total_ssim / max(total_samples, 1),
    )


def prepare_flops_profile_input(
    train_loader: DataLoader,
    model_name: str,
    scale: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    sample_lr, _ = train_loader.dataset[0]
    sample_lr = sample_lr.unsqueeze(0).to(device)
    model_input = prepare_model_input(model_name, sample_lr, scale)
    return model_input, tuple(sample_lr.shape), tuple(model_input.shape)


def build_profile_paths(args) -> Tuple[Path, Path]:
    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.run_tag.strip() if args.run_tag.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.model}_x{args.scale}_adam_{run_tag}"
    return profile_dir / f"{stem}.json", profile_dir / f"{stem}.csv"


def save_profile_csv(csv_path: Path, profile: Dict) -> None:
    epochs = profile.get("epochs", [])
    if not epochs:
        return
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_psnr",
        "val_ssim",
        "train_time_sec",
        "val_time_sec",
        "epoch_time_sec",
        "num_batches",
        "num_samples",
        "forward_evals",
        "backward_evals",
        "est_model_flops",
        "est_optimizer_flops",
        "est_total_flops",
    ]
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in epochs:
            writer.writerow(item)


def run_training(
    args,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger,
    model_kwargs: Dict[str, object],
) -> None:
    use_amp = (not args.no_amp) and device.type == "cuda"
    criterion = nn.L1Loss()
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_params, num_trainable_params = count_model_parameters(model)
    logger.info(f"Model params (all/trainable): {num_params}/{num_trainable_params}")

    profile_input, profile_lr_shape, profile_model_input_shape = prepare_flops_profile_input(
        train_loader=train_loader,
        model_name=args.model,
        scale=args.scale,
        device=device,
    )
    model_macs_per_forward_sample, model_flops_per_forward_sample = estimate_model_macs_flops(
        model, profile_input
    )
    logger.info(f"FLOPs profile LR shape   : {profile_lr_shape}")
    logger.info(f"FLOPs profile input shape: {profile_model_input_shape}")
    logger.info(
        "MACs/FLOPs per forward sample: "
        f"{model_macs_per_forward_sample}/{model_flops_per_forward_sample}"
    )

    optimizer_flops_per_step = estimate_optimizer_flops_per_step(
        num_params=num_trainable_params,
        weight_decay=args.weight_decay,
    )
    logger.info(f"Estimated optimizer FLOPs per step: {optimizer_flops_per_step}")

    best_psnr = float("-inf")
    best_ssim = float("-inf")
    total_train_time = 0.0
    total_val_time = 0.0
    total_forward_evals = 0.0
    total_backward_evals = 0.0
    total_est_model_flops = 0.0
    total_est_optimizer_flops = 0.0
    epoch_records = []
    run_start = perf_counter()

    model_meta = {
        "model_name": args.model.lower(),
        "scale": int(args.scale),
        "model_kwargs": dict(model_kwargs),
    }

    for epoch in range(1, args.epochs + 1):
        train_start = perf_counter()
        train_loss, train_stats = train_one_epoch_gradient(
            model_name=args.model,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scale=args.scale,
            flops_per_sample_forward=model_flops_per_forward_sample,
            optimizer_flops_per_step=optimizer_flops_per_step,
            scaler=scaler if use_amp else None,
            use_amp=use_amp,
        )

        train_time = perf_counter() - train_start
        total_train_time += train_time
        val_time = 0.0
        val_loss = float("nan")
        val_psnr = float("nan")
        val_ssim = float("nan")

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_start = perf_counter()
            val_loss, val_psnr, val_ssim = validate_one_epoch(
                model_name=args.model,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                scale=args.scale,
                use_amp=use_amp,
            )
            val_time = perf_counter() - val_start
            total_val_time += val_time
            logger.info(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val PSNR: {val_psnr:.4f} | "
                f"Val SSIM: {val_ssim:.6f} | "
                f"Train/Val Time: {train_time:.2f}s/{val_time:.2f}s"
            )

            save_latest_checkpoint(
                save_dir=args.save_dir,
                model_name=args.model,
                scale=args.scale,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_psnr,
                extra={
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "optimizer": "adam",
                },
                model_meta=model_meta,
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_ssim = val_ssim
                best_path = save_best_checkpoint(
                    save_dir=args.save_dir,
                    model_name=args.model,
                    scale=args.scale,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_psnr,
                    extra={
                        "val_loss": val_loss,
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "optimizer": "adam",
                    },
                    model_meta=model_meta,
                )
                logger.info(f"Best model updated. Saved to: {best_path}")
        else:
            logger.info(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"Train Loss: {train_loss:.6f} | (skip val) | Train Time: {train_time:.2f}s"
            )

        epoch_model_flops = float(train_stats["est_model_flops"])
        epoch_optimizer_flops = float(train_stats["est_optimizer_flops"])
        epoch_total_flops = epoch_model_flops + epoch_optimizer_flops

        total_forward_evals += float(train_stats["forward_evals"])
        total_backward_evals += float(train_stats["backward_evals"])
        total_est_model_flops += epoch_model_flops
        total_est_optimizer_flops += epoch_optimizer_flops

        epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim),
                "train_time_sec": float(train_time),
                "val_time_sec": float(val_time),
                "epoch_time_sec": float(train_time + val_time),
                "num_batches": int(train_stats["num_batches"]),
                "num_samples": int(train_stats["num_samples"]),
                "forward_evals": int(train_stats["forward_evals"]),
                "backward_evals": int(train_stats["backward_evals"]),
                "est_model_flops": epoch_model_flops,
                "est_optimizer_flops": epoch_optimizer_flops,
                "est_total_flops": epoch_total_flops,
            }
        )

    total_wall_time = perf_counter() - run_start
    summary = {
        "model": args.model,
        "scale": int(args.scale),
        "optimizer": "adam",
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "patch_size": int(args.patch_size),
        "device": str(device),
        "use_amp": bool(use_amp),
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable_params),
        "profile_lr_shape": list(profile_lr_shape),
        "profile_model_input_shape": list(profile_model_input_shape),
        "model_macs_per_forward_sample": int(model_macs_per_forward_sample),
        "model_flops_per_forward_sample": int(model_flops_per_forward_sample),
        "estimated_optimizer_flops_per_step": int(optimizer_flops_per_step),
        "total_train_time_sec": float(total_train_time),
        "total_val_time_sec": float(total_val_time),
        "total_wall_time_sec": float(total_wall_time),
        "avg_epoch_time_sec": float(total_wall_time / max(args.epochs, 1)),
        "total_forward_evals": int(total_forward_evals),
        "total_backward_evals": int(total_backward_evals),
        "total_est_model_flops": float(total_est_model_flops),
        "total_est_optimizer_flops": float(total_est_optimizer_flops),
        "total_est_flops": float(total_est_model_flops + total_est_optimizer_flops),
        "best_psnr": float(best_psnr),
        "best_ssim": float(best_ssim),
    }
    logger.info(
        "Training summary | "
        f"Total wall time: {summary['total_wall_time_sec']:.2f}s | "
        f"Best PSNR/SSIM: {summary['best_psnr']:.4f}/{summary['best_ssim']:.6f} | "
        f"Total est FLOPs: {summary['total_est_flops']:.3e}"
    )

    if not args.no_profile:
        import json

        json_path, csv_path = build_profile_paths(args)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "epochs": epoch_records,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        save_profile_csv(csv_path, payload)
        logger.info(f"Saved training profile JSON: {json_path}")
        logger.info(f"Saved training profile CSV : {csv_path}")

    logger.info("Training finished.")
