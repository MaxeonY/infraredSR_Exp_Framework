import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from datasets.builder import build_m3fd_dataset
from engine.trainer import run_training
from models import SUPPORTED_MODELS, build_model, merge_model_kwargs
from utils.logger import setup_logger
from utils.misc import deep_update, load_yaml
from utils.seed import seed_worker, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train infrared image super-resolution models.")
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS, help="Model name.")
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale factor.")
    parser.add_argument("--train_split", type=str, default="data/processed/train.txt", help="Train split txt path.")
    parser.add_argument("--val_split", type=str, default="data/processed/val.txt", help="Validation split txt path.")
    parser.add_argument("--patch_size", type=int, default=64, help="HR patch size for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic cuDNN mode.")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints", help="Checkpoint directory.")
    parser.add_argument("--log_file", type=str, default="outputs/logs/train.log", help="Log file path.")
    parser.add_argument("--profile_dir", type=str, default="outputs/results/training_profiles", help="Directory to save training profile JSON/CSV.")
    parser.add_argument("--run_tag", type=str, default="", help="Optional run tag for profile filename.")
    parser.add_argument("--no_profile", action="store_true", help="Disable profile JSON/CSV export.")

    parser.add_argument("--dataset_cfg", type=str, default="configs/dataset/m3fd.yaml", help="Dataset config YAML.")
    parser.add_argument("--train_cfg", type=str, default="", help="Optional train config YAML.")
    parser.add_argument("--model_cfg", type=str, default="", help="Optional model config YAML.")
    args = parser.parse_args()
    defaults = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        defaults[action.dest] = action.default
    setattr(args, "_defaults", defaults)
    return args


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _auto_merge_train_configs(args) -> Dict[str, object]:
    merged = {}
    default_cfg = Path("configs/train/default.yaml")
    model_cfg = Path(f"configs/train/{args.model.lower()}.yaml")
    if default_cfg.exists():
        deep_update(merged, load_yaml(str(default_cfg)))
    if model_cfg.exists():
        deep_update(merged, load_yaml(str(model_cfg)))
    if args.train_cfg:
        deep_update(merged, load_yaml(args.train_cfg))
    return merged


def _apply_train_cfg(args, train_cfg: Dict[str, object]) -> None:
    if not train_cfg:
        return
    defaults = getattr(args, "_defaults", {})
    for key, value in train_cfg.items():
        if hasattr(args, key) and key in defaults and getattr(args, key) == defaults[key]:
            setattr(args, key, value)


def build_dataloaders(args, dataset_cfg: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    degradation_cfg = dataset_cfg.get("degradation", {}) if isinstance(dataset_cfg, dict) else {}
    train_set = build_m3fd_dataset(
        split_file=args.train_split,
        scale=args.scale,
        patch_size=args.patch_size,
        mode="train",
        augment=False,
        degradation_cfg=degradation_cfg,
    )
    val_set = build_m3fd_dataset(
        split_file=args.val_split,
        scale=args.scale,
        patch_size=args.patch_size,
        mode="val",
        augment=False,
        degradation_cfg=degradation_cfg,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    # Adam is the only supported optimizer. We keep this attribute around so
    # downstream code (checkpoint meta, profile filenames, training summary)
    # can still tag runs consistently without re-wiring every call site.
    args.optimizer = "adam"

    train_cfg = _auto_merge_train_configs(args)
    _apply_train_cfg(args, train_cfg)
    dataset_cfg = load_yaml(args.dataset_cfg) if args.dataset_cfg and Path(args.dataset_cfg).exists() else {}

    model_cfg = {}
    auto_model_cfg_path = Path(f"configs/model/{args.model.lower()}.yaml")
    if auto_model_cfg_path.exists():
        deep_update(model_cfg, load_yaml(str(auto_model_cfg_path)))
    if args.model_cfg:
        deep_update(model_cfg, load_yaml(args.model_cfg))

    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    logger = setup_logger(name="train", log_file=args.log_file)
    logger.info(f"Model         : {args.model}")
    logger.info(f"Scale         : x{args.scale}")
    logger.info(f"Train split   : {args.train_split}")
    logger.info(f"Val split     : {args.val_split}")
    logger.info(f"Patch size    : {args.patch_size}")
    logger.info(f"Batch size    : {args.batch_size}")
    logger.info(f"Epochs        : {args.epochs}")
    logger.info(f"Learning rate : {args.lr}")
    logger.info(f"Optimizer     : adam")
    logger.info(f"Weight decay  : {args.weight_decay}")
    logger.info(f"Device        : {device}")

    train_loader, val_loader = build_dataloaders(args, dataset_cfg)
    logger.info(f"Train samples : {len(train_loader.dataset)}")
    logger.info(f"Val samples   : {len(val_loader.dataset)}")
    logger.info(f"AMP           : {'on' if ((not args.no_amp) and device.type == 'cuda') else 'off'}")
    logger.info(f"Deterministic : {'on' if args.deterministic else 'off'}")
    logger.info(f"Val interval  : {args.val_interval}")

    model_kwargs = merge_model_kwargs(args.model, config_kwargs=model_cfg)
    model = build_model(model_name=args.model, scale=args.scale, **model_kwargs).to(device)
    run_training(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()
