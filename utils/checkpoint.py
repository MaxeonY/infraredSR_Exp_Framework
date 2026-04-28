from pathlib import Path
from typing import Any, Dict, Optional

import torch


def ensure_dir(save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> None:
    save_path_obj = Path(save_path)
    ensure_dir(save_path_obj)

    checkpoint: Dict[str, Any] = {"model_state_dict": model.state_dict()}

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    if extra is not None:
        checkpoint["extra"] = extra
    if model_meta is not None:
        checkpoint["model_meta"] = model_meta

    torch.save(checkpoint, save_path_obj)


def read_checkpoint(
    checkpoint_path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    return torch.load(path, map_location=map_location)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    checkpoint = read_checkpoint(checkpoint_path=checkpoint_path, map_location=map_location)

    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint missing key: model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_best_checkpoint(
    save_dir: str,
    model_name: str,
    scale: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> str:
    save_dir_obj = Path(save_dir)
    save_dir_obj.mkdir(parents=True, exist_ok=True)
    save_path = save_dir_obj / f"{model_name}_x{scale}_best.pth"

    save_checkpoint(
        save_path=str(save_path),
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_metric=best_metric,
        extra=extra,
        model_meta=model_meta,
    )
    return str(save_path)


def save_latest_checkpoint(
    save_dir: str,
    model_name: str,
    scale: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> str:
    save_dir_obj = Path(save_dir)
    save_dir_obj.mkdir(parents=True, exist_ok=True)
    save_path = save_dir_obj / f"{model_name}_x{scale}_latest.pth"

    save_checkpoint(
        save_path=str(save_path),
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_metric=best_metric,
        extra=extra,
        model_meta=model_meta,
    )
    return str(save_path)
