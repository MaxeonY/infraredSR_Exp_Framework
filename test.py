import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from engine.evaluator import evaluate_models, infer_model_scale_from_checkpoint
from models import SUPPORTED_MODELS
from utils.logger import setup_logger
from utils.misc import load_yaml


@dataclass
class TrainingRun:
    start_time: datetime
    model: str
    scale: int = 2
    completed: bool = False
    best_psnr: float = float("-inf")
    best_ssim: float = float("-inf")
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0


LINE_MODEL = re.compile(r"^\[(?P<ts>[\d\-: ]+)\]\s+\[INFO\]\s+Model\s+:\s+(?P<model>\w+)\s*$")
LINE_SCALE = re.compile(r"^\[[\d\-: ]+\]\s+\[INFO\]\s+Scale\s+:\s+x(?P<scale>\d+)\s*$")
LINE_EPOCH = re.compile(
    r"^\[[\d\-: ]+\]\s+\[INFO\]\s+Epoch\s+\[(?P<epoch>\d+)/(?P<total>\d+)\]\s+"
    r"Train Loss:\s+(?P<train_loss>[-+eE.\d]+)\s+\|\s+"
    r"Val Loss:\s+(?P<val_loss>[-+eE.\d]+)\s+\|\s+"
    r"Val PSNR:\s+(?P<psnr>[-+eE.\d]+)\s+\|\s+"
    r"Val SSIM:\s+(?P<ssim>[-+eE.\d]+)\s*$"
)
LINE_FINISH = re.compile(r"^\[[\d\-: ]+\]\s+\[INFO\]\s+Training finished\.\s*$")


def parse_train_log(log_path: Path) -> List[TrainingRun]:
    if not log_path.exists():
        raise FileNotFoundError(f"Train log not found: {log_path}")

    runs: List[TrainingRun] = []
    current: Optional[TrainingRun] = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_model = LINE_MODEL.match(line)
            if m_model:
                if current is not None:
                    runs.append(current)
                ts = datetime.strptime(m_model.group("ts"), "%Y-%m-%d %H:%M:%S")
                current = TrainingRun(start_time=ts, model=m_model.group("model").lower())
                continue
            if current is None:
                continue

            m_scale = LINE_SCALE.match(line)
            if m_scale:
                current.scale = int(m_scale.group("scale"))
                continue

            m_epoch = LINE_EPOCH.match(line)
            if m_epoch:
                epoch = int(m_epoch.group("epoch"))
                total = int(m_epoch.group("total"))
                val_loss = float(m_epoch.group("val_loss"))
                psnr = float(m_epoch.group("psnr"))
                ssim = float(m_epoch.group("ssim"))
                current.total_epochs = total
                if psnr > current.best_psnr:
                    current.best_psnr = psnr
                    current.best_ssim = ssim
                    current.best_val_loss = val_loss
                    current.best_epoch = epoch
                continue

            if LINE_FINISH.match(line):
                current.completed = True
                continue

    if current is not None:
        runs.append(current)
    return runs


def choose_best_completed_run(runs: List[TrainingRun]) -> TrainingRun:
    completed = [r for r in runs if r.completed and r.model in SUPPORTED_MODELS and r.best_epoch > 0]
    if not completed:
        raise RuntimeError("No completed training runs with valid metrics found in train log.")

    latest_by_model_scale = {}
    for run in completed:
        latest_by_model_scale[(run.model, run.scale)] = run
    return max(latest_by_model_scale.values(), key=lambda r: r.best_psnr)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained infrared SR checkpoints on test split.")
    parser.add_argument("--model", type=str, default=None, choices=SUPPORTED_MODELS, help="Model name (single-model mode).")
    parser.add_argument("--scale", type=int, default=None, help="Upscale factor (single-model mode).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. If omitted, use model+scale best.")
    parser.add_argument("--all_models", action="store_true", help="Test ALL supported models.")
    parser.add_argument("--scales", type=int, nargs="+", default=None, help="Scale factors in multi-model mode.")
    parser.add_argument("--test_split", type=str, default="data/processed/test.txt", help="Test split file path.")
    parser.add_argument("--dataset_cfg", type=str, default="configs/dataset/m3fd.yaml", help="Dataset config YAML.")
    parser.add_argument("--patch_size", type=int, default=64, help="Unused in test mode, kept for interface consistency.")
    parser.add_argument("--batch_size", type=int, default=1, help="Test batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--sample_index", type=int, default=None, help="Evaluate one sample by index in test split.")
    parser.add_argument("--sample_path", type=str, default=None, help="Evaluate one sample by path/name/stem in test split.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Limit tested samples after filtering.")
    parser.add_argument("--auto_from_log", action="store_true", help="Auto-pick best completed run from train log.")
    parser.add_argument("--train_log", type=str, default="outputs/logs/train.log", help="Train log path.")
    parser.add_argument("--save_results_dir", type=str, default="outputs/results", help="Directory for eval outputs.")
    parser.add_argument("--save_visuals", action="store_true", help="Legacy flag. Visuals are enabled unless --no_visuals is set.")
    parser.add_argument("--no_visuals", action="store_true", help="Disable all visualization outputs.")
    parser.add_argument("--max_visuals", type=int, default=20, help="Maximum number of sequential samples to visualize.")
    parser.add_argument("--rank_visuals_k", type=int, default=10, help="Top/Bottom-K samples for ranked visualization.")
    parser.add_argument("--no_rank_visuals", action="store_true", help="Disable best/worst sample visualizations.")
    parser.add_argument("--log_file", type=str, default="outputs/logs/test.log", help="Test log path.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="test", log_file=args.log_file)
    if args.max_test_samples is not None and args.max_test_samples <= 0:
        raise ValueError("--max_test_samples must be > 0")

    dataset_cfg = load_yaml(args.dataset_cfg) if args.dataset_cfg and Path(args.dataset_cfg).exists() else {}
    args.degradation_cfg = dataset_cfg.get("degradation", dataset_cfg) if isinstance(dataset_cfg, dict) else {}

    if args.checkpoint and (args.model is None or args.scale is None):
        inferred_model, inferred_scale = infer_model_scale_from_checkpoint(Path(args.checkpoint))
        if args.model is None and inferred_model in SUPPORTED_MODELS:
            args.model = inferred_model
        if args.scale is None and inferred_scale is not None:
            args.scale = inferred_scale

    if args.auto_from_log and (args.model is None or args.scale is None):
        runs = parse_train_log(Path(args.train_log))
        best_run = choose_best_completed_run(runs)
        args.model = best_run.model
        args.scale = best_run.scale
        logger.info(
            f"Auto selected run from log: {best_run.model} x{best_run.scale}, "
            f"best PSNR={best_run.best_psnr:.4f}, SSIM={best_run.best_ssim:.6f}, "
            f"epoch={best_run.best_epoch}/{best_run.total_epochs}, start={best_run.start_time}"
        )

    sample_scope = "single sample" if (args.sample_index is not None or args.sample_path is not None) else (
        f"subset first {args.max_test_samples} samples" if args.max_test_samples is not None else "all samples"
    )
    logger.info(f"Evaluation sample scope: {sample_scope}")
    if args.sample_index is not None:
        logger.info(f"  sample_index={args.sample_index}")
    if args.sample_path is not None:
        logger.info(f"  sample_path={args.sample_path}")

    evaluate_models(args=args, logger=logger)


if __name__ == "__main__":
    main()
