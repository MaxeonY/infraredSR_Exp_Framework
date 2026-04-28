import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

from models import SUPPORTED_MODELS
from utils.logger import setup_logger


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_COLORS = {
    "srcnn": "#e6194b",
    "srcnn_arf": "#ff6b6b",
    "fsrcnn": "#3cb44b",
    "edsr": "#4363d8",
    "edsr_arf": "#2648b5",
    "edsr_arfmk2": "#7b2fff",
    "rcan": "#f58231",
    "ldynsr": "#2b8a3e",
}


EXTENDED_METRIC_KEYS = ['mse', 'rmse', 'gradient_mae', 'laplacian_mae', 'fft_l1', 'hfen']
PROFILE_KEYS = [
    'params',
    'params_m',
    'model_size_mb',
    'macs',
    'gmacs',
    'flops',
    'gflops',
    'latency_avg_ms',
    'latency_median_ms',
    'latency_p95_ms',
    'fps',
    'peak_gpu_mem_mb',
]


def _color_for(model_name: str) -> str:
    return MODEL_COLORS.get(model_name.lower(), "#888888")


def parse_test_report(report_path: Path) -> Dict[str, str]:
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    data: Dict[str, str] = {}
    with report_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def _resolve_metrics_csv(report: Dict[str, str], report_path: Path) -> Optional[Path]:
    candidates: List[Path] = []
    raw = report.get("metrics_csv", "").strip()
    if raw and raw.lower() != "disabled":
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([Path.cwd() / p, report_path.parent / p, report_path.parent.parent / p])

    candidates.append(report_path.parent / "metrics" / "per_sample_metrics.csv")

    for p in candidates:
        resolved = p.resolve()
        if resolved.exists():
            return resolved
    return None


def _resolve_extended_summary_json(report_path: Path) -> Optional[Path]:
    candidates = list(report_path.parent.glob("*_extended_summary.json"))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def _parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "n/a", "na", "nan", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _load_sample_metrics(csv_path: Optional[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if csv_path is None or (not csv_path.exists()):
        return rows

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "index": int(row.get("index", 0)),
                        "dataset_index": int(row.get("dataset_index", row.get("index", 0))),
                        "path": row.get("path", row.get("filename", "")),
                        "l1": float(row.get("l1", 0.0)),
                        "psnr": float(row.get("psnr", 0.0)),
                        "ssim": float(row.get("ssim", 0.0)),
                        "mse": _parse_optional_float(row.get("mse")),
                        "rmse": _parse_optional_float(row.get("rmse")),
                        "gradient_mae": _parse_optional_float(row.get("gradient_mae")),
                        "laplacian_mae": _parse_optional_float(row.get("laplacian_mae")),
                        "fft_l1": _parse_optional_float(row.get("fft_l1")),
                        "hfen": _parse_optional_float(row.get("hfen")),
                    }
                )
            except Exception:
                continue
    return rows


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return safe[:120] if safe else "sample"


def _extract_cmp_stem(cmp_path: Path) -> str:
    m = re.match(r"^\d+_(.+)_cmp\.png$", cmp_path.name)
    if m:
        return m.group(1)
    return cmp_path.stem.replace("_cmp", "")


def _collect_cmp_images(run_root: Path) -> Dict[str, Path]:
    cmp_dir = run_root / "figures" / "sequential"
    if not cmp_dir.exists():
        return {}

    mapping: Dict[str, Path] = {}
    for p in sorted(cmp_dir.glob("*_cmp.png")):
        mapping[_extract_cmp_stem(p)] = p.resolve()
    return mapping


def _save_gallery(image_paths: List[Path], title: str, save_path: Path, max_images: int = 12) -> Optional[Path]:
    if len(image_paths) == 0:
        return None
    chosen = image_paths[: max(1, max_images)]
    cols = min(4, len(chosen))
    rows = (len(chosen) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    for idx, img_path in enumerate(chosen):
        ax = axes_list[idx]
        try:
            ax.imshow(plt.imread(str(img_path)))
            ax.set_title(_extract_cmp_stem(img_path), fontsize=8)
            ax.axis("off")
        except Exception:
            ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")
            ax.axis("off")

    for ax in axes_list[len(chosen):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_single_model_effect_gallery(result: Dict[str, Any], max_images: int, logger) -> Optional[Path]:
    cmp_map = _collect_cmp_images(result["run_root"])
    if len(cmp_map) == 0:
        return None

    save_path = result["run_root"] / "figures" / "effect_gallery.png"
    title = (
        f"{result['model_name'].upper()} x{result['scale']} | {result['scope_label']} | "
        f"PSNR {result['avg_psnr']:.2f}, SSIM {result['avg_ssim']:.4f}"
    )
    out = _save_gallery(list(cmp_map.values()), title=title, save_path=save_path, max_images=max_images)
    if out is not None:
        logger.info(f"  Saved single-model effect gallery: {out}")
    return out


def save_cross_model_effect_comparison(
    results_at_scale: List[Dict[str, Any]],
    scale: int,
    scope_label: str,
    out_dir: Path,
    max_samples: int,
    logger,
) -> None:
    if len(results_at_scale) < 2:
        return

    image_maps: Dict[str, Dict[str, Path]] = {
        r["model_name"]: _collect_cmp_images(r["run_root"]) for r in results_at_scale
    }
    valid = [r for r in results_at_scale if len(image_maps.get(r["model_name"], {})) > 0]
    if len(valid) < 2:
        logger.info(f"  [cross-model] Skip effect comparison at x{scale} ({scope_label}): no cmp images.")
        return

    common_stems = set(image_maps[valid[0]["model_name"]].keys())
    for r in valid[1:]:
        common_stems &= set(image_maps[r["model_name"]].keys())
    if len(common_stems) == 0:
        logger.info(f"  [cross-model] Skip effect comparison at x{scale} ({scope_label}): no common samples.")
        return

    stems = sorted(common_stems)[: max(1, max_samples)]
    effect_dir = out_dir / "effect_comparison"
    effect_dir.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems):
        fig, axes = plt.subplots(1, len(valid), figsize=(4.2 * len(valid), 4.2))
        if hasattr(axes, "ravel"):
            axes_list = list(axes.ravel())
        else:
            axes_list = [axes]

        for col, r in enumerate(valid):
            ax = axes_list[col]
            img_path = image_maps[r["model_name"]][stem]
            ax.imshow(plt.imread(str(img_path)))
            ax.set_title(f"{r['model_name'].upper()}\navgP {r['avg_psnr']:.2f}", fontsize=9)
            ax.axis("off")

        fig.suptitle(f"x{scale} | {scope_label} | {stem}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = effect_dir / f"{i:03d}_{_safe_name(stem)}_effect_cmp.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  [cross-model] Saved {len(stems)} effect comparison image(s): {effect_dir}")


def save_cross_scale_effect_comparison(
    results_for_model: List[Dict[str, Any]],
    model_name: str,
    scope_label: str,
    out_dir: Path,
    max_samples: int,
    logger,
) -> None:
    if len(results_for_model) < 2:
        return

    rows = sorted(results_for_model, key=lambda r: r["scale"])
    image_maps: Dict[int, Dict[str, Path]] = {
        int(r["scale"]): _collect_cmp_images(r["run_root"]) for r in rows
    }
    valid = [r for r in rows if len(image_maps.get(int(r["scale"]), {})) > 0]
    if len(valid) < 2:
        logger.info(f"  [cross-scale] Skip effect comparison for {model_name.upper()} ({scope_label}): no cmp images.")
        return

    common_stems = set(image_maps[int(valid[0]["scale"])].keys())
    for r in valid[1:]:
        common_stems &= set(image_maps[int(r["scale"])].keys())
    if len(common_stems) == 0:
        logger.info(f"  [cross-scale] Skip effect comparison for {model_name.upper()} ({scope_label}): no common samples.")
        return

    stems = sorted(common_stems)[: max(1, max_samples)]
    effect_dir = out_dir / "effect_comparison"
    effect_dir.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems):
        fig, axes = plt.subplots(1, len(valid), figsize=(4.2 * len(valid), 4.2))
        if hasattr(axes, "ravel"):
            axes_list = list(axes.ravel())
        else:
            axes_list = [axes]

        for col, r in enumerate(valid):
            ax = axes_list[col]
            img_path = image_maps[int(r["scale"])][stem]
            ax.imshow(plt.imread(str(img_path)))
            ax.set_title(f"x{int(r['scale'])}\navgP {r['avg_psnr']:.2f}", fontsize=9)
            ax.axis("off")

        fig.suptitle(f"{model_name.upper()} | {scope_label} | {stem}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = effect_dir / f"{i:03d}_{_safe_name(stem)}_effect_cmp.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  [cross-scale] Saved {len(stems)} effect comparison image(s): {effect_dir}")


def _infer_scope(report: Dict[str, str]) -> Tuple[str, str]:
    num_samples_raw = report.get("num_samples", "").strip()
    num_total_raw = report.get("num_total_samples", "").strip()
    selected_indices = report.get("selected_indices", "").strip().replace(" ", "")

    try:
        num_samples = int(num_samples_raw) if num_samples_raw else 0
    except Exception:
        num_samples = 0
    try:
        num_total = int(num_total_raw) if num_total_raw else 0
    except Exception:
        num_total = 0

    if num_total > 0 and num_samples == num_total:
        return "all_samples", "all samples"

    if num_samples == 1:
        idx_text = selected_indices.split(",")[0] if selected_indices else "unknown"
        try:
            idx_text = f"{int(idx_text):05d}"
        except Exception:
            pass
        return f"sample_{idx_text}", f"single sample ({idx_text})"

    if num_samples > 0:
        digest = hashlib.md5(selected_indices.encode("utf-8")).hexdigest()[:8] if selected_indices else f"n{num_samples}"
        return f"subset_{num_samples}_{digest}", f"subset ({num_samples} samples)"

    return "unknown_scope", "unknown scope"


def load_result_from_report(report_path: Path, logger=None) -> Optional[Dict[str, Any]]:
    try:
        report = parse_test_report(report_path)
        model = str(report["model"]).lower()
        scale = int(report["scale"])
        avg_psnr = float(report["avg_psnr"])
        avg_ssim = float(report["avg_ssim"])
        avg_l1 = float(report["avg_l1_loss"])
    except Exception as e:
        if logger is not None:
            logger.warning(f"Skip invalid report: {report_path} ({e})")
        return None

    metrics_csv = _resolve_metrics_csv(report, report_path)
    sample_metrics = _load_sample_metrics(metrics_csv)
    num_samples = int(report.get("num_samples", len(sample_metrics))) if report.get("num_samples") else len(sample_metrics)
    scope_key, scope_label = _infer_scope(report)

    quality_means = {
        "mse": _parse_optional_float(report.get("avg_mse")),
        "rmse": _parse_optional_float(report.get("avg_rmse")),
        "gradient_mae": _parse_optional_float(report.get("avg_gradient_mae")),
        "laplacian_mae": _parse_optional_float(report.get("avg_laplacian_mae")),
        "fft_l1": _parse_optional_float(report.get("avg_fft_l1")),
        "hfen": _parse_optional_float(report.get("avg_hfen")),
    }
    profile = {k: _parse_optional_float(report.get(k)) for k in PROFILE_KEYS}

    summary_json = _resolve_extended_summary_json(report_path)
    if summary_json is not None:
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            quality_payload = payload.get("quality_metrics", {})
            profile_payload = payload.get("profile", {})
            quality_means["mse"] = quality_means["mse"] if quality_means["mse"] is not None else _parse_optional_float(quality_payload.get("mse_mean"))
            quality_means["rmse"] = quality_means["rmse"] if quality_means["rmse"] is not None else _parse_optional_float(quality_payload.get("rmse_mean"))
            quality_means["gradient_mae"] = (
                quality_means["gradient_mae"]
                if quality_means["gradient_mae"] is not None
                else _parse_optional_float(quality_payload.get("gradient_mae_mean"))
            )
            quality_means["laplacian_mae"] = (
                quality_means["laplacian_mae"]
                if quality_means["laplacian_mae"] is not None
                else _parse_optional_float(quality_payload.get("laplacian_mae_mean"))
            )
            quality_means["fft_l1"] = quality_means["fft_l1"] if quality_means["fft_l1"] is not None else _parse_optional_float(quality_payload.get("fft_l1_mean"))
            quality_means["hfen"] = quality_means["hfen"] if quality_means["hfen"] is not None else _parse_optional_float(quality_payload.get("hfen_mean"))
            for key in PROFILE_KEYS:
                if profile[key] is None:
                    profile[key] = _parse_optional_float(profile_payload.get(key))
        except Exception:
            pass

    return {
        "label": f"{model}_x{scale}",
        "model_name": model,
        "scale": scale,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_loss": avg_l1,
        "num_samples": num_samples,
        "sample_metrics": sample_metrics,
        "scope_key": scope_key,
        "scope_label": scope_label,
        "report_path": report_path.resolve(),
        "run_root": report_path.resolve().parent,
        "mse": quality_means["mse"],
        "rmse": quality_means["rmse"],
        "gradient_mae": quality_means["gradient_mae"],
        "laplacian_mae": quality_means["laplacian_mae"],
        "fft_l1": quality_means["fft_l1"],
        "hfen": quality_means["hfen"],
        "profile": profile,
        "params_m": profile.get("params_m"),
        "model_size_mb": profile.get("model_size_mb"),
        "gmacs": profile.get("gmacs"),
        "gflops": profile.get("gflops"),
        "latency_avg_ms": profile.get("latency_avg_ms"),
        "latency_median_ms": profile.get("latency_median_ms"),
        "latency_p95_ms": profile.get("latency_p95_ms"),
        "fps": profile.get("fps"),
        "peak_gpu_mem_mb": profile.get("peak_gpu_mem_mb"),
    }


def discover_reports(save_results_dir: Path, latest_only: bool = True) -> List[Path]:
    if not save_results_dir.exists():
        return []

    report_paths = [p.resolve() for p in save_results_dir.rglob("*_test_report.txt")]
    if not latest_only:
        return sorted(report_paths, key=lambda p: p.stat().st_mtime)

    latest: Dict[Tuple[str, int, str], Path] = {}
    for rp in report_paths:
        try:
            report = parse_test_report(rp)
            scope_key, _ = _infer_scope(report)
            key = (str(report["model"]).lower(), int(report["scale"]), scope_key)
        except Exception:
            continue

        old = latest.get(key)
        if old is None or rp.stat().st_mtime > old.stat().st_mtime:
            latest[key] = rp

    return sorted(latest.values(), key=lambda p: p.name)


def save_metric_charts(sample_metrics: List[Dict[str, object]], metrics_dir: Path, model_name: str, scale: int) -> List[Path]:
    if len(sample_metrics) == 0:
        return []

    metrics_dir.mkdir(parents=True, exist_ok=True)
    psnr = np.array([float(m["psnr"]) for m in sample_metrics], dtype=np.float32)
    ssim = np.array([float(m["ssim"]) for m in sample_metrics], dtype=np.float32)
    l1 = np.array([float(m["l1"]) for m in sample_metrics], dtype=np.float32)
    idx = np.arange(1, len(sample_metrics) + 1, dtype=np.int32)

    saved: List[Path] = []

    curves_path = metrics_dir / "metric_curves.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(idx, psnr, color="#0068c9", linewidth=1.2)
    axes[0].set_ylabel("PSNR")
    axes[0].set_title(f"{model_name.upper()} x{scale} - Per-sample Metrics")
    axes[0].grid(alpha=0.25)
    axes[1].plot(idx, ssim, color="#09814a", linewidth=1.2)
    axes[1].set_ylabel("SSIM")
    axes[1].grid(alpha=0.25)
    axes[2].plot(idx, l1, color="#c94f00", linewidth=1.2)
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("L1")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(curves_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(curves_path)

    hist_path = metrics_dir / "metric_histograms.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(psnr, bins=30, color="#0068c9", alpha=0.85)
    axes[0].set_title("PSNR Histogram")
    axes[0].grid(alpha=0.2)
    axes[1].hist(ssim, bins=30, color="#09814a", alpha=0.85)
    axes[1].set_title("SSIM Histogram")
    axes[1].grid(alpha=0.2)
    axes[2].hist(l1, bins=30, color="#c94f00", alpha=0.85)
    axes[2].set_title("L1 Histogram")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(hist_path)

    scatter_path = metrics_dir / "psnr_ssim_scatter.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(psnr, ssim, c=l1, cmap="viridis", s=20, alpha=0.85)
    ax.set_title("PSNR vs SSIM (colored by L1)")
    ax.set_xlabel("PSNR")
    ax.set_ylabel("SSIM")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax).set_label("L1")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(scatter_path)

    return saved

def save_horizontal_comparison(current_result: Dict[str, object], reference_results: List[Dict[str, object]], out_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    rows: List[Dict[str, object]] = [current_result]
    rows.extend(reference_results)

    dedup: Dict[str, Dict[str, object]] = {}
    for row in rows:
        label = str(row["label"])
        if label not in dedup:
            dedup[label] = row
    rows = list(dedup.values())

    if len(rows) < 2:
        return None, None

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "horizontal_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "avg_psnr", "avg_ssim", "avg_l1_loss", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "avg_psnr": f"{float(row['avg_psnr']):.6f}",
                    "avg_ssim": f"{float(row['avg_ssim']):.6f}",
                    "avg_l1_loss": f"{float(row['avg_l1_loss']):.6f}",
                    "source": row.get("source", "unknown"),
                }
            )

    labels = [str(r["label"]) for r in rows]
    psnr = np.array([float(r["avg_psnr"]) for r in rows], dtype=np.float32)
    ssim = np.array([float(r["avg_ssim"]) for r in rows], dtype=np.float32)
    l1 = np.array([float(r["avg_l1_loss"]) for r in rows], dtype=np.float32)
    x = np.arange(len(rows))

    fig_path = out_dir / "horizontal_comparison.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].bar(x, psnr, color="#0068c9", alpha=0.9)
    axes[0].set_ylabel("PSNR")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, ssim, color="#09814a", alpha=0.9)
    axes[1].set_ylabel("SSIM")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, l1, color="#c94f00", alpha=0.9)
    axes[2].set_ylabel("L1")
    axes[2].set_xlabel("Model")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return csv_path, fig_path


def save_cross_model_comparison(
    results_at_scale: List[Dict[str, Any]],
    scale: int,
    scope_label: str,
    out_dir: Path,
    logger,
) -> None:
    if len(results_at_scale) < 2:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results_at_scale, key=lambda r: SUPPORTED_MODELS.index(r["model_name"]) if r["model_name"] in SUPPORTED_MODELS else 999)

    labels = [r["model_name"].upper() for r in rows]
    psnr = np.array([r["avg_psnr"] for r in rows], dtype=np.float32)
    ssim = np.array([r["avg_ssim"] for r in rows], dtype=np.float32)
    l1 = np.array([r["avg_loss"] for r in rows], dtype=np.float32)
    colors = [_color_for(r["model_name"]) for r in rows]
    x = np.arange(len(rows))

    csv_path = out_dir / f"cross_model_x{scale}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scale", "avg_psnr", "avg_ssim", "avg_l1"])
        for r in rows:
            writer.writerow([r["model_name"], scale, f"{r['avg_psnr']:.4f}", f"{r['avg_ssim']:.6f}", f"{r['avg_loss']:.6f}"])
    logger.info(f"  [cross-model] Saved CSV: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, psnr, color=colors, alpha=0.9)
    axes[0].set_ylabel("PSNR")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, ssim, color=colors, alpha=0.9)
    axes[1].set_ylabel("SSIM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, l1, color=colors, alpha=0.9)
    axes[2].set_ylabel("L1")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Cross-model comparison at x{scale} ({scope_label})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = out_dir / f"cross_model_x{scale}_bar.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [cross-model] Saved chart: {fig_path}")


def save_cross_scale_comparison(
    results_for_model: List[Dict[str, Any]],
    model_name: str,
    scope_label: str,
    out_dir: Path,
    logger,
) -> None:
    if len(results_for_model) < 2:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results_for_model, key=lambda r: r["scale"])

    labels = [f"x{r['scale']}" for r in rows]
    psnr = np.array([r["avg_psnr"] for r in rows], dtype=np.float32)
    ssim = np.array([r["avg_ssim"] for r in rows], dtype=np.float32)
    l1 = np.array([r["avg_loss"] for r in rows], dtype=np.float32)
    x = np.arange(len(rows))

    csv_path = out_dir / f"cross_scale_{model_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scale", "avg_psnr", "avg_ssim", "avg_l1"])
        for r in rows:
            writer.writerow([model_name, r["scale"], f"{r['avg_psnr']:.4f}", f"{r['avg_ssim']:.6f}", f"{r['avg_loss']:.6f}"])
    logger.info(f"  [cross-scale] Saved CSV: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    color = _color_for(model_name)
    axes[0].bar(x, psnr, color=color, alpha=0.9)
    axes[0].set_ylabel("PSNR")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, ssim, color=color, alpha=0.9)
    axes[1].set_ylabel("SSIM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, l1, color=color, alpha=0.9)
    axes[2].set_ylabel("L1")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Cross-scale comparison for {model_name.upper()} ({scope_label})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = out_dir / f"cross_scale_{model_name}_bar.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [cross-scale] Saved chart: {fig_path}")


def _save_scatter_plot(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    logger,
) -> None:
    valid = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in valid:
        ax.scatter(float(r[x_key]), float(r[y_key]), color=_color_for(r["model_name"]), s=70, alpha=0.9)
        ax.text(float(r[x_key]), float(r[y_key]), r["model_name"].upper(), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved extended chart: {path}")


def _save_bar_metric_plot(
    rows: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    path: Path,
    logger,
) -> None:
    valid = [r for r in rows if r.get(metric_key) is not None]
    if len(valid) == 0:
        return

    valid = sorted(valid, key=lambda r: float(r[metric_key]))
    labels = [f"{r['model_name'].upper()}_x{r['scale']}" for r in valid]
    values = np.array([float(r[metric_key]) for r in valid], dtype=np.float32)
    colors = [_color_for(r["model_name"]) for r in valid]
    x = np.arange(len(valid))

    fig, ax = plt.subplots(figsize=(max(10, len(valid) * 1.2), 5.8))
    ax.bar(x, values, color=colors, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved extended chart: {path}")


def _save_quality_efficiency_pareto(rows: List[Dict[str, Any]], path: Path, logger) -> None:
    x_key = "latency_avg_ms" if any(r.get("latency_avg_ms") is not None for r in rows) else "gmacs"
    x_label = "Latency Avg (ms)" if x_key == "latency_avg_ms" else "GMACs"
    valid = [r for r in rows if r.get(x_key) is not None and r.get("avg_psnr") is not None]
    if len(valid) == 0:
        return

    bubble_raw = np.array([float(r["params_m"]) if r.get("params_m") is not None else 1.0 for r in valid], dtype=np.float32)
    bubble = np.clip(bubble_raw * 40.0, 40.0, 800.0)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, r in enumerate(valid):
        x = float(r[x_key])
        y = float(r["avg_psnr"])
        ax.scatter(x, y, s=float(bubble[i]), color=_color_for(r["model_name"]), alpha=0.45, edgecolors="black", linewidths=0.6)
        ax.text(x, y, r["model_name"].upper(), fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel("PSNR")
    ax.set_title("Quality-Efficiency Pareto")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved extended chart: {path}")


def save_all_models_summary(results: List[Dict[str, Any]], out_dir: Path, logger) -> None:
    if not results:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda r: (r["scale"], SUPPORTED_MODELS.index(r["model_name"]) if r["model_name"] in SUPPORTED_MODELS else 999))

    csv_path = out_dir / "all_models_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scale", "avg_psnr", "avg_ssim", "avg_l1", "num_samples"])
        for r in rows:
            writer.writerow([r["model_name"], r["scale"], f"{r['avg_psnr']:.4f}", f"{r['avg_ssim']:.6f}", f"{r['avg_loss']:.6f}", r["num_samples"]])
    logger.info(f"  Saved summary CSV: {csv_path}")

    ext_csv_path = out_dir / "all_models_extended_summary.csv"
    with ext_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "scale",
                "num_samples",
                "avg_psnr",
                "avg_ssim",
                "avg_l1",
                "mse",
                "rmse",
                "gradient_mae",
                "laplacian_mae",
                "fft_l1",
                "hfen",
                "params_m",
                "model_size_mb",
                "gmacs",
                "gflops",
                "latency_avg_ms",
                "latency_median_ms",
                "latency_p95_ms",
                "fps",
                "peak_gpu_mem_mb",
                "report_path",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["model_name"],
                    r["scale"],
                    r["num_samples"],
                    r["avg_psnr"],
                    r["avg_ssim"],
                    r["avg_loss"],
                    r.get("mse"),
                    r.get("rmse"),
                    r.get("gradient_mae"),
                    r.get("laplacian_mae"),
                    r.get("fft_l1"),
                    r.get("hfen"),
                    r.get("params_m"),
                    r.get("model_size_mb"),
                    r.get("gmacs"),
                    r.get("gflops"),
                    r.get("latency_avg_ms"),
                    r.get("latency_median_ms"),
                    r.get("latency_p95_ms"),
                    r.get("fps"),
                    r.get("peak_gpu_mem_mb"),
                    str(r["report_path"]),
                ]
            )
    logger.info(f"  Saved extended summary CSV: {ext_csv_path}")

    labels = [f"{r['model_name'].upper()}_x{r['scale']}" for r in rows]
    psnr = np.array([r["avg_psnr"] for r in rows], dtype=np.float32)
    ssim = np.array([r["avg_ssim"] for r in rows], dtype=np.float32)
    l1 = np.array([r["avg_loss"] for r in rows], dtype=np.float32)
    colors = [_color_for(r["model_name"]) for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(rows) * 1.8), 12), sharex=True)
    axes[0].bar(x, psnr, color=colors, alpha=0.9)
    axes[0].set_ylabel("PSNR")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, ssim, color=colors, alpha=0.9)
    axes[1].set_ylabel("SSIM")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, l1, color=colors, alpha=0.9)
    axes[2].set_ylabel("L1")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("All Models Test Summary", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = out_dir / "all_models_summary.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved summary chart: {fig_path}")

    _save_scatter_plot(rows, "params_m", "avg_psnr", out_dir / "psnr_vs_params.png", "Params (M)", "PSNR", logger)
    _save_scatter_plot(rows, "gmacs", "avg_psnr", out_dir / "psnr_vs_gmacs.png", "GMACs", "PSNR", logger)
    _save_scatter_plot(rows, "latency_avg_ms", "avg_psnr", out_dir / "psnr_vs_latency.png", "Latency Avg (ms)", "PSNR", logger)
    _save_scatter_plot(rows, "latency_avg_ms", "avg_ssim", out_dir / "ssim_vs_latency.png", "Latency Avg (ms)", "SSIM", logger)
    _save_quality_efficiency_pareto(rows, out_dir / "quality_efficiency_pareto.png", logger)
    _save_bar_metric_plot(rows, "gradient_mae", "Gradient MAE Comparison", "Gradient MAE", out_dir / "gradient_mae_comparison.png", logger)
    _save_bar_metric_plot(rows, "laplacian_mae", "Laplacian MAE Comparison", "Laplacian MAE", out_dir / "laplacian_mae_comparison.png", logger)
    _save_bar_metric_plot(rows, "fft_l1", "FFT L1 Comparison", "FFT L1", out_dir / "fft_l1_comparison.png", logger)
    _save_bar_metric_plot(rows, "hfen", "HFEN Comparison", "HFEN", out_dir / "hfen_comparison.png", logger)


def collect_reference_reports(
    args,
    current_model: str,
    current_scale: int,
    current_scope_key: str,
    target_report: Path,
) -> List[Path]:
    refs: List[Path] = [Path(p).resolve() for p in args.compare_reports]
    root = Path(args.save_results_dir)

    for model_name in args.compare_with_models:
        if model_name == current_model:
            continue
        latest_match: Optional[Path] = None
        if root.exists():
            for rp in root.rglob("*_test_report.txt"):
                rp = rp.resolve()
                try:
                    parsed = parse_test_report(rp)
                    model = str(parsed["model"]).lower()
                    scale = int(parsed["scale"])
                    scope_key, _ = _infer_scope(parsed)
                except Exception:
                    continue
                if model != model_name or scale != current_scale or scope_key != current_scope_key:
                    continue
                if latest_match is None or rp.stat().st_mtime > latest_match.stat().st_mtime:
                    latest_match = rp
        if latest_match is not None:
            refs.append(latest_match)

    if args.quick_compare:
        latest_by_model: Dict[str, Path] = {}
        for rp in root.rglob("*_test_report.txt"):
            rp = rp.resolve()
            if rp == target_report.resolve():
                continue
            try:
                parsed = parse_test_report(rp)
                model = str(parsed["model"]).lower()
                scale = int(parsed["scale"])
                scope_key, _ = _infer_scope(parsed)
            except Exception:
                continue
            if model == current_model or scale != current_scale or scope_key != current_scope_key:
                continue
            old = latest_by_model.get(model)
            if old is None or rp.stat().st_mtime > old.stat().st_mtime:
                latest_by_model[model] = rp
        refs.extend(list(latest_by_model.values()))

    dedup: Dict[str, Path] = {}
    for rp in refs:
        dedup[str(rp)] = rp
    return list(dedup.values())

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison charts from test reports.")
    parser.add_argument("--save_results_dir", type=str, default="outputs/results", help="Root directory for test results.")
    parser.add_argument("--reports", type=str, nargs="*", default=[], help="Explicit *_test_report.txt files.")
    parser.add_argument("--all_reports", action="store_true", help="Use all discovered reports (default: latest per model-scale-scope).")
    parser.add_argument("--comparison_dir", type=str, default=None, help="Directory for cross-model/cross-scale/summary outputs.")

    parser.add_argument("--target_report", type=str, default=None, help="Target report for horizontal comparison.")
    parser.add_argument("--compare_reports", type=str, nargs="*", default=[], help="Reference report files for horizontal comparison.")
    parser.add_argument("--compare_with_models", type=str, nargs="*", default=[], choices=SUPPORTED_MODELS, help="Shortcut model names for horizontal comparison.")
    parser.add_argument("--quick_compare", action="store_true", help="Auto collect same-scale reports for horizontal comparison.")

    parser.add_argument("--no_metric_plots", action="store_true", help="Disable per-model metric chart generation.")
    parser.add_argument("--no_effect_comparison", action="store_true", help="Disable visual effect comparison generation.")
    parser.add_argument("--max_effect_visuals", type=int, default=10, help="Maximum number of effect comparison samples.")
    parser.add_argument("--no_horizontal", action="store_true", help="Disable horizontal comparison generation.")
    parser.add_argument("--no_cross_model", action="store_true", help="Disable cross-model comparison generation.")
    parser.add_argument("--no_cross_scale", action="store_true", help="Disable cross-scale comparison generation.")
    parser.add_argument("--no_summary", action="store_true", help="Disable all-model summary generation.")
    parser.add_argument("--log_file", type=str, default="outputs/logs/compare_results.log", help="Log file path.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="compare_results", log_file=args.log_file)
    if args.max_effect_visuals <= 0:
        raise ValueError("--max_effect_visuals must be > 0")

    if args.reports:
        report_paths = [Path(p).resolve() for p in args.reports]
    else:
        report_paths = discover_reports(Path(args.save_results_dir), latest_only=not args.all_reports)

    if len(report_paths) == 0:
        raise FileNotFoundError("No *_test_report.txt found. Run test.py first or pass --reports.")

    logger.info(f"Discovered {len(report_paths)} report(s).")

    results: List[Dict[str, Any]] = []
    for rp in report_paths:
        item = load_result_from_report(rp, logger=logger)
        if item is not None:
            results.append(item)

    if len(results) == 0:
        raise RuntimeError("No valid reports available for comparison.")

    if not args.no_metric_plots:
        logger.info("Generating per-model metric plots ...")
        for r in results:
            chart_paths = save_metric_charts(r["sample_metrics"], r["run_root"] / "metrics", r["model_name"], r["scale"])
            if len(chart_paths) == 0:
                logger.info(f"  Skip metric plots: {r['label']} (no per-sample metrics)")
            else:
                for p in chart_paths:
                    logger.info(f"  Saved metric chart: {p}")

    if not args.no_effect_comparison:
        logger.info("Generating single-model effect galleries ...")
        for r in results:
            save_single_model_effect_gallery(
                r,
                max_images=max(1, args.max_effect_visuals),
                logger=logger,
            )

    comparison_dir = Path(args.comparison_dir).resolve() if args.comparison_dir else (Path(args.save_results_dir) / "comparison").resolve()

    if not args.no_cross_model:
        logger.info("Generating cross-model comparisons ...")
        grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for r in results:
            grouped.setdefault((r["scale"], r["scope_key"]), []).append(r)

        for (scale, scope_key), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if len(rows) >= 2:
                scope_label = rows[0]["scope_label"]
                logger.info(f"  [cross-model] x{scale} | {scope_label}")
                save_cross_model_comparison(
                    rows,
                    scale,
                    scope_label,
                    comparison_dir / f"cross_model_x{scale}_{scope_key}",
                    logger,
                )
                if not args.no_effect_comparison:
                    save_cross_model_effect_comparison(
                        rows,
                        scale,
                        scope_label,
                        comparison_dir / f"cross_model_x{scale}_{scope_key}",
                        max_samples=max(1, args.max_effect_visuals),
                        logger=logger,
                    )

    if not args.no_cross_scale:
        logger.info("Generating cross-scale comparisons ...")
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in results:
            grouped.setdefault((r["model_name"], r["scope_key"]), []).append(r)

        for (model_name, scope_key), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if len(rows) >= 2:
                scope_label = rows[0]["scope_label"]
                logger.info(f"  [cross-scale] {model_name.upper()} | {scope_label}")
                save_cross_scale_comparison(
                    rows,
                    model_name,
                    scope_label,
                    comparison_dir / f"cross_scale_{model_name}_{scope_key}",
                    logger,
                )
                if not args.no_effect_comparison:
                    save_cross_scale_effect_comparison(
                        rows,
                        model_name,
                        scope_label,
                        comparison_dir / f"cross_scale_{model_name}_{scope_key}",
                        max_samples=max(1, args.max_effect_visuals),
                        logger=logger,
                    )

    if not args.no_summary:
        logger.info("Generating overall summary ...")
        save_all_models_summary(results, comparison_dir, logger)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            grouped.setdefault(r["scope_key"], []).append(r)
        for scope_key, rows in grouped.items():
            if len(rows) >= 2:
                save_all_models_summary(rows, comparison_dir / f"summary_{scope_key}", logger)

    if not args.no_horizontal and args.target_report is not None:
        target_path = Path(args.target_report).resolve()
        target = load_result_from_report(target_path, logger=logger)
        if target is not None:
            refs = collect_reference_reports(
                args,
                target["model_name"],
                target["scale"],
                target["scope_key"],
                target_path,
            )
            ref_rows: List[Dict[str, object]] = []
            for rp in refs:
                rr = load_result_from_report(rp, logger=logger)
                if rr is None:
                    continue
                ref_rows.append(
                    {
                        "label": rr["label"],
                        "avg_psnr": rr["avg_psnr"],
                        "avg_ssim": rr["avg_ssim"],
                        "avg_l1_loss": rr["avg_loss"],
                        "source": str(rr["report_path"]),
                    }
                )

            current = {
                "label": target["label"],
                "avg_psnr": target["avg_psnr"],
                "avg_ssim": target["avg_ssim"],
                "avg_l1_loss": target["avg_loss"],
                "source": "current",
            }
            comp_csv, comp_fig = save_horizontal_comparison(current, ref_rows, target["run_root"] / "metrics")
            if comp_csv is not None and comp_fig is not None:
                logger.info(f"Saved horizontal comparison table: {comp_csv}")
                logger.info(f"Saved horizontal comparison figure: {comp_fig}")
            else:
                logger.info("Horizontal comparison skipped: need at least one valid extra report.")
    elif (len(args.compare_reports) > 0 or len(args.compare_with_models) > 0 or args.quick_compare) and not args.no_horizontal:
        logger.warning("compare_* options were provided but --target_report is missing. Horizontal comparison skipped.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
