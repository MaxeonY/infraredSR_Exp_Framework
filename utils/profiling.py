import logging
from time import perf_counter
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)


def _normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _build_input(input_shape: Sequence[int], device: torch.device) -> torch.Tensor:
    if len(input_shape) != 4:
        raise ValueError(f"input_shape must be 4D, got: {input_shape}")
    return torch.randn(tuple(int(v) for v in input_shape), device=device, dtype=torch.float32)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_model_size_mb(model: nn.Module, bytes_per_param: int = 4) -> float:
    params = count_parameters(model)
    return float(params * float(bytes_per_param) / (1024.0 ** 2))


@torch.no_grad()
def profile_macs_and_flops(model: nn.Module, input_shape: Sequence[int], device) -> Dict[str, Optional[float]]:
    device_obj = _normalize_device(device)
    sample_input = _build_input(input_shape, device_obj)

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
        macs = float(total_macs)
        flops = float(total_macs * 2)
        return {
            "macs": macs,
            "gmacs": macs / 1e9,
            "flops": flops,
            "gflops": flops / 1e9,
        }
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()


@torch.no_grad()
def benchmark_inference_time(
    model: nn.Module,
    input_shape: Sequence[int],
    device,
    warmup: int = 20,
    repeat: int = 100,
) -> Dict[str, Optional[float]]:
    device_obj = _normalize_device(device)
    if repeat <= 0:
        raise ValueError("repeat must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    sample_input = _build_input(input_shape, device_obj)
    was_training = model.training

    try:
        model.eval()
        for _ in range(warmup):
            _ = model(sample_input)
            _sync_if_cuda(device_obj)

        latencies = []
        for _ in range(repeat):
            start = perf_counter()
            _ = model(sample_input)
            _sync_if_cuda(device_obj)
            elapsed_ms = (perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        arr = np.asarray(latencies, dtype=np.float64)
        avg_ms = float(np.mean(arr))
        med_ms = float(np.median(arr))
        p95_ms = float(np.percentile(arr, 95))
        fps = float(1000.0 / avg_ms) if avg_ms > 0 else None
        return {
            "latency_avg_ms": avg_ms,
            "latency_median_ms": med_ms,
            "latency_p95_ms": p95_ms,
            "fps": fps,
        }
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def measure_peak_gpu_memory(model: nn.Module, input_shape: Sequence[int], device) -> Optional[float]:
    device_obj = _normalize_device(device)
    if device_obj.type != "cuda":
        return None

    sample_input = _build_input(input_shape, device_obj)
    was_training = model.training
    try:
        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device_obj)
        _ = model(sample_input)
        torch.cuda.synchronize(device=device_obj)
        peak_mb = torch.cuda.max_memory_allocated(device=device_obj) / (1024.0 ** 2)
        return float(peak_mb)
    finally:
        if was_training:
            model.train()


def profile_model(
    model: nn.Module,
    input_shape: Sequence[int],
    device,
    warmup: int = 20,
    repeat: int = 100,
) -> Dict[str, Optional[float]]:
    params = count_parameters(model)
    output: Dict[str, Optional[float]] = {
        "params": int(params),
        "params_m": float(params / 1e6),
        "model_size_mb": estimate_model_size_mb(model),
        "macs": None,
        "gmacs": None,
        "flops": None,
        "gflops": None,
        "latency_avg_ms": None,
        "latency_median_ms": None,
        "latency_p95_ms": None,
        "fps": None,
        "peak_gpu_mem_mb": None,
    }

    try:
        output.update(profile_macs_and_flops(model, input_shape=input_shape, device=device))
    except Exception as exc:
        LOGGER.warning(f"Failed to profile MACs/FLOPs: {exc}")

    try:
        output.update(
            benchmark_inference_time(
                model,
                input_shape=input_shape,
                device=device,
                warmup=warmup,
                repeat=repeat,
            )
        )
    except Exception as exc:
        LOGGER.warning(f"Failed to benchmark inference runtime: {exc}")

    try:
        output["peak_gpu_mem_mb"] = measure_peak_gpu_memory(model, input_shape=input_shape, device=device)
    except Exception as exc:
        LOGGER.warning(f"Failed to measure peak GPU memory: {exc}")

    return output
