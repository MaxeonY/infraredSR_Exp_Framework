from typing import Callable, Dict, List, Type

import torch.nn as nn


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str, model_cls: Type[nn.Module]) -> None:
    key = name.lower().strip()
    if not key:
        raise ValueError("Model name must be non-empty.")
    if key in MODEL_REGISTRY:
        raise KeyError(f"Model already registered: {key}")
    MODEL_REGISTRY[key] = model_cls


def get_model_class(name: str) -> Type[nn.Module]:
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unsupported model: {name}. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key]


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def build_registered(name: str, build_fn: Callable[[Type[nn.Module]], nn.Module]) -> nn.Module:
    model_cls = get_model_class(name)
    return build_fn(model_cls)
