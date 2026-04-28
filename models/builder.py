from typing import Any, Dict, Mapping, Optional

from models.edsr import EDSR
from models.edsr_arf import EDSR_ARF
from models.edsr_arfmk2 import EDSR_ARFMk2
from models.fsrcnn import FSRCNN
from models.ldynsr import LDynSR
from models.rcan import RCAN
from models.registry import build_registered, list_models, register_model
from models.srcnn import SRCNN
from models.srcnn_arf import SRCNN_ARF


register_model("srcnn", SRCNN)
register_model("srcnn_arf", SRCNN_ARF)
register_model("fsrcnn", FSRCNN)
register_model("edsr", EDSR)
register_model("edsr_arf", EDSR_ARF)
register_model("edsr_arfmk2", EDSR_ARFMk2)
register_model("rcan", RCAN)
register_model("ldynsr", LDynSR)


MODEL_DEFAULT_KWARGS: Dict[str, Dict[str, Any]] = {
    "srcnn": {
        "in_channels": 1,
        "out_channels": 1,
        "num_features_1": 64,
        "num_features_2": 32,
    },
    "srcnn_arf": {
        "in_channels": 1,
        "out_channels": 1,
        "num_features_1": 64,
        "num_features_2": 32,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 1.0,
        "t_min": 3.0,
        "t_max": 11.0,
    },
    "fsrcnn": {
        "in_channels": 1,
        "out_channels": 1,
        "d": 56,
        "s": 12,
        "m": 4,
    },
    "edsr": {
        "in_channels": 1,
        "out_channels": 1,
        "n_resblocks": 16,
        "n_feats": 64,
        "res_scale": 0.1,
    },
    "edsr_arf": {
        "in_channels": 1,
        "out_channels": 1,
        "n_resblocks": 16,
        "n_feats": 64,
        "res_scale": 0.1,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 1.0,
        "t_min": 3.0,
        "t_max": 11.0,
    },
    "edsr_arfmk2": {
        "in_channels": 1,
        "out_channels": 1,
        "n_resblocks": 16,
        "n_feats": 64,
        "res_scale": 0.1,
        "router_hidden_dim": 32,
    },
    "rcan": {
        "in_channels": 1,
        "out_channels": 1,
        "n_resgroups": 5,
        "n_resblocks": 10,
        "n_feats": 64,
        "reduction": 16,
        "res_scale": 0.1,
    },
    "ldynsr": {
        "in_channels": 1,
        "out_channels": 1,
        "feat_channels": 48,
        "num_dyna": 6,
        "dam_reduction": 16,
    },
}


def get_model_default_kwargs(model_name: str) -> Dict[str, Any]:
    name = model_name.lower()
    if name not in MODEL_DEFAULT_KWARGS:
        raise ValueError(
            f"Unsupported model: {model_name}. Available: {list_models()}"
        )
    return dict(MODEL_DEFAULT_KWARGS[name])


def merge_model_kwargs(
    model_name: str,
    config_kwargs: Optional[Mapping[str, Any]] = None,
    override_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs = get_model_default_kwargs(model_name)
    if config_kwargs:
        kwargs.update(dict(config_kwargs))
    if override_kwargs:
        kwargs.update(dict(override_kwargs))
    return kwargs


def build_model(model_name: str, scale: int = 2, **kwargs: Any):
    name = model_name.lower()

    def _build(model_cls):
        if name in ("srcnn", "srcnn_arf"):
            return model_cls(**kwargs)
        return model_cls(scale=scale, **kwargs)

    return build_registered(name, _build)


SUPPORTED_MODELS = tuple(list_models())
