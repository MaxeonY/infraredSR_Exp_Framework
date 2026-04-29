from typing import Callable

import numpy as np


def compose(*fns: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _apply(x: np.ndarray) -> np.ndarray:
        for fn in fns:
            x = fn(x)
        return x

    return _apply
