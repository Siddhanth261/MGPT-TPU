from dataclasses import dataclass
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp


@dataclass
class AdamWConfig:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


class AdamWState(NamedTuple):
    count: int
    m: Any
    v: Any