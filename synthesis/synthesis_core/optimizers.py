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


def adamw(config: AdamWConfig):
    """
    Returns (init_fn, update_fn) for AdamW optimizer.
    """

    def init_fn(params):
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        return AdamWState(count=0, m=m, v=v)

    def update_fn(grads, state, params):
        count = state.count + 1

        def update_param(g, m, v, p):
            # Momentum updates
            m = config.beta1 * m + (1 - config.beta1) * g
            v = config.beta2 * v + (1 - config.beta2) * (g * g)

            # Bias correction
            m_hat = m / (1 - config.beta1 ** count)
            v_hat = v / (1 - config.beta2 ** count)

            # Weight decay (decoupled)
            update = m_hat / (jnp.sqrt(v_hat) + config.eps)
            update = update + config.weight_decay * p

            # Parameter update
            new_p = p - config.lr * update
            return new_p, m, v

        new_params, new_m, new_v = jax.tree.map(
            update_param,
            grads,
            state.m,
            state.v,
            params
        )

        new_state = AdamWState(count=count, m=new_m, v=new_v)
        return new_params, new_state

    return init_fn, update_fn
