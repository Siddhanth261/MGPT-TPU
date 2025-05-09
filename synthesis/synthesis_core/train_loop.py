from dataclasses import dataclass
import jax
@dataclass
class TrainConfig:
    steps: int
    global_batch: int
    lr: float
    log_every: int = 50

def train(cfg, init_fn, loss_fn, batch_fn, rng):
    params = init_fn(rng)
    grad_fn = jax.value_and_grad(loss_fn)

    return params