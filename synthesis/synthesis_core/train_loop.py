import shutil

if os.path.exists("logs/train.jsonl"):
    shutil.move("logs/train.jsonl", f"logs/train_old_{int(time.time())}.jsonl")

import os
import time
start_time = time.time()

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
from synthesis.synthesis_core.schedules import linear_warmup

from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp

from .devices import get_tpu_mesh
from .sharding import shard_batch
from .optimizers import adamw, AdamWConfig
from .logging import JsonLogger
from .checkpoints import save_ckpt


@dataclass
class TrainConfig:
    steps: int
    global_batch: int
    lr: float
    log_every: int = 50
    ckpt_path: str = "checkpoints/ckpt.pkl"


def train(cfg: TrainConfig,
          init_fn: Callable,
          loss_fn: Callable,
          batch_fn: Callable,
          rng):

    mesh = get_tpu_mesh()
    n_devices = mesh.n_devices

    assert cfg.global_batch % n_devices == 0, \
        "Global batch must be divisible by number of devices."

    # Init model
    rng, init_rng = jax.random.split(rng)
    params = init_fn(init_rng)

    # Setup optimizer
    opt_init, opt_update = adamw(AdamWConfig(lr=cfg.lr))
    opt_state = opt_init(params)

    # Prepare logger
    logger = JsonLogger("logs/train.jsonl")

    # Define train step
    def train_step(params, opt_state, batch, rng):
        def loss_wrap(p, b, key):
            return loss_fn(p, b, key)

        grad_fn = jax.value_and_grad(loss_wrap)
        loss, grads = grad_fn(params, batch, rng)

        # Aggregate across devices
        loss = jax.lax.pmean(loss, axis_name="dev")
        grads = jax.lax.pmean(grads, axis_name="dev")
        print(f"step {step} loss: {loss_scalar}")

        new_params, new_state = opt_update(grads, opt_state, params)
        return new_params, new_state, loss

    max_norm = 1.0
    g_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in jax.tree.leaves(grads)]))
    scale = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
    grads = jax.tree.map(lambda g: g * scale, grads)

    # pmap train_step
    p_train_step = jax.pmap(train_step, axis_name="dev")

    # Replicate params & optimizer
    params = jax.device_put_replicated(params, mesh.devices)
    opt_state = jax.device_put_replicated(opt_state, mesh.devices)

    # Training loop
    for step in range(cfg.steps):
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num=n_devices)

        batch_global = batch_fn(step)
        batch = {k: shard_batch(v, n_devices) for k, v in batch_global.items()}

        lr = linear_warmup(step, warmup_steps=50, base_lr=cfg.lr)
        cfg.lr = lr
        params, opt_state, loss = p_train_step(params, opt_state, batch, step_rngs)
        loss_scalar = float(loss[0])
        
        elapsed = time.time() - start_time
        steps_left = cfg.steps - step
        eta = (elapsed / (step + 1e-9)) * steps_left
        print(f"ETA: {eta:.2f}s remaining")

        if step % cfg.log_every == 0:
            logger.log({"step": step, "loss": loss_scalar})

    # Save checkpoint
    host_params = jax.tree.map(lambda x: x[0], params)
    host_opt = jax.tree.map(lambda x: x[0], opt_state)
    save_ckpt(cfg.ckpt_path, {"params": host_params, "opt_state": host_opt})

    return host_params

