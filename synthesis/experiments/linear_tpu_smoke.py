import jax
import jax.numpy as jnp

from synthesis.synthesis_core.train_loop import train, TrainConfig
from synthesis.synthesis_core.sharding import shard_batch


def init_params_fn(rng):
    # Simple linear model: y = Wx + b
    W = jax.random.normal(rng, (32, 1))
    b = jnp.zeros((1,))
    return {"W": W, "b": b}


def loss_fn(params, batch, rng):
    x = batch["x"]
    y = batch["y"]
    pred = jnp.matmul(x, params["W"]) + params["b"]
    loss = jnp.mean((pred - y) ** 2)
    return loss


def create_dataset():
    # Generate synthetic data
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (8192, 32))  # global batch
    true_w = jax.random.normal(key, (32, 1))
    y = jnp.matmul(X, true_w)

    return X, y


def batch_fn(step):
    X, y = create_dataset()
    return {"x": X, "y": y}


def main():
    cfg = TrainConfig(
        steps=20,
        global_batch=8192,
        lr=1e-3,
        log_every=5,
    )

    rng = jax.random.PRNGKey(42)
    params = train(cfg, init_params_fn, loss_fn, batch_fn, rng)
    print("Training complete.")


if __name__ == "__main__":
    main()
