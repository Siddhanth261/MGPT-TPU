import jax

def shard_rng(rng, n_devices):
    return jax.random.split(rng, n_devices)
