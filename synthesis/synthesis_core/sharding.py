import jax.numpy as jnp

def shard_batch(batch, n_devices):
    global_bs = batch.shape[0]
    assert global_bs % n_devices == 0
    local_bs = global_bs // n_devices
    return batch.reshape(n_devices, local_bs, *batch.shape[1:])
def check_sharded(batch, n_devices):
    for k, v in batch.items():
        assert v.shape[0] == n_devices, f"{k} not sharded correctly."
