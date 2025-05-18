import jax

def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))
