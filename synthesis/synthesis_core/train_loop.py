from dataclasses import dataclass

@dataclass
class TrainConfig:
    steps: int
    global_batch: int
    lr: float
    log_every: int = 50

def train(cfg, init_fn, loss_fn, batch_fn, rng):
    pass
