import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    # note to self: seed every library that can introduce randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # note to self: benchmark mode picks algorithms by timing them, which breaks reproducibility
    torch.backends.cudnn.benchmark = False

    if deterministic:
        # note to self: cublas needs this env var set before the first cuda matmul
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def make_generator(seed: int) -> torch.Generator:
    # note to self: pass this to DataLoader so shuffling is tied to my seed
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    # note to self: each dataloader worker needs its own seed derived from torch's
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)