import random

import numpy as np
import torch

from mlbench.seed import set_seed


def _draw():
    # note to self: one draw from each library
    return random.random(), np.random.rand(3), torch.rand(3)


def test_same_seed_gives_same_numbers():
    set_seed(42)
    a = _draw()
    set_seed(42)
    b = _draw()
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_different_seed_gives_different_numbers():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a[0] != b[0]
    assert not torch.equal(a[2], b[2])


def _tiny_training_run(seed):
    # note to self: init, data and updates all depend on the seed, so weights must match across runs
    set_seed(seed)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(32, 4)
    y = torch.randn(32, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return model.weight.detach().clone()


def test_tiny_training_is_reproducible_on_cpu():
    assert torch.equal(_tiny_training_run(0), _tiny_training_run(0))