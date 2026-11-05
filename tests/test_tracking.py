from mlbench.config import load_config
from mlbench.tracking import flatten_params


def test_flatten_params_prefixes_sections():
    # note to self: every shipped config must flatten without losing its training settings
    cfg = load_config("configs/adult_mlp.yaml")
    p = flatten_params(cfg)
    assert p["model"] == "mlp"
    assert p["train.lr"] == 0.01
    assert p["data.split_seed"] == 123
    assert p["model.hidden_dims"] == [128, 64]