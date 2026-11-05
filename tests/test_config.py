import pytest

from mlbench.config import load_config


def test_loads_example_config():
    cfg = load_config("configs/adult_logreg.yaml")
    assert cfg.dataset == "adult"
    assert len(cfg.seeds) == 10


def test_rejects_unknown_model(tmp_path):
    # note to self: bad configs should fail at load time, not mid-run
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\ndataset: adult\nmodel: nonsense\nseeds: [0]\n")
    with pytest.raises(ValueError):
        load_config(p)