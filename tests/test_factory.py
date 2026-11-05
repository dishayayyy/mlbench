import pytest

from mlbench.config import load_config
from mlbench.models.factory import build_model


@pytest.mark.parametrize("path,expected", [
    ("configs/adult_logreg.yaml", "logreg"),
    ("configs/adult_mlp.yaml", "mlp"),
    ("configs/adult_xgb.yaml", "xgboost"),
])
def test_factory_builds_each_model(path, expected):
    # note to self: every shipped config must build the model it claims to
    cfg = load_config(path)
    assert build_model(cfg, seed=0).name == expected