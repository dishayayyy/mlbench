from mlbench.data.adult import load_adult
from mlbench.metrics import compute_metrics
from mlbench.models.logreg import TorchLogReg
from mlbench.models.mlp import TorchMLP
from mlbench.seed import set_seed

# note to self: quick sanity script, not part of the final framework
d = load_adult()
for Model in (TorchLogReg, TorchMLP):
    set_seed(0)
    m = Model(seed=0)
    info = m.fit(d.X_train, d.y_train, d.X_val, d.y_val)
    scores = compute_metrics(d.y_test, m.predict_proba(d.X_test))
    print(m.name, info)
    print({k: round(v, 4) for k, v in scores.items()})