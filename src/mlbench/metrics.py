import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def compute_metrics(y_true, p, threshold: float = 0.5) -> dict:
    # note to self: one place for every metric, so each model is scored the exact same way
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, p)),
        "f1": float(f1_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
    }