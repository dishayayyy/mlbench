import numpy as np
from xgboost import XGBClassifier

from mlbench.models.base import BaseModel


class XGBModel(BaseModel):
    # note to self: subsample/colsample below 1.0 is what makes the seed matter, keep them
    name = "xgboost"

    def __init__(self, seed=0, device="cpu", n_estimators=500, max_depth=6,
                 learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                 early_stopping_rounds=20, n_jobs=None):
        self.seed = seed
        self.device = device
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            device=device,
            random_state=seed,
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
        )

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        # note to self: same keys as the torch models, so the runner can log all models the same way
        return {
            "epochs_run": int(self.model.get_booster().num_boosted_rounds()),
            "best_epoch": int(self.model.best_iteration) + 1,
            "best_val_loss": float(self.model.best_score),
        }

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]