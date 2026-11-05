from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    # note to self: every model (torch or xgboost) must look like this to the runner
    name = "base"

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        """Train the model. Returns a dict of training info (epochs run, best val loss, ...)."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return P(class 1) as a 1-D array, one value per row."""