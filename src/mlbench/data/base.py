from dataclasses import dataclass

import numpy as np


@dataclass
class TabularSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]