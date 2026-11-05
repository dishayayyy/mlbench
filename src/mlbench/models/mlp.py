from torch import nn

from mlbench.models.torch_trainer import TorchBinaryClassifier


class TorchMLP(TorchBinaryClassifier):
    # note to self: same trainer as logreg, only the network changes, so the comparison stays fair
    name = "mlp"

    def __init__(self, hidden_dims=(128, 64), dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

    def build_network(self, n_features):
        layers, prev = [], n_features
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)