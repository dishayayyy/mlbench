from torch import nn

from mlbench.models.torch_trainer import TorchBinaryClassifier


class TorchLogReg(TorchBinaryClassifier):
    # note to self: logistic regression is just one linear layer with a sigmoid on top
    name = "logreg"

    def build_network(self, n_features):
        return nn.Linear(n_features, 1)