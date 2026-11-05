import copy

import numpy as np
import torch
from torch import nn

from mlbench.models.base import BaseModel


class TorchBinaryClassifier(BaseModel):
    # note to self: logreg and mlp only override build_network, the loop stays identical

    def __init__(self, seed=0, device="cpu", epochs=50, batch_size=256,
                 lr=0.01, weight_decay=0.0, patience=5):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but cuda is not available")
        self.seed = seed
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.net = None

    def build_network(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _tensor(self, X):
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _logits(self, X_t, chunk=8192):
        # note to self: chunked so a big test set never blows up the 4 GB gpu
        self.net.eval()
        out = [self.net(X_t[i:i + chunk]).squeeze(-1) for i in range(0, len(X_t), chunk)]
        return torch.cat(out)

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        # note to self: seeding here too, so the model is reproducible even without set_seed
        torch.manual_seed(self.seed)

        X_tr, y_tr = self._tensor(X_train), self._tensor(y_train)
        X_va, y_va = self._tensor(X_val), self._tensor(y_val)
        n = len(X_tr)

        self.net = self.build_network(X_tr.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        # note to self: own seeded generator for shuffling, so batch order depends on my seed only
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
        epochs_run = 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X_tr[idx]).squeeze(-1), y_tr[idx])
                loss.backward()
                opt.step()
            epochs_run = epoch

            val_loss = loss_fn(self._logits(X_va), y_va).item()
            if val_loss < best_loss:
                best_loss, best_epoch, bad = val_loss, epoch, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        # note to self: go back to the best epoch, not the last one
        self.net.load_state_dict(best_state)
        return {"epochs_run": epochs_run, "best_epoch": best_epoch, "best_val_loss": best_loss}

    def predict_proba(self, X) -> np.ndarray:
        logits = self._logits(self._tensor(X))
        return torch.sigmoid(logits).cpu().numpy()