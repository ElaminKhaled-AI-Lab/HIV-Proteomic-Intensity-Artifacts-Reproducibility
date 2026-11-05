"""
vae.py: product-of-experts multimodal VAE used in the revision.

Architecture identical to the submitted version (layer-specific one-hidden-layer
encoders producing Gaussian posteriors, precision-weighted product-of-experts
fusion with a standard-normal prior expert, layer-specific decoders, KL
annealing with a free-bits floor). Additions for the revision:
  * fit()/transform() API so it can be trained strictly inside a training fold;
  * internal early stopping on a held-out 15% of the TRAINING fold only;
  * native masking of absent layers (missing-modality experiment);
  * optional supervised anchor head (cross-entropy weighted by lambda).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads()))


class _Enc(nn.Module):
    def __init__(self, d_in, hidden, latent, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.lv(h)


class _Dec(nn.Module):
    def __init__(self, latent, hidden, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d_out))

    def forward(self, z):
        return self.net(z)


def _poe(mus, lvs, masks):
    # mus, lvs: [L, n, d]; masks: [L, n] (1 = layer observed)
    T = torch.exp(-lvs) * masks.unsqueeze(-1)
    Ts = 1.0 + T.sum(0)                       # prior expert precision = 1
    mu = (mus * T).sum(0) / Ts
    return mu, torch.log(1.0 / Ts)


class PoEVAE:
    def __init__(self, latent=8, hidden=64, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 free_bits=0.5, anneal=80, max_epochs=400, patience=40,
                 anchor_lambda=0.0, seed=0):
        self.kw = dict(latent=latent, hidden=hidden, dropout=dropout)
        self.lr, self.wd = lr, weight_decay
        self.free_bits, self.anneal = free_bits, anneal
        self.max_epochs, self.patience = max_epochs, patience
        self.anchor_lambda = anchor_lambda
        self.seed = seed

    # X: list of np arrays (n x d_l), masks: n x L (1 observed)
    def _tensors(self, X, masks):
        Xt = [torch.tensor(np.nan_to_num(x), dtype=torch.float32) for x in X]
        Mt = torch.tensor(masks.T, dtype=torch.float32)  # L x n
        return Xt, Mt

    def _forward(self, Xt, Mt, idx, sample):
        mus, lvs = [], []
        for l, x in enumerate(Xt):
            m, v = self.enc[l](x[idx])
            mus.append(m); lvs.append(v)
        mu, lv = _poe(torch.stack(mus), torch.stack(lvs), Mt[:, idx])
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv) if sample else mu
        return mu, lv, z

    def _loss(self, Xt, Mt, idx, beta, sample, y=None):
        mu, lv, z = self._forward(Xt, Mt, idx, sample)
        rec = 0.0
        for l, x in enumerate(Xt):
            w = Mt[l, idx].unsqueeze(-1)
            err = ((self.dec[l](z) - x[idx]) ** 2) * w
            rec = rec + err.sum() / (w.sum() * x.shape[1] + 1e-8)
        kl = torch.clamp((-0.5 * (1 + lv - mu.pow(2) - lv.exp())).mean(0), min=self.free_bits).sum()
        loss = rec + beta * kl
        if y is not None and self.anchor_lambda > 0:
            loss = loss + self.anchor_lambda * nn.functional.cross_entropy(self.head(mu), y[idx])
        return loss, rec

    def fit(self, X, masks=None, y_anchor=None):
        torch.manual_seed(self.seed)
        rs = np.random.RandomState(self.seed)
        n = X[0].shape[0]
        masks = np.ones((n, len(X))) if masks is None else masks
        Xt, Mt = self._tensors(X, masks)
        dims = [x.shape[1] for x in X]
        self.enc = nn.ModuleList([_Enc(d, self.kw["hidden"], self.kw["latent"], self.kw["dropout"]) for d in dims])
        self.dec = nn.ModuleList([_Dec(self.kw["latent"], self.kw["hidden"], d, self.kw["dropout"]) for d in dims])
        self.head = nn.Linear(self.kw["latent"], 2)
        params = list(self.enc.parameters()) + list(self.dec.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        y = None if y_anchor is None else torch.tensor(np.asarray(y_anchor, int))
        perm = rs.permutation(n)
        nv = max(4, int(round(0.15 * n)))
        va, tr = perm[:nv], perm[nv:]
        best, best_state, wait = np.inf, None, 0
        self.history = []
        for ep in range(1, self.max_epochs + 1):
            self._train(True)
            opt.zero_grad()
            loss, rec = self._loss(Xt, Mt, tr, min(1.0, ep / self.anneal), True, y)
            loss.backward(); opt.step()
            self._train(False)
            with torch.no_grad():
                _, vrec = self._loss(Xt, Mt, va, 1.0, False)
            self.history.append((ep, float(rec), float(vrec)))
            if ep > self.anneal // 2 and float(vrec) < best - 1e-4:
                best, wait = float(vrec), 0
                best_state = {k: v.clone() for k, v in self._state().items()}
            elif ep > self.anneal // 2:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self._load(best_state)
        self.best_val_rec = best
        self.train_rec = float(rec)
        return self

    def transform(self, X, masks=None):
        n = X[0].shape[0]
        masks = np.ones((n, len(X))) if masks is None else masks
        Xt, Mt = self._tensors(X, masks)
        self._train(False)
        with torch.no_grad():
            mu, _, _ = self._forward(Xt, Mt, np.arange(n), False)
        return mu.numpy()

    def reconstruction_error(self, X, masks=None):
        n = X[0].shape[0]
        masks = np.ones((n, len(X))) if masks is None else masks
        Xt, Mt = self._tensors(X, masks)
        self._train(False)
        with torch.no_grad():
            _, rec = self._loss(Xt, Mt, np.arange(n), 1.0, False)
        return float(rec)

    # helpers
    def _modules(self):
        return [self.enc, self.dec, self.head]

    def _train(self, flag):
        for m in self._modules():
            m.train(flag)

    def _state(self):
        return {f"{i}.{k}": v for i, m in enumerate(self._modules()) for k, v in m.state_dict().items()}

    def _load(self, st):
        for i, m in enumerate(self._modules()):
            m.load_state_dict({k.split(".", 1)[1]: v for k, v in st.items() if k.startswith(f"{i}.")})
