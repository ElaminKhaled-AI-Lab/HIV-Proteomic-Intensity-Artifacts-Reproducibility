#!/usr/bin/env python3
"""
R03_learning_curve.py
Sample-size dependence of the VAE-versus-PCA comparison (Reviewer 1 comment 2;
Reviewer 2 comment 1). For n in {40, 55, 70, 89}, draw stratified subsamples
(20 per size; the full cohort once per repeat at n = 89), and run 5-fold
stratified CV within the subsample with all preprocessing fold-internal. The
VAE uses the configuration most often selected by the nested benchmark
(read from R02 output; default latent 8, hidden 64, dropout 0.3).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common as C
from vae import PoEVAE

C.log_environment(__file__)
ids, RNA, PROT, MET, CLIN = C.load_all()
y = CLIN["METS"].astype(int).values
LAY = [RNA.values.astype(float), PROT.values.astype(float), MET.values.astype(float)]
try:
    cfgs = pd.read_csv(C.TAB / "R02_vae_selected_configs.csv")
    best = cfgs.groupby(["latent", "hidden", "dropout"]).size().idxmax()
    CFG = dict(latent=int(best[0]), hidden=int(best[1]), dropout=float(best[2]))
except FileNotFoundError:
    CFG = dict(latent=8, hidden=64, dropout=0.3)


def prep(tr, te, idx):
    A, B = [], []
    for X in LAY:
        X = X[idx]
        mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd < 1e-6, 1, sd)
        a = np.clip((X[tr] - mu) / sd, -10, 10); b = np.clip((X[te] - mu) / sd, -10, 10)
        k = np.sort(np.argsort(a.var(0))[::-1][:C.FEATURE_CAP])
        A.append(a[:, k]); B.append(b[:, k])
    return A, B


rows = []
for n_sub in [40, 55, 70, 89]:
    for rep in range(20 if n_sub < 89 else 5):
        r = C.rng(n_sub * 100 + rep)
        if n_sub < 89:
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            k1 = int(round(n_sub * len(pos) / len(y)))
            idx = np.sort(np.r_[r.choice(pos, k1, replace=False), r.choice(neg, n_sub - k1, replace=False)])
        else:
            idx = np.arange(len(y))
        yy = y[idx]
        pr = {"PCA": np.zeros(len(idx)), "PoE-VAE": np.zeros(len(idx))}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(idx)), yy):
            A, B = prep(tr, te, idx)
            p = PCA(8, random_state=0).fit(np.hstack(A))
            pr["PCA"][te] = LogisticRegression(max_iter=5000).fit(p.transform(np.hstack(A)), yy[tr]).predict_proba(
                p.transform(np.hstack(B)))[:, 1]
            v = PoEVAE(**CFG, seed=rep).fit(A)
            pr["PoE-VAE"][te] = LogisticRegression(max_iter=5000).fit(v.transform(A), yy[tr]).predict_proba(
                v.transform(B))[:, 1]
        for k, v in pr.items():
            rows.append(dict(n=n_sub, rep=rep, method=k, auroc=roc_auc_score(yy, v)))
    print(n_sub, flush=True)
d = pd.DataFrame(rows)
d.to_csv(C.TAB / "R03_learning_curve_raw.csv", index=False)
s = d.groupby(["n", "method"]).auroc.agg(["mean", "std", lambda x: x.quantile(.025), lambda x: x.quantile(.975)])
s.columns = ["mean", "sd", "p2_5", "p97_5"]
s.reset_index().to_csv(C.FIG / "FR2c_learning_curve.csv", index=False)
print(s.round(3)); print("VAE config:", CFG)
