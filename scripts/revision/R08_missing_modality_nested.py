#!/usr/bin/env python3
"""
R08_missing_modality_nested.py
Nested re-run of the missing-modality experiment (Reviewer 1 minor comment 3).

For each missingness level f in {0, 0.1, ..., 0.5} and each of R random
masks, one complete layer is removed at random from a fraction f of ALL
participants (training and test alike). Then, within each outer fold
(5-fold, stratified on METS), every preprocessing step is fitted on the
training fold only:
  PoE-VAE native masking  : absent layers excluded from fusion and reconstruction
  PCA + mean imputation   : training-fold means
  PCA + kNN imputation    : k = 5 neighbours from the training fold
  complete case           : trained on complete training participants, evaluated
                            on complete test participants only
Pooled out-of-fold AUROC per mask; summary = mean and 2.5/97.5 percentiles
across masks (variability of the masking, not a confidence interval).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common as C
from vae import PoEVAE

ap = argparse.ArgumentParser()
ap.add_argument("--masks", type=int, default=10)
args = ap.parse_args()
C.log_environment(__file__)
ids, RNA, PROT, MET, CLIN = C.load_all()
y = CLIN["METS"].astype(int).values
LAY = [RNA.values.astype(float), PROT.values.astype(float), MET.values.astype(float)]
n = len(y)


def prep(tr, te, Xs):
    out_tr, out_te, keep_all = [], [], []
    for X in Xs:
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd = np.where(sd < 1e-6, 1, sd)
        a = np.clip((X[tr] - mu) / sd, -10, 10); b = np.clip((X[te] - mu) / sd, -10, 10)
        keep = np.sort(np.argsort(np.nanvar(a, 0))[::-1][:C.FEATURE_CAP])
        out_tr.append(a[:, keep]); out_te.append(b[:, keep])
    return out_tr, out_te


def clf(Ztr, ytr, Zte):
    return LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, ytr).predict_proba(Zte)[:, 1]


rows = []
for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for m in range(args.masks if f > 0 else 1):
        r = C.rng(1000 * int(f * 10) + m)
        mask = np.ones((n, 3))
        who = r.choice(n, int(round(f * n)), replace=False)
        mask[who, r.integers(0, 3, len(who))] = 0
        Xs = [X.copy() for X in LAY]
        for l in range(3):
            Xs[l][mask[:, l] == 0] = np.nan
        pred = {k: np.full(n, np.nan) for k in ["PoE-VAE (native masking)", "PCA + mean imputation",
                                                 "PCA + kNN imputation", "Complete case"]}
        skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED + m)
        for tr, te in skf.split(np.zeros(n), y):
            Atr, Ate = prep(tr, te, Xs)
            # VAE: zeros at missing positions are ignored through the mask
            vae = PoEVAE(latent=8, hidden=64, dropout=0.3, seed=0).fit(
                [np.nan_to_num(a) for a in Atr], masks=mask[tr])
            Ztr = vae.transform([np.nan_to_num(a) for a in Atr], mask[tr])
            Zte = vae.transform([np.nan_to_num(a) for a in Ate], mask[te])
            pred["PoE-VAE (native masking)"][te] = clf(Ztr, y[tr], Zte)
            Ctr, Cte = np.hstack(Atr), np.hstack(Ate)
            colmean = np.nanmean(Ctr, 0)
            Mtr = np.where(np.isnan(Ctr), colmean, Ctr); Mte = np.where(np.isnan(Cte), colmean, Cte)
            p = PCA(8, random_state=0).fit(Mtr)
            pred["PCA + mean imputation"][te] = clf(p.transform(Mtr), y[tr], p.transform(Mte))
            knn = KNNImputer(n_neighbors=5).fit(Ctr)
            Ktr, Kte = knn.transform(Ctr), knn.transform(Cte)
            p = PCA(8, random_state=0).fit(Ktr)
            pred["PCA + kNN imputation"][te] = clf(p.transform(Ktr), y[tr], p.transform(Kte))
            ctr = tr[mask[tr].all(1)]; cte = te[mask[te].all(1)]
            Btr, Bte = prep(ctr, cte, LAY)
            p = PCA(8, random_state=0).fit(np.hstack(Btr))
            pred["Complete case"][cte] = clf(p.transform(np.hstack(Btr)), y[ctr], p.transform(np.hstack(Bte)))
        for k, v in pred.items():
            ok = ~np.isnan(v)
            rows.append(dict(missing_frac=f, mask=m, method=k, n_evaluated=int(ok.sum()),
                             auroc=roc_auc_score(y[ok], v[ok])))
        print(f, m, flush=True)

d = pd.DataFrame(rows)
d.to_csv(C.TAB / "R08_missing_modality_per_mask.csv", index=False)
s = d.groupby(["missing_frac", "method"]).auroc.agg(mean="mean", p2_5=lambda x: x.quantile(.025),
                                                     p97_5=lambda x: x.quantile(.975)).reset_index()
s.to_csv(C.TAB / "R08_missing_modality_summary.csv", index=False)
s.to_csv(C.FIG / "FR8_missing_modality.csv", index=False)
print(s.pivot(index="missing_frac", columns="method", values="mean").round(3))
