#!/usr/bin/env python3
"""
R02_nested_benchmark.py
Fully nested, leakage-free benchmark of integration methods for the held-out
endpoint (metabolic syndrome, METS). Reviewer 1 comments 1, 2 and 7; Reviewer 2
comment 1.

Every data-dependent step is fitted on the outer TRAINING fold only and then
applied to the outer TEST fold:
  per-feature standardisation -> top-variance feature cap (1,000 per layer)
  -> optional proteomic shift adjustment (protein-specific slopes)
  -> representation (PCA / per-layer SVD / PoE-VAE / none)
  -> L2 logistic regression whose C is chosen by inner 4-fold CV.
PoE-VAE hyperparameters are chosen by inner 4-fold CV on the training fold
(criterion: inner held-out AUROC of the same classifier head).
The same outer splits (R repeats x 5 folds, stratified) are used for every
method, so between-method differences are paired.

Uncertainty: corrected repeated k-fold t-test (Nadeau & Bengio 2003;
Bouckaert & Frank 2004) on the k*R paired per-fold differences; the
percentile spread of per-repeat pooled AUROCs is reported only as a
description of split-to-split variability, not as a confidence interval.
A whole-pipeline label-permutation test is run for PCA and the VAE.

Usage: python R02_nested_benchmark.py [--quick]
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common as C
from vae import PoEVAE

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--n_perm", type=int, default=200)
ap.add_argument("--repeats", type=str, default="",
                help="comma list or a-b range of repeat indices to run (resumable chunks)")
ap.add_argument("--summarize_only", action="store_true")
args = ap.parse_args()
R = 2 if args.quick else C.N_OUTER_REPEATS
C.log_environment(__file__)

ids, RNA, PROT, MET, CLIN = C.load_all()
y = CLIN["METS"].astype(int).values
y_snf = (CLIN["cluster"].astype(int).values == CLIN["cluster"].astype(int).max()).astype(int)
SHIFT = C.shift_score(PROT).values
LAYERS = {"rna": RNA.values.astype(float), "prot": PROT.values.astype(float),
          "met": MET.values.astype(float)}

clin_cov = pd.get_dummies(CLIN[["AGE", "GENDER", "X3rd_Drug", "CD4"]],
                          columns=["GENDER", "X3rd_Drug"], drop_first=True).astype(float)
clin_cov = clin_cov.fillna(clin_cov.median()).values

VAE_GRID = [dict(latent=l, hidden=h, dropout=d)
            for l, h, d in itertools.product([4, 8], [32, 64], [0.3])]
if args.quick:
    VAE_GRID = VAE_GRID[:2]
VAE_SEEDS = [0, 1, 2]


# ------------------------------------------------------------------ helpers
def prep_layers(tr, te, prot_mode="raw", layers=("rna", "prot", "met")):
    """Fold-internal preprocessing. Returns lists of train/test arrays."""
    Xtr, Xte = [], []
    for name in layers:
        A = LAYERS[name]
        a_tr, a_te = A[tr].copy(), A[te].copy()
        if name == "prot" and prot_mode == "shift_adjusted":
            s_tr, s_te = SHIFT[tr], SHIFT[te]
            sc = s_tr - s_tr.mean()
            beta = (sc @ (a_tr - a_tr.mean(0))) / (sc @ sc)
            a_tr = a_tr - np.outer(s_tr - s_tr.mean(), beta)
            a_te = a_te - np.outer(s_te - s_tr.mean(), beta)
        elif name == "prot" and prot_mode == "median_centred":
            a_tr = a_tr - np.median(a_tr, 1, keepdims=True)
            a_te = a_te - np.median(a_te, 1, keepdims=True)
        mu, sd = C.zscore_fit(a_tr)
        z_tr = np.clip((a_tr - mu) / sd, -10, 10)
        z_te = np.clip((a_te - mu) / sd, -10, 10)
        keep = C.cap_fit(z_tr)
        Xtr.append(np.nan_to_num(z_tr[:, keep])); Xte.append(np.nan_to_num(z_te[:, keep]))
    return Xtr, Xte


def head_fit_predict(Ztr, ytr, Zte, seed):
    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=StratifiedKFold(C.N_INNER_FOLDS, shuffle=True,
                               random_state=seed), scoring="roc_auc", max_iter=5000)
    clf.fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1], clf.predict_proba(Ztr)[:, 1]


def rep_pca(Xtr, Xte, k=8):
    A, B = np.hstack(Xtr), np.hstack(Xte)
    p = PCA(k, random_state=0).fit(A)
    return p.transform(A), p.transform(B)


def rep_svd(Xtr, Xte, k=3):
    outs = [TruncatedSVD(k, random_state=0).fit(a) for a in Xtr]
    return (np.hstack([o.transform(a) for o, a in zip(outs, Xtr)]),
            np.hstack([o.transform(b) for o, b in zip(outs, Xte)]))


def fit_vae(Xtr, cfg, seed, lam=0.0, y_anchor=None):
    return PoEVAE(**cfg, anchor_lambda=lam, seed=seed).fit(Xtr, y_anchor=y_anchor)


def rep_vae(Xtr, Xte, cfg, lam=0.0, y_anchor=None):
    """Ensemble over seeds: concatenate seed-specific latents would change
    dimensionality, so average the predictions instead (returned per seed)."""
    out = []
    for s in VAE_SEEDS:
        m = fit_vae(Xtr, cfg, s, lam, y_anchor)
        out.append((m.transform(Xtr), m.transform(Xte), m.train_rec, m.reconstruction_error(Xte)))
    return out


def select_vae_cfg(tr, prot_mode, seed):
    """Inner CV on the outer-training fold to choose the VAE configuration."""
    inner = StratifiedKFold(C.N_INNER_FOLDS, shuffle=True, random_state=seed)
    scores = {i: [] for i in range(len(VAE_GRID))}
    for itr, ite in inner.split(tr, y[tr]):
        a, b = tr[itr], tr[ite]
        Xa, Xb = prep_layers(a, b, prot_mode)
        for i, cfg in enumerate(VAE_GRID):
            m = fit_vae(Xa, cfg, 0)
            Za, Zb = m.transform(Xa), m.transform(Xb)
            pr = LogisticRegression(C=1.0, max_iter=5000).fit(Za, y[a]).predict_proba(Zb)[:, 1]
            scores[i].append(roc_auc_score(y[b], pr))
    best = max(scores, key=lambda i: np.mean(scores[i]))
    return VAE_GRID[best], {str(VAE_GRID[i]): float(np.mean(v)) for i, v in scores.items()}


# ------------------------------------------------------------------ methods
def run_methods(tr, te, seed, y_use):
    res = {}
    Xtr, Xte = prep_layers(tr, te, "raw")
    Xtr_c, Xte_c = prep_layers(tr, te, "shift_adjusted")

    def score(name, Ztr, Zte):
        p_te, p_tr = head_fit_predict(Ztr, y_use[tr], Zte, seed)
        res[name] = (p_te, roc_auc_score(y_use[tr], p_tr))

    score("PCA (concatenated)", *rep_pca(Xtr, Xte))
    score("PCA (concatenated), shift-adjusted proteomics", *rep_pca(Xtr_c, Xte_c))
    score("Per-layer SVD", *rep_svd(Xtr, Xte))
    for i, nm in enumerate(["transcriptomics", "proteomics", "metabolomics"]):
        score(f"PCA ({nm} only)", *rep_pca([Xtr[i]], [Xte[i]]))
    score("PCA (proteomics only), shift-adjusted", *rep_pca([Xtr_c[1]], [Xte_c[1]]))
    score("PCA (transcriptomics + metabolomics)", *rep_pca([Xtr[0], Xtr[2]], [Xte[0], Xte[2]]))
    # early concatenation, elastic net on all capped features
    A, B = np.hstack(Xtr), np.hstack(Xte)
    en = LogisticRegressionCV(Cs=[0.01, 0.1, 1], l1_ratios=[0.5], penalty="elasticnet", solver="saga",
                              cv=StratifiedKFold(C.N_INNER_FOLDS, shuffle=True, random_state=seed),
                              scoring="roc_auc", max_iter=3000).fit(A, y_use[tr])
    res["Elastic net (early concatenation)"] = (en.predict_proba(B)[:, 1],
                                                roc_auc_score(y_use[tr], en.predict_proba(A)[:, 1]))
    # clinical only (no METS components) and clinical + PCA
    cm, cs = clin_cov[tr].mean(0), clin_cov[tr].std(0) + 1e-8
    Ctr, Cte = (clin_cov[tr] - cm) / cs, (clin_cov[te] - cm) / cs
    score("Clinical covariates only", Ctr, Cte)
    Ptr, Pte = rep_pca(Xtr_c, Xte_c)
    score("Clinical + PCA (shift-adjusted)", np.hstack([Ctr, Ptr]), np.hstack([Cte, Pte]))
    # PoE-VAE, unanchored, nested hyperparameter selection
    cfg, inner = select_vae_cfg(tr, "raw", seed)
    vs = rep_vae(Xtr, Xte, cfg)
    preds = [head_fit_predict(a, y_use[tr], b, seed) for a, b, _, _ in vs]
    res["PoE-VAE (unanchored, tuned)"] = (np.mean([p[0] for p in preds], 0),
                                          np.mean([roc_auc_score(y_use[tr], p[1]) for p in preds]))
    res["_vae_cfg"] = cfg
    res["_vae_rec"] = (np.mean([v[2] for v in vs]), np.mean([v[3] for v in vs]))
    res["_vae_inner"] = inner
    # PoE-VAE anchored to the published SNF cluster (lambda = 1), as in the
    # submitted manuscript, but trained on the training fold only
    va = rep_vae(Xtr, Xte, dict(latent=8, hidden=64, dropout=0.3), lam=1.0, y_anchor=y_snf[tr])
    preds = [head_fit_predict(a, y_use[tr], b, seed) for a, b, _, _ in va]
    res["PoE-VAE (SNF-anchored, lambda = 1)"] = (np.mean([p[0] for p in preds], 0),
                                                 np.mean([roc_auc_score(y_use[tr], p[1]) for p in preds]))
    return res


# ------------------------------------------------------------------ main loop
PART = C.TAB / "R02_parts"; PART.mkdir(exist_ok=True)
if args.repeats:
    rr = args.repeats
    todo = list(range(int(rr.split("-")[0]), int(rr.split("-")[1]) + 1)) if "-" in rr else [int(x) for x in rr.split(",")]
else:
    todo = list(range(R))
t0 = time.time()
if not args.summarize_only:
    for r in todo:
        if (PART / f"rep{r:02d}_folds.csv").exists():
            print(f"repeat {r} already done", flush=True); continue
        rows_r, cfg_r, oof_r = [], [], {}
        skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            res = run_methods(tr, te, C.SEED + 100 * r + f, y)
            for name, val in res.items():
                if name.startswith("_"):
                    continue
                p_te, auc_tr = val
                rows_r.append(dict(repeat=r, fold=f, method=name, auc_test=roc_auc_score(y[te], p_te),
                                   auc_train=auc_tr, n_train=len(tr), n_test=len(te)))
                oof_r.setdefault(name, np.zeros(len(y)))[te] = p_te
            cfg_r.append(dict(repeat=r, fold=f, **res["_vae_cfg"], vae_train_rec=res["_vae_rec"][0],
                              vae_test_rec=res["_vae_rec"][1], inner=str(res["_vae_inner"])))
            print(f"repeat {r} fold {f} done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(cfg_r).to_csv(PART / f"rep{r:02d}_cfg.csv", index=False)
        pd.DataFrame(oof_r).assign(repeat=r).to_csv(PART / f"rep{r:02d}_oof.csv", index=False)
        pd.DataFrame(rows_r).to_csv(PART / f"rep{r:02d}_folds.csv", index=False)   # written last = completion marker
    if args.repeats:
        raise SystemExit(0)
done = sorted(PART.glob("rep*_folds.csv"))
R = len(done)
rows = pd.concat([pd.read_csv(f) for f in done]).to_dict("records")
cfg_rows = pd.concat([pd.read_csv(str(f).replace("_folds", "_cfg")) for f in done]).to_dict("records")
oof = {}
for f in done:
    o = pd.read_csv(str(f).replace("_folds", "_oof"))
    rep = int(o["repeat"].iloc[0])
    for m in o.columns:
        if m != "repeat":
            oof[(m, rep)] = o[m].values
print(f"summarizing {R} repeats", flush=True)
folds = pd.DataFrame(rows)
folds.to_csv(C.TAB / "R02_per_fold_auroc.csv", index=False)
pd.DataFrame(cfg_rows).to_csv(C.TAB / "R02_vae_selected_configs.csv", index=False)

# pooled out-of-fold AUROC per repeat
pooled = pd.DataFrame([dict(method=m, repeat=r, auc_pooled=roc_auc_score(y, p))
                       for (m, r), p in oof.items()])
pooled.to_csv(C.TAB / "R02_pooled_auroc_per_repeat.csv", index=False)

ref = "PCA (concatenated)"
summ = []
for m, g in folds.groupby("method"):
    pr = pooled[pooled.method == m].auc_pooled
    d = g.set_index(["repeat", "fold"]).auc_test - folds[folds.method == ref].set_index(["repeat", "fold"]).auc_test
    md, lo, hi, p = C.corrected_resampled_ttest(d.values, g.n_train.mean(), g.n_test.mean(), C.N_FOLDS, R)
    mean, mlo, mhi, _ = C.corrected_resampled_ttest(g.auc_test.values - 0.5, g.n_train.mean(),
                                                     g.n_test.mean(), C.N_FOLDS, R)
    summ.append(dict(method=m, auc_mean_fold=g.auc_test.mean(), auc_ci_low=mlo + 0.5, auc_ci_high=mhi + 0.5,
                     auc_pooled_mean=pr.mean(), auc_pooled_p2_5=pr.quantile(.025), auc_pooled_p97_5=pr.quantile(.975),
                     auc_train_mean=g.auc_train.mean(),
                     diff_vs_PCA=md, diff_ci_low=lo, diff_ci_high=hi, p_diff_corrected=p,
                     frac_repeats_better_than_PCA=float((pr.values > pooled[pooled.method == ref].auc_pooled.values).mean())))
summ = pd.DataFrame(summ).sort_values("auc_mean_fold", ascending=False)
summ.to_csv(C.TAB / "R02_benchmark_summary.csv", index=False)
summ.to_csv(C.FIG / "FR2a_nested_benchmark.csv", index=False)
print(summ.round(3).to_string())

# ------------------------------------------------------------------ permutation
if args.n_perm > 0:
    rs = np.random.default_rng(C.SEED)
    null = {"PCA (concatenated)": [], "PoE-VAE (fixed 8/64/0.3)": []}
    obs = {}
    def one_cv(yv, method):
        skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)
        pr = np.zeros(len(yv))
        for tr, te in skf.split(np.zeros(len(yv)), yv):
            Xtr, Xte = prep_layers(tr, te, "raw")
            if method.startswith("PCA"):
                Ztr, Zte = rep_pca(Xtr, Xte)
            else:
                mdl = fit_vae(Xtr, dict(latent=8, hidden=64, dropout=0.3), 0)
                Ztr, Zte = mdl.transform(Xtr), mdl.transform(Xte)
            pr[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, yv[tr]).predict_proba(Zte)[:, 1]
        return roc_auc_score(yv, pr)
    for m in null:
        obs[m] = one_cv(y, m)
        n_p = args.n_perm if m.startswith("PCA") else max(50, args.n_perm // 4)
        for i in range(n_p):
            null[m].append(one_cv(rs.permutation(y), m))
    perm = pd.DataFrame([dict(method=m, observed=obs[m], n_perm=len(v), null_mean=np.mean(v),
                              p_perm=(1 + np.sum(np.array(v) >= obs[m])) / (1 + len(v))) for m, v in null.items()])
    perm.to_csv(C.TAB / "R02_permutation_tests.csv", index=False)
    print(perm)
print(f"total {time.time()-t0:.0f}s")
