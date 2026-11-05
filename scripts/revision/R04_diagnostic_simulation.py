#!/usr/bin/env python3
"""
R04_diagnostic_simulation.py
Semi-synthetic simulation of the four layer-dominance diagnostics
(Reviewer 1 comment 4; Reviewer 2 comment 4).

Generator
  Proteomic and "other-layer" (metabolomic) backgrounds are drawn from Gaussian
  models whose covariance matches the real COCOMO data (top 30 principal
  components + diagonal residual), after removing the observed global
  intensity component from the proteome. Samples of any size n can be drawn.
  A latent binary biological group g (50/50) is drawn for every participant.

Scenarios (effects added to the standardised background)
  null               no group effect, no global shift
  technical(rho)     per-sample offset u_i added to every protein with protein-
                     specific loadings sampled from the observed loadings; u
                     correlates with g at rho in {0, 0.3, 0.6}; other layer
                     carries a modest specific group signal
  bio_global(pi)     group g raises a fraction pi in {0.05,0.2,0.5,0.9} of
                     proteins by 0.5 SD (all upward) AND carries a specific
                     signal in the other layer (biology is multi-layer)
  bio_specific       3% of proteins +/-0.8 SD (bidirectional) + other-layer signal
  mixture            technical(rho = 0.3) + bio_specific

Pipeline per replicate (identical to the manuscript): standardise, cap 1,000
features, PCA(10) of the concatenated layers, k-means (k = 2) -> partition;
then the four diagnostics, a threshold sweep for each, and the agreement of the
partition with the true group (ARI).
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200)
args = ap.parse_args()
C.log_environment(__file__)
ids = C.participant_ids()
P = C.olink_matrix(ids)
M = C.metabolomics_matrix(ids)
s = C.shift_score(P).values
sc = s - s.mean()

# protein loadings on the observed shift (slope per NPX unit), and shift scale
Pc = P.values - P.values.mean(0)
beta = (sc @ Pc) / (sc @ sc)
P_res = Pc - np.outer(sc, beta)
shift_sd = s.std()
psd = P_res.std(0)
beta_std = beta / psd   # effect of a 1-unit shift in SD units of each protein

# restrict to the 1,000 most variable features per layer to keep it fast
pk = np.argsort(P_res.var(0))[::-1][:1000]
mk = np.argsort(M.values.var(0))[::-1][:877]
P_res, beta_std = P_res[:, pk] / psd[pk], beta_std[pk]
Mz = (M.values[:, mk] - M.values[:, mk].mean(0)) / (M.values[:, mk].std(0) + 1e-8)


def make_gen(X, k=30):
    p = PCA(k, random_state=0).fit(X)
    L = p.components_.T * np.sqrt(p.explained_variance_)
    resid = X - p.inverse_transform(p.transform(X))
    return L, resid.std(0)


GP, GM = make_gen(P_res), make_gen(Mz)


def draw(gen, n, r):
    L, d = gen
    return r.standard_normal((n, L.shape[1])) @ L.T + r.standard_normal((n, len(d))) * d


def pipeline(Xp, Xo):
    zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
    zo = (Xo - Xo.mean(0)) / (Xo.std(0) + 1e-8)
    km = lambda Z: KMeans(2, n_init=10, random_state=0).fit_predict(
        PCA(min(10, Z.shape[1]), random_state=0).fit_transform(Z))
    part = km(np.hstack([zp, zo]))
    other = km(zo)
    prot = km(zp)
    a, b = Xp[part == 0], Xp[part == 1]
    if min(len(a), len(b)) < 3:
        return None
    med = np.median(Xp, 1)
    p_med = stats.mannwhitneyu(med[part == 0], med[part == 1]).pvalue
    pv = stats.mannwhitneyu(a, b, axis=0).pvalue
    q = C.bh(pv)
    diff = a.mean(0) - b.mean(0)
    sig = q < 0.05
    nsig = int(sig.sum())
    imb = max((diff[sig] > 0).sum(), (diff[sig] < 0).sum()) / nsig if nsig else np.nan
    return dict(p_median=p_med, n_sig=nsig, imbalance=imb,
                ari_survival=adjusted_rand_score(part, other),
                ari_dominance=adjusted_rand_score(part, prot), part=part)


def scenario_effects(name, n, r):
    g = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    r.shuffle(g)
    gc = g - g.mean()                 # +/-0.5
    Xp, Xo = draw(GP, n, r), draw(GM, n, r)
    tech = np.zeros(n)
    kind, par = name

    def multi_layer_bio(X, frac=0.20, size=0.8):
        """bidirectional group signal on a fraction of features (diff = 2*size SD)"""
        k = int(frac * X.shape[1]); idx = r.choice(X.shape[1], k, replace=False)
        X[:, idx] += np.outer(gc, r.choice([-size, size], k)) * 2
        return X

    def technical(X, mag, rho):
        z = r.standard_normal(n)
        u = rho * (gc / gc.std()) + np.sqrt(1 - rho ** 2) * z
        t = u * shift_sd * mag
        load = r.choice(beta_std, X.shape[1], replace=True)
        return X + np.outer(t, load), t

    if kind == "bio_multilayer":                    # true shared biology only
        Xp, Xo = multi_layer_bio(Xp), multi_layer_bio(Xo)
    elif kind == "technical":                       # (magnitude, rho), no biology
        Xp, tech = technical(Xp, *par)
    elif kind == "technical+bio":                   # shift on top of true biology
        Xp, Xo = multi_layer_bio(Xp), multi_layer_bio(Xo)
        Xp, tech = technical(Xp, *par)
    elif kind == "bio_global":                      # proteome-wide upward biology
        k = int(par * Xp.shape[1]); idx = r.choice(Xp.shape[1], k, replace=False)
        Xp[:, idx] += np.outer(gc, np.full(k, 0.5)) * 2
        Xo = multi_layer_bio(Xo)                    # same biology visible elsewhere
    return Xp, Xo, g, tech


SCEN = ([("null", None), ("bio_multilayer", None)] +
        [("technical", (m, 0.0)) for m in (1, 2, 3)] +
        [("technical+bio", (m, rho)) for m in (1, 2, 3) for rho in (0.0, 0.3)] +
        [("bio_global", p_) for p_ in (0.2, 0.5, 0.9)])
NS = [50, 89, 200]
rows = []
t0 = time.time()
for (scen, n) in itertools.product(SCEN, NS):
    r = C.rng(abs(hash(str(scen) + str(n))) % 10_000)
    for rep in range(args.reps):
        Xp, Xo, g, tech = scenario_effects(scen, n, r)
        d = pipeline(Xp, Xo)
        if d is None:
            continue
        part = d.pop("part")
        d.update(scenario=scen[0], param=str(scen[1]), n=n, rep=rep,
                 ari_truth=adjusted_rand_score(part, g),
                 abs_r_partition_tech=abs(np.corrcoef(part, tech)[0, 1]) if tech.std() > 0 else 0.0)
        rows.append(d)
    print(scen, n, f"{time.time()-t0:.0f}s", flush=True)

sim = pd.DataFrame(rows)
sim.to_csv(C.TAB / "R04_simulation_replicates.csv", index=False)

# default-threshold flags
sim["f1_global"] = sim.p_median < 1e-3
sim["f2_imbalance"] = (sim.n_sig >= 10) & (sim.imbalance > 0.90)
sim["f3_survival"] = sim.ari_survival < 0.2
sim["f4_dominance"] = sim.ari_dominance > 0.7
sim["n_flags"] = sim[["f1_global", "f2_imbalance", "f3_survival", "f4_dominance"]].sum(1)
flag = sim.groupby(["scenario", "param", "n"], dropna=False).agg(
    f1=("f1_global", "mean"), f2=("f2_imbalance", "mean"), f3=("f3_survival", "mean"),
    f4=("f4_dominance", "mean"), ge3=("n_flags", lambda x: (x >= 3).mean()),
    all4=("n_flags", lambda x: (x == 4).mean()), ari_truth=("ari_truth", "median"),
    reps=("rep", "size")).reset_index()
flag.to_csv(C.TAB / "R04_flag_rates.csv", index=False)
flag.to_csv(C.FIG / "FR3a_flag_rates.csv", index=False)

# operating characteristics of the >=3-flag rule against a ground truth:
# "artefact-dominated" = partition tracks the injected technical offset
# (|r| > 0.5) and does not recover the true group (ARI < 0.2).
sim["truth_artifact"] = (sim.abs_r_partition_tech > 0.5) & (sim.ari_truth < 0.2)
sim["truth_biology"] = sim.ari_truth > 0.5
oc = []
for n_, g in sim.groupby("n"):
    for rule in (2, 3, 4):
        pos = g.n_flags >= rule
        oc.append(dict(n=n_, rule=f">={rule} flags",
                       sensitivity=pos[g.truth_artifact].mean(),
                       false_flag_rate_when_biology_recovered=pos[g.truth_biology].mean(),
                       n_artifact=int(g.truth_artifact.sum()), n_biology=int(g.truth_biology.sum())))
pd.DataFrame(oc).to_csv(C.TAB / "R04_operating_characteristics.csv", index=False)
print(pd.DataFrame(oc).round(3).to_string())

# threshold sweeps (n = 89)
s89 = sim[sim.n == 89]
sweep = []
for thr in [1e-2, 1e-3, 1e-4, 1e-6]:
    sweep += [dict(diagnostic="global_shift_p", threshold=thr, scenario=f"{a}|{b}", rate=(g.p_median < thr).mean())
              for (a, b), g in s89.groupby(["scenario", "param"], dropna=False)]
for thr in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    sweep += [dict(diagnostic="imbalance", threshold=thr, scenario=f"{a}|{b}",
                   rate=((g.n_sig >= 10) & (g.imbalance > thr)).mean())
              for (a, b), g in s89.groupby(["scenario", "param"], dropna=False)]
for thr in [0.05, 0.1, 0.2, 0.3, 0.5]:
    sweep += [dict(diagnostic="survival_ARI_below", threshold=thr, scenario=f"{a}|{b}",
                   rate=(g.ari_survival < thr).mean())
              for (a, b), g in s89.groupby(["scenario", "param"], dropna=False)]
for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
    sweep += [dict(diagnostic="dominance_ARI_above", threshold=thr, scenario=f"{a}|{b}",
                   rate=(g.ari_dominance > thr).mean())
              for (a, b), g in s89.groupby(["scenario", "param"], dropna=False)]
sw = pd.DataFrame(sweep)
sw.to_csv(C.TAB / "R04_threshold_sweep.csv", index=False)
sw.to_csv(C.FIG / "FR3b_threshold_sweep.csv", index=False)

# pairwise concordance of diagnostics (phi coefficient) across all replicates
F = sim[["f1_global", "f2_imbalance", "f3_survival", "f4_dominance"]].astype(float)
F.corr().to_csv(C.TAB / "R04_flag_concordance_phi.csv")
print(flag.round(2).to_string())
print(F.corr().round(2))
print(f"total {time.time()-t0:.0f}s")
