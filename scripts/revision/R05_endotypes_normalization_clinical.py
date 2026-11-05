#!/usr/bin/env python3
"""
R05_endotypes_normalization_clinical.py
Frozen re-derivation of the initial and corrected partitions, normalisation
sensitivity, cluster-existence testing, stability, split-half replication,
clinical characterisation and covariate adjustment.
(R1 comments 5, 6, 7, 9 and minor 3; R2 comments 3, 4, 5, 7; R3 Fig 3b/4d.)

Normalisation arms for the proteomic layer (all per-sample or per-protein,
none uses group labels):
  none                 vendor NPX as delivered (intensity-normalised by Olink)
  shift_adjusted       PRIMARY (pre-specified): every protein regressed on the
                       per-sample global intensity score (protein-specific slope)
  median_centred       global per-sample median subtracted (submitted method)
  block_median         per-sample median subtracted within each Explore block
  quantile             quantile normalisation to the mean sorted distribution
  rank_inverse_normal  per-sample rank-based inverse normal transform
  remove_PC1           first principal component of the proteome removed (RUV-like)
  shift_adjusted_LOD   primary arm restricted to assays with >=50% samples > LOD

Clustering is unchanged from the manuscript (per-feature z, top-1,000 features
per layer, PCA(10) of concatenated layers, k-means k = 2, 20 restarts).
Higher-risk group = the cluster with the higher METS frequency.
Protein differences BETWEEN DATA-DERIVED CLUSTERS are reported descriptively
(effect sizes; counts given for comparability with the submitted version)
because inference on clusters estimated from the same features is circular
(Gao, Bien & Witten 2022). Inferential protein results use METS, which was
never used to derive any representation.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

import common as CM

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n_stab", type=int, default=500)
ap.add_argument("--n_null", type=int, default=1000)
ap.add_argument("--n_split", type=int, default=200)
args = ap.parse_args()
CM.log_environment(__file__)

ids, RNA, PROT, MET, CLIN = CM.load_all()
ann = CM.olink_assay_annotation().reindex(PROT.columns)
npx = CM.olink_long(ids)
SHIFT = CM.shift_score(PROT)
mets = CLIN["METS"].astype(int).values
s01 = pd.read_csv(CM.TAB / "R01_sample_level_metadata.csv", index_col=0)
s01.index = s01.index.astype(str)
s01 = s01.loc[ids]


# ---------------------------------------------------------------- arms
def arm(name):
    P = PROT.copy()
    if name == "none":
        return P
    if name in ("shift_adjusted", "shift_adjusted_LOD"):
        s = SHIFT - SHIFT.mean()
        beta = (s.values @ (P - P.mean()).values) / (s @ s)
        P = P - np.outer(s, beta)
        if name.endswith("LOD"):
            P = P[[c for c in P.columns if ann.loc[c, "frac_above_LOD"] >= CM.LOD_MIN_FRAC]]
        return P
    if name == "median_centred":
        return P.sub(P.median(1), axis=0)
    if name == "block_median":
        out = P.copy()
        for panel, cols in ann.groupby("Panel").groups.items():
            cols = [c for c in cols if c in P.columns]
            out[cols] = P[cols].sub(P[cols].median(1), axis=0)
        return out
    if name == "quantile":
        X = P.values
        R = np.argsort(np.argsort(X, 1), 1)
        ref = np.sort(X, 1).mean(0)
        return pd.DataFrame(ref[R], index=P.index, columns=P.columns)
    if name == "rank_inverse_normal":
        X = P.values
        out = np.vstack([stats.norm.ppf(stats.rankdata(r) / (len(r) + 1)) for r in X])
        return pd.DataFrame(out, index=P.index, columns=P.columns)
    if name == "remove_PC1":
        Z = P - P.mean()
        p = PCA(1, random_state=0).fit(Z)
        return P - p.inverse_transform(p.transform(Z))
    raise ValueError(name)


ARMS = ["none", "shift_adjusted", "median_centred", "block_median", "quantile",
        "rank_inverse_normal", "remove_PC1", "shift_adjusted_LOD"]
PRIMARY = "shift_adjusted"


def z_cap(X):
    X = np.asarray(X, float)
    mu, sd = CM.zscore_fit(X)
    Z = np.nan_to_num((X - mu) / sd)
    return Z[:, CM.cap_fit(Z)]


Rz, Mz = z_cap(RNA.values), z_cap(MET.values)


def embed(P, layers=("rna", "prot", "met")):
    blocks = {"rna": Rz, "prot": z_cap(P.values), "met": Mz}
    X = np.hstack([blocks[l] for l in layers])
    return PCA(CM.N_PCS, random_state=0).fit_transform(X)


def km(E, k=2, seed=0):
    return KMeans(k, n_init=20, random_state=seed).fit_predict(E)


def orient(part):
    """1 = higher-METS cluster."""
    r = [mets[part == k].mean() for k in (0, 1)]
    return part if r[1] >= r[0] else 1 - part


def diagnostics(P, part):
    med = P.median(1).values
    p_med = stats.mannwhitneyu(med[part == 1], med[part == 0]).pvalue
    pv = stats.mannwhitneyu(P.values[part == 1], P.values[part == 0], axis=0).pvalue
    q = CM.bh(pv)
    d = P.values[part == 1].mean(0) - P.values[part == 0].mean(0)
    sig = q < CM.FDR
    up, dn = int((d[sig] > 0).sum()), int((d[sig] < 0).sum())
    ari_wo = adjusted_rand_score(part, km(embed(P, ("rna", "met"))))
    ari_p = adjusted_rand_score(part, km(PCA(CM.N_PCS, random_state=0).fit_transform(z_cap(P.values))))
    flags = dict(f_global=p_med < 1e-3, f_imbalance=(up + dn) >= 10 and max(up, dn) / (up + dn) > 0.9,
                 f_survival=ari_wo < 0.2, f_dominance=ari_p > 0.7)
    return dict(p_median=p_med, n_sig=int(sig.sum()), n_up=up, n_down=dn, ari_without_prot=ari_wo,
                ari_prot_only=ari_p, n_flags=int(sum(flags.values())), **flags), pd.DataFrame(
        {"protein": P.columns, "diff_high_minus_low": d, "p": pv, "q": q})


# ---------------------------------------------------------------- per-arm loop
arm_rows, parts, diffs, prot_mets = [], {}, {}, {}
covars = CLIN[["AGE", "GENDER"]].copy()
for a in ARMS:
    P = arm(a)
    E = embed(P)
    part = orient(km(E))
    parts[a] = part
    dg, dd = diagnostics(P, part)
    diffs[a] = dd
    sil = {k: silhouette_score(E, km(E, k)) for k in range(2, 7)}
    orr, lo, hi, p_or = CM.odds_ratio_ci(part, mets)
    # protein ~ METS + age + sex (+ shift score for arm 'none' is NOT added;
    # the arm itself defines the adjustment)
    X = pd.get_dummies(covars, columns=["GENDER"], drop_first=True).astype(float)
    X.insert(0, "METS", mets)
    X = sm.add_constant(X)
    pv, beta = [], []
    for c in P.columns:
        f = sm.OLS(P[c].values, X).fit()
        pv.append(f.pvalues["METS"]); beta.append(f.params["METS"])
    pm = pd.DataFrame({"protein": P.columns, "beta_METS": beta, "p": pv, "q": CM.bh(pv)})
    prot_mets[a] = pm
    arm_rows.append(dict(arm=a, n_assays=P.shape[1], size_high=int(part.sum()), size_low=int((1 - part).sum()),
                         METS_high=mets[part == 1].mean(), METS_low=mets[part == 0].mean(),
                         OR_METS=orr, OR_lo=lo, OR_hi=hi, p_METS_fisher=p_or,
                         silhouette_k2=sil[2], best_k=max(sil, key=sil.get),
                         ARI_vs_primary=np.nan, ARI_vs_none=np.nan,
                         n_protein_METS_q05=int((pm.q < CM.FDR).sum()), **dg))
    print(a, "done", flush=True)

arms = pd.DataFrame(arm_rows)
for i, a in enumerate(ARMS):
    arms.loc[i, "ARI_vs_primary"] = adjusted_rand_score(parts[a], parts[PRIMARY])
    arms.loc[i, "ARI_vs_none"] = adjusted_rand_score(parts[a], parts["none"])


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


ref_set = set(prot_mets[PRIMARY].query("q < 0.05").protein)
arms["Jaccard_METS_proteins_vs_primary"] = [jacc(prot_mets[a].query("q < 0.05").protein, ref_set) for a in ARMS]
arms["rho_METS_effects_vs_primary"] = [stats.spearmanr(
    prot_mets[a].set_index("protein").beta_METS.reindex(prot_mets[PRIMARY].protein),
    prot_mets[PRIMARY].beta_METS, nan_policy="omit")[0] for a in ARMS]
arms.to_csv(CM.TAB / "R05_normalization_arms.csv", index=False)
arms.to_csv(CM.FIG / "FR4_normalization_arms.csv", index=False)
for a in ARMS:
    diffs[a].to_csv(CM.TAB / f"R05_cluster_differences_{a}.csv", index=False)
    prot_mets[a].to_csv(CM.TAB / f"R05_protein_METS_{a}.csv", index=False)
pd.DataFrame(parts, index=ids).to_csv(CM.TAB / "R05_partitions_all_arms.csv")
print(arms[["arm", "n_flags", "n_sig", "n_up", "n_down", "OR_METS", "p_METS_fisher", "silhouette_k2",
            "ARI_vs_primary", "n_protein_METS_q05", "Jaccard_METS_proteins_vs_primary"]].round(3).to_string())

# figure source data: volcanoes for the initial (none) and primary partitions
for a, tag in (("none", "FR5a_volcano_initial"), (PRIMARY, "FR5b_volcano_primary"),
               ("median_centred", "FR5c_volcano_median_centred")):
    d = diffs[a].copy()
    d["direction"] = np.where(d.q >= CM.FDR, "ns", np.where(d.diff_high_minus_low > 0, "higher", "lower"))
    d.to_csv(CM.FIG / f"{tag}.csv", index=False)

# ---------------------------------------------------------------- primary arm
P = arm(PRIMARY)
E = embed(P)
part = parts[PRIMARY]

# (a) cluster-existence test: k-means 2-cluster index vs Gaussian null
def ci2(E_):
    k = KMeans(2, n_init=10, random_state=0).fit(E_)
    return k.inertia_ / ((E_ - E_.mean(0)) ** 2).sum()
obs_ci = ci2(E)
cov = np.cov(E.T)
r = CM.rng(5)
null_ci = np.array([ci2(r.multivariate_normal(E.mean(0), cov, len(E))) for _ in range(args.n_null)])
p_sig = (1 + (null_ci <= obs_ci).sum()) / (1 + len(null_ci))
gap = {}
for k in range(1, 7):
    Wk = KMeans(k, n_init=10, random_state=0).fit(E).inertia_ if k > 1 else ((E - E.mean(0)) ** 2).sum()
    Wnull = []
    for b in range(50):
        Rn = r.uniform(E.min(0), E.max(0), E.shape)
        Wnull.append(np.log(KMeans(k, n_init=5, random_state=0).fit(Rn).inertia_ if k > 1 else ((Rn - Rn.mean(0)) ** 2).sum()))
    gap[k] = (np.mean(Wnull) - np.log(Wk), np.std(Wnull) * np.sqrt(1 + 1 / 50))
k_gap = next((k for k in range(1, 6) if gap[k][0] >= gap[k + 1][0] - gap[k + 1][1]), 6)
exist = dict(cluster_index_observed=obs_ci, cluster_index_null_median=float(np.median(null_ci)),
             p_sigclust_gaussian=p_sig, gap_selected_k=k_gap,
             gap_values={k: round(v[0], 4) for k, v in gap.items()},
             silhouette={k: silhouette_score(E, km(E, k)) for k in range(2, 7)})
pd.Series(exist).to_json(CM.TAB / "R05_cluster_existence.json", indent=2, default_handler=str)
print("cluster existence:", exist)

# (b) subsampling stability with full refit (80%, no replacement)
r = CM.rng(7)
n = len(ids)
labs = []
for b in range(args.n_stab):
    idx = np.sort(r.choice(n, int(0.8 * n), replace=False))
    Pb = PROT.iloc[idx]
    sb = SHIFT.iloc[idx] - SHIFT.iloc[idx].mean()
    beta_b = (sb.values @ (Pb - Pb.mean()).values) / (sb @ sb)
    Pb = Pb - np.outer(sb, beta_b)
    Xb = np.hstack([z_cap(RNA.values[idx]), z_cap(Pb.values), z_cap(MET.values[idx])])
    lab = np.full(n, -1)
    lab[idx] = KMeans(2, n_init=10, random_state=0).fit_predict(PCA(CM.N_PCS, random_state=0).fit_transform(Xb))
    labs.append(lab)
ari_ref = [adjusted_rand_score(part[l >= 0], l[l >= 0]) for l in labs]
pair = []
for i in range(0, len(labs) - 1, 2):
    m = (labs[i] >= 0) & (labs[i + 1] >= 0)
    pair.append(adjusted_rand_score(labs[i][m], labs[i + 1][m]))
# null stability: same procedure on Gaussian data with matched covariance
null_pair = []
for b in range(100):
    G = r.multivariate_normal(E.mean(0), cov, n)
    i1, i2 = r.choice(n, int(0.8 * n), replace=False), r.choice(n, int(0.8 * n), replace=False)
    l1, l2 = np.full(n, -1), np.full(n, -1)
    l1[i1] = KMeans(2, n_init=10, random_state=0).fit_predict(G[i1])
    l2[i2] = KMeans(2, n_init=10, random_state=0).fit_predict(G[i2])
    m = (l1 >= 0) & (l2 >= 0)
    null_pair.append(adjusted_rand_score(l1[m], l2[m]))
stab = dict(ari_vs_full_median=float(np.median(ari_ref)), ari_vs_full_iqr=list(np.percentile(ari_ref, [25, 75])),
            pairwise_ari_median=float(np.median(pair)), null_pairwise_ari_median=float(np.median(null_pair)),
            p_stability_vs_null=float((1 + np.sum(np.array(null_pair) >= np.median(pair))) / (1 + len(null_pair))))
pd.Series(stab).to_json(CM.TAB / "R05_stability.json", indent=2)
pd.DataFrame({"ari_vs_full": ari_ref}).to_csv(CM.FIG / "FR6a_stability_subsampling.csv", index=False)
pd.DataFrame({"pairwise": pair}).to_csv(CM.TAB / "R05_stability_pairwise.csv", index=False)
pd.DataFrame({"null_pairwise": null_pair}).to_csv(CM.TAB / "R05_stability_null.csv", index=False)
print("stability:", stab)

# (c) split-half replication (derive on half A, project half B)
r = CM.rng(11)
rep_rows = []
for b in range(args.n_split):
    perm = r.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    def fitset(idx):
        Pi = PROT.iloc[idx]; si = SHIFT.iloc[idx] - SHIFT.iloc[idx].mean()
        bi = (si.values @ (Pi - Pi.mean()).values) / (si @ si)
        return bi
    bA = fitset(A)
    def transform(idx, b_, ref_idx):
        Pi = PROT.iloc[idx] - np.outer(SHIFT.iloc[idx] - SHIFT.iloc[ref_idx].mean(), b_)
        return Pi.values
    blocksA, blocksB = [], []
    for XA, XB in ((RNA.values[A], RNA.values[B]), (transform(A, bA, A), transform(B, bA, A)),
                   (MET.values[A], MET.values[B])):
        mu, sd = CM.zscore_fit(XA)
        ZA, ZB = np.nan_to_num((XA - mu) / sd), np.clip(np.nan_to_num((XB - mu) / sd), -10, 10)
        keep = CM.cap_fit(ZA)
        blocksA.append(ZA[:, keep]); blocksB.append(ZB[:, keep])
    pca = PCA(CM.N_PCS, random_state=0).fit(np.hstack(blocksA))
    EA, EB = pca.transform(np.hstack(blocksA)), pca.transform(np.hstack(blocksB))
    kmA = KMeans(2, n_init=20, random_state=0).fit(EA)
    projB = kmA.predict(EB)
    hiA = int(np.argmax([mets[A][kmA.labels_ == k].mean() for k in (0, 1)]))
    projB = (projB == hiA).astype(int)
    ownB = KMeans(2, n_init=20, random_state=0).fit_predict(PCA(CM.N_PCS, random_state=0).fit_transform(np.hstack(blocksB)))
    orB = CM.odds_ratio_ci(projB, mets[B]) if 0 < projB.sum() < len(B) else (np.nan,) * 4
    rep_rows.append(dict(rep=b, ari_projected_vs_own=adjusted_rand_score(projB, ownB),
                         OR_METS_in_B=orB[0], METS_rate_high_B=mets[B][projB == 1].mean() if projB.sum() else np.nan,
                         METS_rate_low_B=mets[B][projB == 0].mean() if (1 - projB).sum() else np.nan))
rep = pd.DataFrame(rep_rows)
rep.to_csv(CM.TAB / "R05_split_half_replication.csv", index=False)
rep.to_csv(CM.FIG / "FR6b_split_half.csv", index=False)
rep_sum = dict(median_ari=float(rep.ari_projected_vs_own.median()),
               median_OR_METS_held_out=float(rep.OR_METS_in_B.median()),
               frac_OR_gt1=float((rep.OR_METS_in_B > 1).mean()))
pd.Series(rep_sum).to_json(CM.TAB / "R05_split_half_summary.json", indent=2)
print("split half:", rep_sum)

# (d) continuous metabolic-risk axis: projection on the centroid axis
cent = np.array([E[part == k].mean(0) for k in (0, 1)])
axis = (E - cent.mean(0)) @ (cent[1] - cent[0]) / np.linalg.norm(cent[1] - cent[0])
axis = (axis - axis.mean()) / axis.std()
lr = sm.Logit(mets, sm.add_constant(axis)).fit(disp=0)
axis_or = dict(OR_per_SD=float(np.exp(lr.params[1])), lo=float(np.exp(lr.conf_int()[1, 0])),
               hi=float(np.exp(lr.conf_int()[1, 1])), p=float(lr.pvalues[1]))
pd.DataFrame({"COCOMO_ID": ids, "axis": axis, "group": part, "METS": mets, "PC1": E[:, 0], "PC2": E[:, 1]}).to_csv(
    CM.FIG / "FR6c_axis_and_embedding.csv", index=False)

# (e) clinical table for the primary partition
rows = []
cl = CLIN.copy()
for v in CM.CLINICAL_CONTINUOUS:
    x = pd.to_numeric(cl[v], errors="coerce")
    a, b = x[part == 1].dropna(), x[part == 0].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    g, glo, ghi = CM.hedges_g(a, b)
    rows.append(dict(variable=v, type="continuous", n_high=len(a), n_low=len(b),
                     high=f"{a.median():.2f} [{a.quantile(.25):.2f}-{a.quantile(.75):.2f}]",
                     low=f"{b.median():.2f} [{b.quantile(.25):.2f}-{b.quantile(.75):.2f}]",
                     effect="Hedges g", estimate=g, ci_low=glo, ci_high=ghi,
                     cliffs_delta=CM.cliffs_delta(a, b), p=stats.mannwhitneyu(a, b).pvalue,
                     METS_component=v in CM.METS_COMPONENTS))
for v in CM.CLINICAL_BINARY + ["smoker"]:
    x = pd.to_numeric(cl[v], errors="coerce")
    m = x.notna()
    o, lo, hi, p = CM.odds_ratio_ci(part[m], x[m].astype(int))
    rows.append(dict(variable=v, type="binary", n_high=int((part[m] == 1).sum()), n_low=int((part[m] == 0).sum()),
                     high=f"{x[m][part[m]==1].mean()*100:.0f}%", low=f"{x[m][part[m]==0].mean()*100:.0f}%",
                     effect="odds ratio", estimate=o, ci_low=lo, ci_high=hi, p=p,
                     METS_component=v in CM.METS_COMPONENTS or v == "METS"))
for v in CM.CLINICAL_CATEGORICAL:
    x = cl[v]; m = x.notna()
    t = pd.crosstab(part[m], x[m])
    rows.append(dict(variable=v, type="categorical", n_high=int((part[m] == 1).sum()), n_low=int((part[m] == 0).sum()),
                     high="; ".join(f"{k} {t.loc[1, k]}" for k in t.columns),
                     low="; ".join(f"{k} {t.loc[0, k]}" for k in t.columns),
                     effect="Cramer V", estimate=np.sqrt(stats.chi2_contingency(t)[0] / (t.values.sum() * (min(t.shape) - 1))),
                     p=stats.chi2_contingency(t)[1], METS_component=False))
ct = pd.DataFrame(rows)
ct["q_BH_all_clinical"] = CM.bh(ct.p)
ct.to_csv(CM.TAB / "R05_clinical_table_primary.csv", index=False)
ct.to_csv(CM.FIG / "FR6d_clinical_effects.csv", index=False)
print(ct[["variable", "n_high", "n_low", "high", "low", "estimate", "ci_low", "ci_high", "p", "q_BH_all_clinical"]].round(3).to_string())

# (f) adjusted association of the group / axis with METS
df = CLIN[["AGE", "GENDER", "BMI", "X3rd_Drug", "first_drug", "CD4"]].copy()
df["METS"], df["group"], df["axis"] = mets, part, axis
df["platelet_score"], df["intracellular_score"] = s01.platelet_score.values, s01.intracellular_score.values
df["first_drug"] = df.first_drug.replace({"AZT": "Other"})
for c in ["AGE", "CD4", "BMI", "platelet_score", "intracellular_score"]:
    df[c] = (df[c] - df[c].mean()) / df[c].std()   # standardized for numerical stability
adj = []
for form in ["METS ~ group", "METS ~ group + AGE + C(GENDER)",
             "METS ~ group + AGE + C(GENDER) + C(X3rd_Drug) + C(first_drug) + CD4",
             "METS ~ group + AGE + C(GENDER) + platelet_score + intracellular_score",
             "METS ~ axis", "METS ~ axis + AGE + C(GENDER) + C(X3rd_Drug) + C(first_drug) + CD4",
             "METS ~ axis + AGE + C(GENDER) + platelet_score + intracellular_score"]:
    try:
        f = smf.logit(form, df.dropna(subset=[c for c in df.columns if c in form or c in ("METS",)])).fit(disp=0, method="newton", maxiter=200)
        assert f.mle_retvals.get("converged", True), "not converged"
        term = "group" if "group" in form else "axis"
        ci = f.conf_int().loc[term]
        adj.append(dict(model=form, n=int(f.nobs), OR=np.exp(f.params[term]), lo=np.exp(ci[0]),
                        hi=np.exp(ci[1]), p=f.pvalues[term]))
    except Exception as e:
        adj.append(dict(model=form, error=str(e)))
adj = pd.DataFrame(adj)
adj.to_csv(CM.TAB / "R05_adjusted_METS_models.csv", index=False)
print(adj.round(3).to_string())

# (g) does the corrected axis track pre-analytical scores?
pre = dict(rho_axis_platelet=stats.spearmanr(axis, s01.platelet_score)[0],
           rho_axis_intracellular=stats.spearmanr(axis, s01.intracellular_score)[0],
           rho_axis_shift=stats.spearmanr(axis, SHIFT)[0],
           rho_initial_axis_shift=None)
E0 = embed(arm("none"))
p0 = parts["none"]
c0 = np.array([E0[p0 == k].mean(0) for k in (0, 1)])
ax0 = (E0 - c0.mean(0)) @ (c0[1] - c0[0])
pre["rho_initial_axis_shift"] = stats.spearmanr(ax0, SHIFT)[0]
# effect sizes of corrected cluster differences vs each protein's shift loading
L = pd.read_csv(CM.TAB / "R01_protein_shift_loading.csv", index_col=0)
dprim = diffs[PRIMARY].set_index("protein").diff_high_minus_low
pre["rho_corrected_effect_vs_shift_loading"] = stats.spearmanr(dprim, L.rho_with_shift.reindex(dprim.index))[0]
pd.Series(pre).to_json(CM.TAB / "R05_preanalytical_checks.json", indent=2)
print("pre-analytical:", pre, "axis OR:", axis_or)
pd.Series(axis_or).to_json(CM.TAB / "R05_axis_OR.json", indent=2)

# (h) top-40 heat map for the primary partition (z-scored, ordered by group)
top = diffs[PRIMARY].sort_values("p").protein.head(40).tolist()
H = P[top]
H = (H - H.mean()) / H.std()
H.insert(0, "group", part)
H.sort_values("group", ascending=False).to_csv(CM.FIG / "FR6e_heatmap_top40_primary.csv")

# (i) per-sample median NPX by initial and primary partition (Figure 3a source)
pd.DataFrame({"COCOMO_ID": ids, "median_NPX_vendor": PROT.median(1).values,
              "median_NPX_primary": P.median(1).values,
              "initial_group": parts["none"], "primary_group": parts[PRIMARY]}).to_csv(
    CM.FIG / "FR5d_sample_medians_by_partition.csv", index=False)
d0 = arms.set_index("arm")
pd.DataFrame({"comparison": ["proteomics only", "without proteomics"],
              "ARI_initial": [d0.loc["none", "ari_prot_only"], d0.loc["none", "ari_without_prot"]],
              "ARI_primary": [d0.loc[PRIMARY, "ari_prot_only"], d0.loc[PRIMARY, "ari_without_prot"]]}).to_csv(
    CM.FIG / "FR5e_layer_ARI.csv", index=False)
