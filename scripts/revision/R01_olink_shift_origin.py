#!/usr/bin/env python3
"""
R01_olink_shift_origin.py
Where does the global per-sample proteomic intensity component come from?
(Reviewer 1 comments 3 and 9; Reviewer 2 comments 2 and 5.)

Uses the vendor NPX export (PlateID, QC_Warning, LOD) and the sample layout map
(sample plate and well) that accompany the public COCOMO repository.

Outputs (results/revision/):
  tables/R01_shift_associations.csv   shift score vs plate, well, QC, LOD, clinical
  tables/R01_plate_balance.csv        clinical balance across sample plates
  tables/R01_location_loading.csv     shift loading by subcellular location
  tables/R01_plate_effect_proteins.csv per-assay plate test (negative control)
  figure_data/FR1a..FR1f_*.csv        source data for the new Figure 4
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

import common as C

C.log_environment(__file__)
ids = C.participant_ids()
P = C.olink_matrix(ids)
npx = C.olink_long(ids)
smap = C.olink_sample_map().loc[ids]
clin = C.clinical(ids)
ann = C.olink_assay_annotation().reindex(P.columns)

shift = C.shift_score(P)
meta = smap[["SampleID", "sample_plate", "well", "well_row", "well_col"]].copy()
meta["shift"] = shift
meta["qc_warn_frac"] = npx.groupby("COCOMO_ID")["QC_Warning"].apply(
    lambda s: (s == "WARN").mean()).reindex(ids)
meta["frac_above_LOD"] = (npx["NPX"] > npx["LOD"]).groupby(npx["COCOMO_ID"]).mean().reindex(ids)

# ---- (1) PC1 of the proteome vs shift --------------------------------------
Z = (P - P.mean()) / P.std()
pca = PCA(10, random_state=0).fit(Z)
pc = pca.transform(Z)
if np.corrcoef(pc[:, 0], shift)[0, 1] < 0:
    pc[:, 0] *= -1
meta["proteome_PC1"] = pc[:, 0]
rho_pc1 = stats.spearmanr(pc[:, 0], shift)[0]
rho_protein = P.apply(lambda c: stats.spearmanr(c, shift)[0])

# ---- (2) block (panel) concordance -----------------------------------------
blk = npx.groupby(["COCOMO_ID", "Panel"])["NPX"].median().unstack().loc[ids]
blk_corr = blk.corr(method="pearson")
off = blk_corr.values[np.triu_indices(len(blk_corr), 1)]

# ---- (3) associations table --------------------------------------------------
rows = []
def add(var, kind, stat, p, n, note=""):
    rows.append(dict(variable=var, test=kind, statistic=stat, p=p, n=n, note=note))

g1, g2 = meta.loc[meta.sample_plate == 1, "shift"], meta.loc[meta.sample_plate == 2, "shift"]
add("sample_plate", "Mann-Whitney", stats.mannwhitneyu(g1, g2).statistic,
    stats.mannwhitneyu(g1, g2).pvalue, len(meta),
    f"median plate1 {g1.median():.3f}, plate2 {g2.median():.3f}")
kw = stats.kruskal(*[g["shift"] for _, g in meta.groupby("well_row")])
add("well_row", "Kruskal-Wallis", kw.statistic, kw.pvalue, len(meta))
r, p = stats.spearmanr(meta.well_col, meta["shift"]); add("well_column", "Spearman", r, p, len(meta))
r, p = stats.spearmanr(meta.qc_warn_frac, meta["shift"]); add("QC_warning_fraction", "Spearman", r, p, len(meta))
r, p = stats.spearmanr(meta.frac_above_LOD, meta["shift"]); add("fraction_assays_above_LOD", "Spearman", r, p, len(meta))
for v in C.CLINICAL_CONTINUOUS:
    x = pd.to_numeric(clin[v], errors="coerce"); m = x.notna()
    if m.sum() > 10:
        r, p = stats.spearmanr(x[m], shift[m]); add(v, "Spearman", r, p, int(m.sum()))
for v in C.CLINICAL_BINARY + C.CLINICAL_CATEGORICAL + ["smoker"]:
    x = clin[v]; m = x.notna()
    groups = [shift[m][x[m] == k] for k in x[m].unique() if (x[m] == k).sum() >= 3]
    if len(groups) > 1:
        t = stats.kruskal(*groups)
        add(v, "Kruskal-Wallis", t.statistic, t.pvalue, int(m.sum()),
            "; ".join(f"{k}: n={(x[m]==k).sum()}" for k in x[m].unique()))
assoc = pd.DataFrame(rows)
assoc["q_BH"] = C.bh(assoc["p"])
assoc.to_csv(C.TAB / "R01_shift_associations.csv", index=False)

# ---- (4) plate balance of clinical variables -------------------------------
bal = []
for v in ["METS", "AGE", "BMI", "CD4", "CD8", "GENDER", "X3rd_Drug"]:
    x = clin[v]
    if x.dtype == object or v in ("METS", "GENDER", "X3rd_Drug"):
        t = pd.crosstab(meta.sample_plate, x)
        bal.append(dict(variable=v, test="chi-square", p=stats.chi2_contingency(t)[1],
                        detail=t.to_dict()))
    else:
        x = pd.to_numeric(x, errors="coerce")
        bal.append(dict(variable=v, test="Mann-Whitney",
                        p=stats.mannwhitneyu(x[meta.sample_plate == 1].dropna(),
                                             x[meta.sample_plate == 2].dropna()).pvalue,
                        detail=f"medians {x[meta.sample_plate==1].median():.1f} vs {x[meta.sample_plate==2].median():.1f}"))
pd.DataFrame(bal).to_csv(C.TAB / "R01_plate_balance.csv", index=False)

# ---- (5) loading by subcellular location -----------------------------------
L = pd.DataFrame({"rho_with_shift": rho_protein, "location": ann["location"],
                  "panel": ann["Panel"], "frac_above_LOD": ann["frac_above_LOD"]})
loc_tab = L.groupby("location")["rho_with_shift"].describe()
kw_loc = stats.kruskal(*[g["rho_with_shift"].dropna() for _, g in L.groupby("location")])
top = L.sort_values("rho_with_shift", ascending=False).head(200)
fish = []
for loc in ["intracellular", "membrane", "secreted"]:
    a = (top.location == loc).sum(); b = 200 - a
    c = (L.location == loc).sum() - a; d = len(L) - 200 - c
    orr, p = stats.fisher_exact([[a, b], [c, d]])
    fish.append(dict(location=loc, top200_frac=a / 200, background_frac=(L.location == loc).mean(),
                     odds_ratio=orr, p=p))
loc_tab.to_csv(C.TAB / "R01_location_loading.csv")
pd.DataFrame(fish).to_csv(C.TAB / "R01_location_top200_enrichment.csv", index=False)
L.to_csv(C.TAB / "R01_protein_shift_loading.csv")

# ---- (6) replicate assays measured in every block --------------------------
rep = npx[npx.Assay.isin(C.EXPLORE_REPLICATE_ASSAYS)]
rep_w = rep.pivot_table(index=["COCOMO_ID", "Assay"], columns="Panel", values="NPX")
rep_rows = []
for a, sub in rep_w.groupby(level="Assay"):
    s = sub.droplevel("Assay").loc[ids].dropna(axis=1, how="all")
    cc = s.corr().values[np.triu_indices(s.shape[1], 1)]
    rep_rows.append(dict(assay=a, n_blocks=s.shape[1], median_between_block_r=np.median(cc),
                         rho_mean_with_shift=stats.spearmanr(s.mean(1), shift)[0]))
pd.DataFrame(rep_rows).to_csv(C.TAB / "R01_replicate_assays.csv", index=False)

# ---- (7) plate as a batch: per-assay plate test (negative control) ----------
pp = np.array([stats.mannwhitneyu(P.loc[meta.sample_plate == 1, c], P.loc[meta.sample_plate == 2, c]).pvalue
               for c in P.columns])
pd.DataFrame({"assay": P.columns, "p_plate": pp, "q_plate": C.bh(pp)}).to_csv(
    C.TAB / "R01_plate_effect_proteins.csv", index=False)

# ---- (8) pre-analytical marker scores ---------------------------------------
plat = [m for m in C.PLATELET_MARKERS if m in P.columns]
meta["platelet_score"] = Z[plat].mean(1)
intra = L.index[L.location == "intracellular"]
meta["intracellular_score"] = Z[intra].mean(1)
meta["secreted_score"] = Z[L.index[L.location == "secreted"]].mean(1)
meta.to_csv(C.TAB / "R01_sample_level_metadata.csv")

# ---- figure source data ------------------------------------------------------
meta[["sample_plate", "shift", "frac_above_LOD", "proteome_PC1", "platelet_score",
      "intracellular_score", "secreted_score", "qc_warn_frac"]].to_csv(C.FIG / "FR1a_sample_shift_by_plate.csv")
blk_corr.to_csv(C.FIG / "FR1b_block_concordance.csv")
L[["rho_with_shift", "location"]].to_csv(C.FIG / "FR1c_loading_by_location.csv")
pd.DataFrame({"PC": np.arange(1, 11), "variance_explained": pca.explained_variance_ratio_}).to_csv(
    C.FIG / "FR1d_proteome_scree.csv", index=False)

summary = {
    "n": len(ids), "n_assays": P.shape[1],
    "plate_n": meta.sample_plate.value_counts().to_dict(),
    "PC1_var": float(pca.explained_variance_ratio_[0]),
    "rho_PC1_shift": float(rho_pc1),
    "frac_proteins_pos_with_shift": float((rho_protein > 0).mean()),
    "median_rho_protein_shift": float(rho_protein.median()),
    "block_concordance_mean_r": float(off.mean()), "block_concordance_min_r": float(off.min()),
    "n_plate_sig_q05": int((C.bh(pp) < 0.05).sum()),
    "platelet_markers_used": plat,
    "rho_platelet_shift": float(stats.spearmanr(meta.platelet_score, shift)[0]),
    "rho_intracellular_shift": float(stats.spearmanr(meta.intracellular_score, shift)[0]),
    "rho_secreted_shift": float(stats.spearmanr(meta.secreted_score, shift)[0]),
    "kruskal_location_p": float(kw_loc.pvalue),
}
pd.Series(summary).to_json(C.TAB / "R01_summary.json", indent=2)
print(pd.Series(summary).to_string())
print(assoc.sort_values("p").head(12).to_string())
print(pd.DataFrame(fish))
print(pd.DataFrame(rep_rows))
print(pd.DataFrame(bal)[["variable", "p"]])
