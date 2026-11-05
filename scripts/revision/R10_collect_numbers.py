#!/usr/bin/env python3
"""
R10_collect_numbers.py
Collects every number quoted in the revised manuscript and response letter
from the result files into results/revision/manuscript_numbers.json, so that
no value is typed by hand. Run after R01-R09.
"""
import json

import numpy as np
import pandas as pd

import common as C

T, F = C.TAB, C.FIG
N = {}


def f(x, d=2):
    return f"{x:.{d}f}"


def p_fmt(p):
    if p < 0.001:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        return f"{m:.1f} × 10^{{{e}}}"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


ids = C.participant_ids()
N["n"] = len(ids)
N["provisional"] = len(ids) != 89

# ---------------- R01
r1 = json.load(open(T / "R01_summary.json"))
assoc = pd.read_csv(T / "R01_shift_associations.csv").set_index("variable")
loc = pd.read_csv(T / "R01_location_top200_enrichment.csv").set_index("location")
rep = pd.read_csv(T / "R01_replicate_assays.csv")
bal = pd.read_csv(T / "R01_plate_balance.csv").set_index("variable")
N.update(
    plate1=r1["plate_n"]["1"], plate2=r1["plate_n"]["2"],
    plate_p=f(assoc.loc["sample_plate", "p"]),
    pc1_var=f(100 * r1["PC1_var"], 1), rho_pc1=f(r1["rho_PC1_shift"]),
    frac_pos=f(100 * r1["frac_proteins_pos_with_shift"], 1),
    block_r=f(r1["block_concordance_mean_r"]), block_min=f(r1["block_concordance_min_r"]),
    block_max=f(pd.read_csv(F / "FR1b_block_concordance.csv", index_col=0).values[np.triu_indices(8, 1)].max()),
    rep_r_min=f(rep.median_between_block_r.min()), rep_r_max=f(rep.median_between_block_r.max()),
    lod_rho=f(assoc.loc["fraction_assays_above_LOD", "statistic"]),
    plat_rho=f(r1["rho_platelet_shift"]),
    intra_top=f(100 * loc.loc["intracellular", "top200_frac"], 0), intra_bg=f(100 * loc.loc["intracellular", "background_frac"], 0),
    intra_or=f(loc.loc["intracellular", "odds_ratio"]), intra_p=p_fmt(loc.loc["intracellular", "p"]),
    sec_top=f(100 * loc.loc["secreted", "top200_frac"], 0), sec_bg=f(100 * loc.loc["secreted", "background_frac"], 0),
    sec_or=f(loc.loc["secreted", "odds_ratio"]), sec_p=p_fmt(loc.loc["secreted", "p"]),
    cd8_rho=f(assoc.loc["CD8", "statistic"]), cd8_p=f(assoc.loc["CD8", "p"], 3), cd8_q=f(assoc.loc["CD8", "q_BH"]),
    plate_bal_mets_p=f(bal.loc["METS", "p"]), plate_bal_min_p=f(bal.p.min()),
    n_plate_sig=r1["n_plate_sig_q05"], n_platelet_markers=len(r1["platelet_markers_used"]),
)

# ---------------- R04
oc = pd.read_csv(T / "R04_operating_characteristics.csv")
o89 = oc[oc.n == 89].set_index("rule")
fr = pd.read_csv(T / "R04_flag_rates.csv", keep_default_na=False, na_values=[""])
sim = pd.read_csv(T / "R04_simulation_replicates.csv", keep_default_na=False, na_values=[""])
bm = sim[sim.scenario == "bio_multilayer"]
nf = ((bm.p_median < 1e-3).astype(int) + ((bm.n_sig >= 10) & (bm.imbalance > .9)).astype(int) +
      (bm.ari_survival < .2).astype(int) + (bm.ari_dominance > .7).astype(int))
phi = pd.read_csv(T / "R04_flag_concordance_phi.csv", index_col=0).values
nul = fr[(fr.scenario == "null") & (fr.n == 89)].iloc[0]
N.update(sim_bio_zero=int((nf == 0).sum()), sim_bio_total=len(nf), sim_bio_max=int(nf.max()),
         sim_sens3=f(100 * o89.loc[">=3 flags", "sensitivity"], 1), sim_sens4=f(100 * o89.loc[">=4 flags", "sensitivity"], 1),
         sim_ff4=f(100 * o89.loc[">=4 flags", "false_flag_rate_when_biology_recovered"], 1),
         sim_ff3=f(100 * o89.loc[">=3 flags", "false_flag_rate_when_biology_recovered"], 1),
         sim_nbio=int(o89.loc[">=4 flags", "n_biology"]),
         sim_null_ge3=f(100 * nul.ge3, 0),
         phi_min=f(phi[np.triu_indices(4, 1)].min()), phi_max=f(phi[np.triu_indices(4, 1)].max()),
         sim_reps=int(fr.reps.max()), sim_cells=len(fr))

# ---------------- R05
arms = pd.read_csv(T / "R05_normalization_arms.csv").set_index("arm")
ex = json.load(open(T / "R05_cluster_existence.json"))
st = json.load(open(T / "R05_stability.json"))
sh = json.load(open(T / "R05_split_half_summary.json"))
ax = json.load(open(T / "R05_axis_OR.json"))
pre = json.load(open(T / "R05_preanalytical_checks.json"))
adj = pd.read_csv(T / "R05_adjusted_METS_models.csv")
ct = pd.read_csv(T / "R05_clinical_table_primary.csv").set_index("variable")
a0, a1 = arms.loc["none"], arms.loc["shift_adjusted"]
N.update(
    init_sig=int(a0.n_sig), init_hi=int(max(a0.n_up, a0.n_down)), init_lo=int(min(a0.n_up, a0.n_down)),
    init_pct_one_dir=f(100 * max(a0.n_up, a0.n_down) / a0.n_sig, 1),
    init_pmed=p_fmt(a0.p_median), init_ari_wo=f(a0.ari_without_prot, 3), init_ari_p=f(a0.ari_prot_only, 3),
    init_flags=int(a0.n_flags), init_size_hi=int(a0.size_high), init_size_lo=int(a0.size_low),
    init_axis_rho=f(pre["rho_initial_axis_shift"]),
    prim_flags=int(a1.n_flags), prim_sig=int(a1.n_sig), prim_up=int(a1.n_up), prim_down=int(a1.n_down),
    prim_size_hi=int(a1.size_high), prim_size_lo=int(a1.size_low),
    prim_mets_hi=f(100 * a1.METS_high, 0), prim_mets_lo=f(100 * a1.METS_low, 0),
    prim_or=f(a1.OR_METS), prim_or_lo=f(a1.OR_lo), prim_or_hi=f(a1.OR_hi), prim_or_p=f(a1.p_METS_fisher, 3),
    prim_ari_wo=f(a1.ari_without_prot, 3), prim_ari_p=f(a1.ari_prot_only, 3), prim_pmed=f(a1.p_median),
    prim_sil=f(a1.silhouette_k2), init_sil=f(a0.silhouette_k2),
    sigclust_p=f(ex["p_sigclust_gaussian"]), gap_k=ex["gap_selected_k"],
    stab_full=f(st["ari_vs_full_median"]), stab_pair=f(st["pairwise_ari_median"]), stab_null=f(st["null_pairwise_ari_median"]),
    stab_p=f(st["p_stability_vs_null"]),
    split_ari=f(sh["median_ari"], 3), split_or=f(sh["median_OR_METS_held_out"]), split_frac=f(100 * sh["frac_OR_gt1"], 0),
    axis_or=f(ax["OR_per_SD"]), axis_lo=f(ax["lo"]), axis_hi=f(ax["hi"]), axis_p=f(ax["p"], 3),
    rho_axis_plat=f(pre["rho_axis_platelet"]), rho_axis_intra=f(pre["rho_axis_intracellular"]),
    rho_axis_shift=f(pre["rho_axis_shift"]), rho_eff_load=f(pre["rho_corrected_effect_vs_shift_loading"]),
    n_prot_mets=int(a1.n_protein_METS_q05),
    jac_min=f(arms.drop(["shift_adjusted", "remove_PC1"]).Jaccard_METS_proteins_vs_primary.min()),
    jac_max=f(arms.drop(["shift_adjusted", "remove_PC1"]).Jaccard_METS_proteins_vs_primary.max()),
    jac_pc1=f(arms.loc["remove_PC1", "Jaccard_METS_proteins_vs_primary"]),
    rhoeff_min=f(arms.drop(["shift_adjusted"]).rho_METS_effects_vs_primary.min()),
    rhoeff_max=f(arms.drop(["shift_adjusted"]).rho_METS_effects_vs_primary.max()),
    ari_arm_min=f(arms.drop(["shift_adjusted"]).ARI_vs_primary.min(), 2),
    ari_arm_max=f(arms.drop(["shift_adjusted"]).ARI_vs_primary.max(), 2),
    ari_lod=f(arms.loc["shift_adjusted_LOD", "ARI_vs_primary"]),
    or_arm_min=f(arms.OR_METS.min()), or_arm_max=f(arms.OR_METS.max()),
    n_clin=len(ct), clin_min_q=f(ct.q_BH_all_clinical.min()),
    hdl_g=f(ct.loc["Hdl", "estimate"]), hdl_p=f(ct.loc["Hdl", "p"]),
    cd8_g=f(ct.loc["CD8", "estimate"]), cd8_pp=f(ct.loc["CD8", "p"]),
    tg_g=f(ct.loc["Tgl", "estimate"]), tg_p=f(ct.loc["Tgl", "p"]),
    vat_g=f(ct.loc["VAT", "estimate"]), vat_p=f(ct.loc["VAT", "p"]),
    dur_g=f(ct.loc["Duration", "estimate"]), dur_p=f(ct.loc["Duration", "p"]),
)
for i, r in adj.iterrows():
    N[f"adj{i}_or"], N[f"adj{i}_lo"], N[f"adj{i}_hi"], N[f"adj{i}_p"], N[f"adj{i}_n"] = \
        f(r.OR), f(r.lo), f(r.hi), f(r.p, 3), int(r.n)
for a in arms.index:
    N[f"arm_{a}_flags"] = int(arms.loc[a, "n_flags"])

# ---------------- R06 / R07 / R09
r6 = pd.read_csv(T / "R06_rna_summary.csv").set_index("comparison")
N.update(rna_mets_n=int(r6.loc["METS", "n_FDR05"]), rna_mets_pi0=f(r6.loc["METS", "pi0"]),
         rna_mdd=f(r6.loc["METS", "min_detectable_log2FC_80pct_bonferroni"]))
try:
    sym = pd.read_csv(C.DATA / "processed" / "gencode_v50_gene_symbols.csv")
    lm = pd.read_csv(T / "R06_limma_METS.csv").merge(sym, left_on="gene", right_on="gene_id", how="left")
    s = lm[lm["adj.P.Val"] < 0.05]
    N.update(rna_mets_down=int((s.logFC < 0).sum()))
    if "METS (DESeq2, all genes)" in r6.index:
        N["rna_deseq_mets"] = int(r6.loc["METS (DESeq2, all genes)", "n_FDR05"])
        N["rna_deseq_genes"] = int(r6.loc["METS (DESeq2, all genes)", "genes"])
except FileNotFoundError:
    pass
rob = pd.read_csv(T / "R07_key_term_robustness.csv")
g = rob[(rob.analysis == "fgsea_METS")]
def gq(term, arm):
    x = g[(g.term == term) & (g.arm == arm)]
    return (f(x.q.iloc[0], 3), f(x.NES.iloc[0])) if len(x) else ("NA", "NA")
N["plt_q"], N["plt_nes"] = gq("REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION", "shift_adjusted")
N["plt_q_lod"], _ = gq("REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION", "shift_adjusted_LOD50")
N["hem_q"], N["hem_nes"] = gq("REACTOME_HEMOSTASIS", "shift_adjusted")
N["cyt_q"], N["cyt_nes"] = gq("KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION", "shift_adjusted")
ora = pd.read_csv(T / "R07_ORA_shift_adjusted.csv")
up = ora[(ora.list_name == "METS_up")].sort_values("q")
N["ora_up_n_terms"] = int((up.q < 0.05).sum())
N["ora_up_top"] = up.term.iloc[0]; N["ora_up_top_q"] = f(up.q.iloc[0], 3)
N["ora_up_top_k"], N["ora_up_n"], N["ora_up_K"] = int(up.k.iloc[0]), int(up.n.iloc[0]), int(up.K.iloc[0])
N["n_mets_up"] = int(up.n.iloc[0])
cc = up[up.term == "KEGG_COMPLEMENT_AND_COAGULATION_CASCADES"].iloc[0]
N["ora_cc_k"], N["ora_cc_K"], N["ora_cc_q"] = int(cc.k), int(cc.K), f(cc.q, 3)
N["ora_up_sig_terms"] = up[up.q < 0.05].term.tolist()
N["ora_up_sig"] = [dict(term=r.term, k=int(r.k), K=int(r.K), q=f(r.q, 3)) for r in up[up.q < 0.05].itertuples()]
N["clin_sig"] = [dict(var=r.Index, effect=r.effect, est=f(r.estimate), lo=f(r.ci_low), hi=f(r.ci_high),
                      q=f(r.q_BH_all_clinical, 3), comp=bool(r.METS_component))
                 for r in ct[ct.q_BH_all_clinical < 0.05].itertuples()]
dn = ora[(ora.list_name == "METS_down")]
N["n_mets_down"] = int(dn.n.iloc[0]) if len(dn) else 0
N["ora_down_n_terms"] = int((dn.q < 0.05).sum()) if len(dn) else 0
gs = pd.read_csv(T / "R07_fgsea_METS_shift_adjusted.csv")
gs = gs[gs.padj_within_library < 0.05].sort_values("pval")
N["gsea_n_terms"] = len(gs)
k = gs[gs.pathway == "KEGG_CHEMOKINE_SIGNALING_PATHWAY"]
N["chemo_q"], N["chemo_nes"] = (f(k.padj_within_library.iloc[0], 4), f(k.NES.iloc[0])) if len(k) else ("NA", "NA")
ch = ora[(ora.list_name == "cluster_higher")].sort_values("q")
N["clhi_top"] = ch.term.iloc[0]; N["clhi_q"] = p_fmt(ch.q.iloc[0])
d9 = pd.read_csv(T / "R09_direct_targets.csv")
N["f2r_q"] = f(d9[d9.direct_target == "F2R"].q_METS.iloc[0])
N["serpinc1_q"] = f(d9[d9.direct_target == "SERPINC1"].q_METS.iloc[0])

# ---------------- R02 / R03 / R08 (if present)
try:
    b = pd.read_csv(T / "R02_benchmark_summary.csv").set_index("method")
    nm = {"PCA (concatenated)": "pca", "PoE-VAE (unanchored, tuned)": "vae", "PoE-VAE (SNF-anchored, lambda = 1)": "vaea",
          "Per-layer SVD": "svd", "Elastic net (early concatenation)": "en", "Clinical covariates only": "clin",
          "Clinical + PCA (shift-adjusted)": "clinpca", "PCA (transcriptomics only)": "rna", "PCA (proteomics only)": "prot",
          "PCA (metabolomics only)": "met", "PCA (transcriptomics + metabolomics)": "rnamet",
          "PCA (concatenated), shift-adjusted proteomics": "pcac", "PCA (proteomics only), shift-adjusted": "protc"}
    for m, k_ in nm.items():
        if m in b.index:
            r = b.loc[m]
            N[f"b_{k_}"] = f(r.auc_mean_fold, 3); N[f"b_{k_}_lo"] = f(r.auc_ci_low, 3); N[f"b_{k_}_hi"] = f(r.auc_ci_high, 3)
            N[f"b_{k_}_d"] = f(r.diff_vs_PCA, 3); N[f"b_{k_}_dlo"] = f(r.diff_ci_low, 3); N[f"b_{k_}_dhi"] = f(r.diff_ci_high, 3)
            N[f"b_{k_}_dp"] = f(r.p_diff_corrected, 3) if not np.isnan(r.p_diff_corrected) else "NA"
            N[f"b_{k_}_tr"] = f(r.auc_train_mean)
            N[f"b_{k_}_frac"] = f(100 * r.frac_repeats_better_than_PCA, 0)
    N["b_repeats"] = int(len(list((T / "R02_parts").glob("rep*_folds.csv"))))
    cf = pd.read_csv(T / "R02_vae_selected_configs.csv")
    top = cf.groupby(["latent", "hidden"]).size().sort_values(ascending=False)
    N["vae_cfg_top"] = f"latent {top.index[0][0]}, hidden {top.index[0][1]}"
    N["vae_cfg_top_pct"] = f(100 * top.iloc[0] / top.sum(), 0)
    N["vae_rec_tr"], N["vae_rec_te"] = f(cf.vae_train_rec.mean()), f(cf.vae_test_rec.mean())
    pm = pd.read_csv(T / "R02_permutation_tests.csv").set_index("method")
    N["perm_pca_p"] = f(pm.iloc[0].p_perm, 3); N["perm_vae_p"] = f(pm.iloc[1].p_perm, 3)
    N["perm_pca_n"], N["perm_vae_n"] = int(pm.iloc[0].n_perm), int(pm.iloc[1].n_perm)
except FileNotFoundError:
    pass
try:
    lc = pd.read_csv(F / "FR2c_learning_curve.csv")
    for _, r in lc.iterrows():
        N[f"lc_{r.method.replace('-', '').lower()}_{int(r.n)}"] = f(r["mean"])
except FileNotFoundError:
    pass
try:
    mm = pd.read_csv(F / "FR8_missing_modality.csv")
    pv = mm.pivot(index="missing_frac", columns="method", values="mean")
    N["mm_vae0"], N["mm_vae50"] = f(pv.loc[0.0, "PoE-VAE (native masking)"], 3), f(pv.loc[0.5, "PoE-VAE (native masking)"], 3)
    N["mm_pca0"], N["mm_pca50"] = f(pv.loc[0.0, "PCA + mean imputation"], 3), f(pv.loc[0.5, "PCA + mean imputation"], 3)
    N["mm_knn50"], N["mm_cc50"] = f(pv.loc[0.5, "PCA + kNN imputation"], 3), f(pv.loc[0.5, "Complete case"], 3)
    N["mm_vae_better_levels"] = int((pv["PoE-VAE (native masking)"] > pv["PCA + mean imputation"]).sum())
except FileNotFoundError:
    pass

json.dump(N, open(C.OUT / "manuscript_numbers.json", "w"), indent=1, default=str)
print(json.dumps(N, indent=1, default=str)[:6000])
