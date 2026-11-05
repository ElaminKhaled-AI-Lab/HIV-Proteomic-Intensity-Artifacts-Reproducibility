"""
common.py: shared configuration, data loading and helpers for the revision
pipeline (scripts/revision/R*.py). Every number reported in the revised
manuscript is produced by scripts that import this module, so all of them use
the same participant set, the same layer definitions, the same seeds and the
same direction conventions.

Direction convention (used everywhere): differences are
    "higher-risk group minus lower-risk group", or
    "per unit increase of the predictor",
and a positive value means "higher in the higher-risk group".
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "external" / "Immunometabolism_HIV"
DATA = ROOT / "data"
PUB = DATA / "intermediate_public"
OUT = ROOT / "results" / "revision"
FIG = OUT / "figure_data"
TAB = OUT / "tables"
LOG = OUT / "logs"
for p in (OUT, FIG, TAB, LOG):
    p.mkdir(parents=True, exist_ok=True)

NPX_FILE = EXT / "data" / "Olink" / "Q-01546_Neogi_NPX_2022-06-09.csv"
OLINK_MAP = EXT / "data" / "clinical" / "COCOMID_LabelID Negin.xlsx"
UNIPROT_LOC = EXT / "data" / "uniprot" / (
    "uniprot-compressed_true_download_true_fields_accession_2Creviewed_2C-"
    "2023.01.12-13.19.37.62.tsv")
CLIN_FILE = EXT / "processing" / "clinical_data_clean_with_clusters_and_categories.csv"

# --------------------------------------------------------------------------
# Fixed analysis constants (pre-specified for the revision)
# --------------------------------------------------------------------------
SEED = 20261030
N_OUTER_REPEATS = int(os.environ.get("N_OUTER_REPEATS", 20))   # 20 x 5-fold
N_FOLDS = 5
N_INNER_FOLDS = 4
FEATURE_CAP = 1000          # top-variance features per layer (fold-internal)
N_PCS = 10                  # PCs used for clustering
LOD_MIN_FRAC = 0.50         # assay kept in LOD-filtered arm if >=50% samples > LOD
FDR = 0.05
CLINICAL_CONTINUOUS = ["AGE", "BMI", "VAT", "SAT", "Tgl", "Hdl", "Ldl",
                       "CD4", "CD4_nadir", "CD8", "CDA_CD8", "eGFR", "alat",
                       "Duration", "log10VL_ART"]
CLINICAL_BINARY = ["METS", "central_obesity", "hypertension", "diabetes"]
CLINICAL_CATEGORICAL = ["GENDER", "X3rd_Drug", "first_drug"]
METS_COMPONENTS = {"central_obesity", "hypertension", "diabetes", "Tgl", "Hdl"}
ADJUST_COVARIATES = ["AGE", "GENDER", "BMI", "X3rd_Drug", "first_drug", "CD4"]

# Assays that Olink Explore 3072 measures in every panel block (internal
# replicates). The original matrix SUMMED the replicate NPX values (an
# artefact of reshape2::dcast(fun.aggregate = sum)); here they are averaged.
EXPLORE_REPLICATE_ASSAYS = ["CXCL8", "IDO1", "IL6", "LMOD1", "SCRIB", "TNF"]

# A-priori marker sets for pre-analytical confounding (fixed before analysis;
# only assays present on the panel are used).
PLATELET_MARKERS = ["PPBP", "PF4", "SELP", "GP6", "ITGA2B", "TREML1", "CLEC1B",
                    "GP1BA", "PECAM1", "MPIG6B", "GP5", "TBXAS1"]


def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def log_environment(script: str) -> None:
    import sklearn
    info = {"script": script, "python": sys.version.split()[0],
            "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": stats.__name__ and
            __import__("scipy").__version__, "sklearn": sklearn.__version__,
            "seed": SEED, "N_OUTER_REPEATS": N_OUTER_REPEATS}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    (LOG / f"{Path(script).stem}_environment.json").write_text(
        json.dumps(info, indent=2))


# --------------------------------------------------------------------------
# Participants
# --------------------------------------------------------------------------
def participant_ids() -> list[str]:
    link = pd.read_csv(PUB / "master_patient_linkage.csv")
    ids = link["COCOMO_ID"].astype(int).astype(str).tolist()
    assert len(ids) == 89 and len(set(ids)) == 89
    # Provisional mode: restrict to participants present in the RNA matrix
    # (used only while one RNA-seq run awaits re-quantification).
    if os.environ.get("RNA_COMPLETE_ONLY") == "1":
        for p in (DATA / "processed" / "integrated" / "RNA_top_variable.csv", PUB / "RNA_top_variable.csv"):
            if p.exists():
                have = set(pd.read_csv(p, index_col=0, usecols=[0]).index.astype(str))
                miss = [i for i in ids if i not in have]
                if miss:
                    print(f"[PROVISIONAL] {len(miss)} participant(s) without RNA excluded: {miss}", file=sys.stderr)
                ids = [i for i in ids if i in have]
                break
    return ids


# --------------------------------------------------------------------------
# Olink
# --------------------------------------------------------------------------
def olink_sample_map() -> pd.DataFrame:
    """COCOMO_ID -> Olink SampleID, sample plate (1/2) and well."""
    m = pd.read_excel(OLINK_MAP)
    a = m[["COCOMO ID", "Olink ID Negin", "Olink layout Plate 1"]].copy()
    a.columns = ["COCOMO_ID", "neg", "well"]
    a["sample_plate"] = 1
    b = m[["COCOMO ID.1", "Olink ID Negin.1", "Olink layout Plate 2"]].copy()
    b.columns = ["COCOMO_ID", "neg", "well"]
    b["sample_plate"] = 2
    m = pd.concat([a, b]).dropna(subset=["neg"])
    m["SampleID"] = (m["neg"].astype(str).str.replace("N.", "", regex=False)
                     .astype(int).map(lambda x: f"N{x:03d}"))
    m["COCOMO_ID"] = m["COCOMO_ID"].astype(int).astype(str)
    m["well_row"] = m["well"].str[0]
    m["well_col"] = m["well"].str[1:].astype(int)
    return m.set_index("COCOMO_ID")


def olink_long(ids: list[str] | None = None) -> pd.DataFrame:
    npx = pd.read_csv(NPX_FILE, sep=";", low_memory=False)
    npx = npx[npx["SampleID"].str.match(r"N\d")].copy()
    smap = olink_sample_map()
    s2c = dict(zip(smap["SampleID"], smap.index))
    npx["COCOMO_ID"] = npx["SampleID"].map(s2c)
    if ids is not None:
        npx = npx[npx["COCOMO_ID"].isin(ids)]
    return npx


def olink_matrix(ids: list[str], lod_filter: bool = False) -> pd.DataFrame:
    """Participants x assays NPX matrix built directly from the vendor export.
    Replicate assays measured in several blocks are averaged."""
    npx = olink_long(ids)
    X = npx.pivot_table(index="COCOMO_ID", columns="Assay", values="NPX",
                        aggfunc="mean").loc[ids]
    if lod_filter:
        above = (npx["NPX"] > npx["LOD"]).groupby(npx["Assay"]).mean()
        X = X[[c for c in X.columns if above.get(c, 0) >= LOD_MIN_FRAC]]
    return X


def olink_assay_annotation() -> pd.DataFrame:
    npx = pd.read_csv(NPX_FILE, sep=";", usecols=["Assay", "UniProt", "Panel",
                                                     "NPX", "LOD", "SampleID"])
    npx = npx[npx["SampleID"].str.match(r"N\d")]
    ann = npx.groupby("Assay").agg(UniProt=("UniProt", "first"),
                                   Panel=("Panel", "first"))
    ann["frac_above_LOD"] = (npx["NPX"] > npx["LOD"]).groupby(npx["Assay"]).mean()
    loc = pd.read_csv(UNIPROT_LOC, sep="\t").set_index("From")[
        "Subcellular location [CC]"].fillna("")

    def cls(acc):
        s = ";".join(loc.get(a, "") for a in str(acc).split("_"))
        if s == "":
            return "unannotated"
        if "Secreted" in s:
            return "secreted"
        if "membrane" in s.lower():
            return "membrane"
        return "intracellular"
    ann["location"] = ann["UniProt"].map(cls)
    return ann


def shift_score(P: pd.DataFrame) -> pd.Series:
    """Per-participant global intensity score = median NPX across assays."""
    return P.median(axis=1)


# --------------------------------------------------------------------------
# Other layers
# --------------------------------------------------------------------------
def rna_matrix(ids: list[str]) -> pd.DataFrame:
    for p in (DATA / "processed" / "integrated" / "RNA_top_variable.csv",
              PUB / "RNA_top_variable.csv"):
        if p.exists():
            R = pd.read_csv(p, index_col=0)
            R.index = R.index.astype(str)
            return R.loc[ids]
    alt = os.environ.get("RNA_FALLBACK")
    if alt:
        print(f"[WARNING] using provisional RNA matrix {alt}", file=sys.stderr)
        R = pd.read_csv(alt, index_col=0)
        R.index = R.index.astype(str)
        R = R.loc[ids]
        top = R.var().sort_values(ascending=False).index[:3000]
        return R[top]
    raise FileNotFoundError(
        "RNA_top_variable.csv not found; run "
        "scripts/preprocessing/R0_recover_rna_matrix.sh first.")


def metabolomics_matrix(ids: list[str]) -> pd.DataFrame:
    M = pd.read_csv(PUB / "metabolomics.csv", index_col=0)
    M.index = M.index.astype(str)
    return M.loc[ids]


def clinical(ids: list[str]) -> pd.DataFrame:
    C = pd.read_csv(CLIN_FILE)
    C["COCOMO_ID"] = C["COCOMO_ID"].astype(int).astype(str)
    C = C.set_index("COCOMO_ID").loc[ids]
    if "curr_smoking" in C:
        C["smoker"] = (pd.to_numeric(C["curr_smoking"], errors="coerce") > 0
                       ).astype(float).where(C["curr_smoking"].notna())
    return C


def load_all():
    ids = participant_ids()
    return ids, rna_matrix(ids), olink_matrix(ids), metabolomics_matrix(ids), clinical(ids)


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    o = np.argsort(pv)
    q = pv[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty(n)
    tmp[o] = np.clip(q, 0, 1)
    out[ok] = tmp
    return out


def hedges_g(a, b, n_boot=2000, seed=0):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]

    def g(x, y):
        nx, ny = len(x), len(y)
        sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
        d = (x.mean() - y.mean()) / sp
        return d * (1 - 3 / (4 * (nx + ny) - 9))
    est = g(a, b)
    r = np.random.default_rng(seed)
    bs = [g(r.choice(a, len(a)), r.choice(b, len(b))) for _ in range(n_boot)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cliffs_delta(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def odds_ratio_ci(x_group, y_binary):
    """OR (group1 vs group0) with Haldane-corrected Woolf CI and Fisher p."""
    t = pd.crosstab(np.asarray(x_group), np.asarray(y_binary)).reindex(
        index=[1, 0], columns=[1, 0], fill_value=0).values.astype(float)
    p = stats.fisher_exact(t)[1]
    t = t + 0.5
    lor = np.log(t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0]))
    se = np.sqrt((1 / t).sum())
    return np.exp(lor), np.exp(lor - 1.96 * se), np.exp(lor + 1.96 * se), p


def corrected_resampled_ttest(diffs, n_train, n_test, k, r):
    """Nadeau & Bengio (2003) / Bouckaert & Frank (2004) corrected repeated
    k-fold CV t-test on per-fold paired differences."""
    d = np.asarray(diffs, float)
    m = d.mean()
    var = d.var(ddof=1)
    corr = (1 / (k * r) + n_test / n_train) * var
    t = m / np.sqrt(corr) if corr > 0 else np.nan
    df = k * r - 1
    p = 2 * stats.t.sf(abs(t), df)
    half = stats.t.ppf(0.975, df) * np.sqrt(corr)
    return m, m - half, m + half, p


def zscore_fit(X):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd = np.where(sd < 1e-6, 1.0, sd)   # constant features: leave unscaled
    return mu, sd


def cap_fit(Xz, cap=FEATURE_CAP):
    if Xz.shape[1] <= cap:
        return np.arange(Xz.shape[1])
    return np.sort(np.argsort(np.nanvar(Xz, 0))[::-1][:cap])
