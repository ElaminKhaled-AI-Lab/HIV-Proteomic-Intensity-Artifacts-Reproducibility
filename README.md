# Proteomic Intensity Artifacts in HIV Multi-Omics Integration

Code, analysis-ready matrices, and per-panel figure source data for the manuscript

> **Proteomic Intensity Artifacts in HIV Multi-Omics Integration: Deep Learning Benchmarking and Corrected Metabolic-Risk Endotyping**
> K. M. Elamin, S. M. I. Elbashir, I. Adam. *Proteomes* (MDPI), revised version, 2026.

## Release v2.0 (revision)

All results of the revised manuscript are produced by `scripts/revision/`. One command regenerates every number, table, and figure source file from the public inputs:

```bash
conda env create -f environment.yml && conda activate hiv_integration
bash scripts/preprocessing/R0_recover_rna_matrix_stream.sh   # only if the RNA matrix must be rebuilt from SRA
bash scripts/revision/run_all.sh                             # R01-R10, fixed seeds (SEED = 20261030)
```

Then, in MATLAB (R2023b or later), draw every figure panel:

```matlab
cd matlab
make_revision_figures('/path/to/this/repository')   % writes results/revision/figures_matlab/*.png|pdf|fig
```

| Script | Content | Manuscript |
|---|---|---|
| `common.py` | paths, seeds, data loaders (Olink matrix rebuilt from the vendor NPX export; replicate assays averaged), statistics helpers | all |
| `vae.py` | product-of-experts VAE (native masking, optional SNF anchor) | 2.4 |
| `R01_olink_shift_origin.py` | global intensity score vs plate, well, QC, LOD, panel blocks, subcellular location, platelet score, clinical variables | 3.4, Fig. 4, Table S3 |
| `R02_nested_benchmark.py` | fully nested CV benchmark (inner-CV tuning, corrected repeated-CV t-test, whole-pipeline permutation); resumable with `--repeats a-b` and `--summarize_only` | 3.2, Fig. 2 |
| `R03_learning_curve.py` | sample-size dependence of PCA and PoE-VAE | Fig. 2C |
| `R04_diagnostic_simulation.py` | semi-synthetic simulation of the four layer-dominance checks, threshold sweeps, concordance | 3.5, Fig. 5 |
| `R05_endotypes_normalization_clinical.py` | eight normalization arms, partitions, checks, cluster existence (SigClust-type, gap), stability, split-half replication, continuous axis, clinical table, adjusted models | 3.3, 3.6, 3.7, Figs. 3, 6, 7, Tables S4, S5 |
| `R06_rna_differential_power.R` | limma (3,000 genes) and DESeq2 (all genes), proportion of true nulls, minimal detectable fold change | 3.8 |
| `R07_enrichment_olink_background.R` | ORA and fgsea with the Olink assay universe as background (MSigDB v7.5.1) | 3.8, Fig. 8, Table S6 |
| `R08_missing_modality_nested.py` | nested missing-modality experiment | 3.9, Fig. 9 |
| `R09_druggability_direct_targets.py` | ChEMBL direct targets vs METS associations | Section S5, Table S8 |
| `R10_collect_numbers.py` | writes `results/revision/manuscript_numbers.json`, the single source of every number in the text | all |

Outputs: `results/revision/tables/` (all tables), `results/revision/figure_data/` (one CSV per figure panel), `results/revision/manuscript_numbers.json`. Table S7 of the Supplementary Information maps every manuscript result to its output file.

Software: Python 3.11+ (PyTorch, scikit-learn, statsmodels, pandas, NumPy, SciPy); R 4.3+ (tximport, DESeq2, limma, msigdbr 7.5.1, fgsea); fastp 1.4.0; Salmon 2.8.0 (decoy-aware GENCODE v50 index). Versions are logged at run time.

### Corrections relative to the submitted version (v1.0)

The v1.0 scripts (listed below) are kept for transparency. In v1.0, representations were fitted before cross-validation, the six Olink assays measured in every panel were summed instead of averaged, enrichment used a genome background, and several reported counts came from superseded runs. These issues are corrected in v2.0; the response to reviewers lists every change.

---

## Submitted version (v1.0) scripts

## Overview

The study assembles a linked four-layer multi-omics resource (transcriptomics, Olink proteomics, metabolomics, clinical) for 89 antiretroviral-treated people living with HIV, benchmarks a product-of-experts variational autoencoder against linear integration, and shows that the initially derived endotypes were driven by a global per-sample intensity shift in the proteomic layer. The repository provides:

- the RNA-seq preprocessing pipeline (raw reads to gene counts),
- the participant-linkage reconstruction that joins the public deposits,
- the integration models and the outcome-anchored benchmark,
- the four-test artifact-diagnostic framework and the normalization-strategy benchmark,
- endotype derivation, pathway enrichment, and drug mapping,
- the missing-modality experiment,
- per-panel figure source data.

---

## Repository structure

```
.
├── README.md
├── LICENSE                     # MIT (code)
├── LICENSE-DATA                # CC-BY-4.0 (derived figure data)
├── CITATION.cff                # how to cite this repository
├── requirements.txt            # Python environment
├── environment.yml             # conda environment (equivalent)
├── data/
│   ├── README.md               # how to obtain the public source data
│   ├── source_manifest.tsv     # exact data provenance
│   └── intermediate_public/    # reviewer-facing transformed matrices
├── scripts/
│   ├── preprocessing/          # 00–06: RNA-seq + linkage + assembly
│   │   ├── 00_build_salmon_index.sh
│   │   ├── 01_quantify_rna_salmon.sh
│   │   ├── 02_build_count_matrix.R
│   │   ├── 03_normalize_qc.R
│   │   ├── 04_build_patient_linkage.R
│   │   ├── 05_assemble_integrated_dataset.R
│   │   └── 06_build_outcomes_table.R
│   ├── mosaic_stage1_core.py            # PoE-VAE
│   ├── mosaic_stage2_anchor.py          # anchor sweep + baselines
│   ├── mosaic_stage2b_rigor.py          # bootstrap CIs
│   ├── mosaic_stage3_endotypes.py       # PCA vs MOFA-like, endotypes
│   ├── mosaic_stage4_characterize.py    # differential features
│   ├── mosaic_stage4b_artifact_check.py # 4-test artifact diagnostic
│   ├── mosaic_stage5_fix_and_verify.py  # correction + verification
│   ├── mosaic_stage6_pathways.py        # enrichment
│   ├── mosaic_stage7_drugs.py           # LINCS reversal (negative result)
│   ├── mosaic_stage7b_drugs_refined.py  # refined reversal
│   ├── mosaic_stage8_pathway_drugs.py   # DGIdb pathway-targeted (used)
│   ├── exp_A_missing_modality.py        # Figure 6
│   ├── exp_B_batch_correction.py        # Figure 7
│   └── export_figure_data.py            # writes per-panel CSVs
└── results/
    └── figure_data/            # per-panel source CSVs (one per figure panel)
```

---

## Data availability

Raw sequencing files and the complete original source repository are not duplicated here. Public primary sources are cited below. For peer-review reproducibility, selected analysis-ready transformed matrices may be released in `data/intermediate_public/` after verification that redistribution is permitted and that no direct identifiers are present:

| Layer | Source | Accession |
|---|---|---|
| Transcriptomics (raw reads) | NCBI Sequence Read Archive | BioProject **PRJNA983231** |
| Proteomics (Olink), metabolomics, clinical | Public COCOMO analysis repository | see `data/README.md` |

The participant-linkage step (`scripts/preprocessing/04_build_patient_linkage.R`) reconstructs the mapping between sequencing runs and the cohort identifier. Exact source-data provenance is listed in `data/source_manifest.tsv`. Reviewers can reproduce the full workflow or start from the intermediate matrices described in `data/intermediate_public/README.md`.

`results/figure_data/` contains the derived, non-identifying per-panel values used to render each figure; these are released under CC-BY-4.0.

---

## Reproducing the analysis

### 1. Environment

```bash
# option A: conda
conda env create -f environment.yml
conda activate hiv_integration

# option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The supplied Conda environment includes Python 3.11, R 4.3, Salmon, fastp, FastQC, MultiQC, SRA Toolkit, `tximport`, and `DESeq2`, together with the Python analysis dependencies.

### 2. Obtain the public data

Follow `data/README.md` to download the SRA reads and the COCOMO proteomic, metabolomic and clinical matrices into `data/`.

### 3. Preprocess and assemble

```bash
cd scripts/preprocessing
bash   00_build_salmon_index.sh
bash   01_quantify_rna_salmon.sh
Rscript 02_build_count_matrix.R
Rscript 03_normalize_qc.R
Rscript 04_build_patient_linkage.R
Rscript 05_assemble_integrated_dataset.R
Rscript 06_build_outcomes_table.R
```

### 4. Run the analysis stages

```bash
cd ../..
python scripts/mosaic_stage2_anchor.py         # benchmark (Fig 2)
python scripts/mosaic_stage4b_artifact_check.py # artifact diagnostics (Fig 3)
python scripts/mosaic_stage5_fix_and_verify.py  # correction
python scripts/mosaic_stage6_pathways.py        # pathways (Fig 5a,b)
python scripts/mosaic_stage8_pathway_drugs.py   # drugs (Fig 5c,d)
python scripts/exp_A_missing_modality.py        # Fig 6
python scripts/exp_B_batch_correction.py        # Fig 7
python scripts/export_figure_data.py            # writes results/figure_data/
```

### 5. Figure-level source data

The numerical source data underlying the study figures are provided in `results/figure_data/` as per-panel CSV files. These files support direct inspection of the plotted values and independent replotting using compatible statistical or graphics software.

No figure-generation software-specific files are included in this repository.

---

## Notes and caveats

- **Paths.** Project-root paths are resolved automatically from the repository location; no user-specific absolute path is required.
- **Drug candidates** are computational hypotheses only and are not clinical recommendations; antiretroviral-interaction annotations are a screening aid requiring pharmacological verification.
- **Batch correction** is empirical (per-sample), as no plate or batch metadata accompany the deposited proteomic matrix.
- **Determinism.** Random seeds are fixed, but exact numerical values may vary slightly across BLAS/library versions.

---

## Citing this work

If you use this code or the diagnostic framework, please cite the manuscript (details in `CITATION.cff`). A DOI for this repository can be minted via Zenodo on release.

## License

Code is released under the MIT License (`LICENSE`). Derived per-figure data in `results/figure_data/` are released under CC-BY-4.0 (`LICENSE-DATA`).

## Contact

Corresponding author: Prof. Ishag Adam — ia.ahmed@qu.edu.sa
Code author: K. M. Elamin — Kumamoto University.
