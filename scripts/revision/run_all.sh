#!/usr/bin/env bash
# =============================================================================
# run_all.sh: regenerate EVERY number, table and figure source file of the
# revised manuscript from the public inputs, in order, with fixed seeds.
# Prerequisites: data/processed/integrated/RNA_top_variable.csv (from
# scripts/preprocessing/R0_recover_rna_matrix.sh or 00-05), the cloned COCOMO
# source repository in external/Immunometabolism_HIV, Python env
# (environment.yml) and an R env with msigdbr 7.5.1, limma, fgsea.
# Runtime on 32 cores: ~1.5 h.  Logs: results/revision/logs/
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
L=../../results/revision/logs; mkdir -p "$L"
run(){ echo "=== $1 ==="; "${@:2}" 2>&1 | tee "$L/$1.log"; }
run R01 python R01_olink_shift_origin.py
run R02 python R02_nested_benchmark.py --n_perm 200
run R03 python R03_learning_curve.py
run R04 python R04_diagnostic_simulation.py --reps 200
run R05 python R05_endotypes_normalization_clinical.py --n_stab 500 --n_null 1000 --n_split 200
run R06 Rscript R06_rna_differential_power.R
run R07 Rscript R07_enrichment_olink_background.R
run R08 python R08_missing_modality_nested.py --masks 10
run R09 python R09_druggability_direct_targets.py
run R10 python R10_collect_numbers.py
echo "All revision analyses complete. Build figures in MATLAB with:"
echo "  >> make_revision_figures('$(cd ../.. && pwd)')"
