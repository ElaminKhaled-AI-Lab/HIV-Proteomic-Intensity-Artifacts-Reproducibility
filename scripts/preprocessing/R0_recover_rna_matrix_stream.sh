#!/usr/bin/env bash
# =============================================================================
# R0_recover_rna_matrix.sh
# Rebuild the deleted transcriptomic matrices exactly as in the original study:
#   data/processed/transcriptomics_counts.csv / _tpm.csv / _vst.csv
#   data/processed/integrated/RNA_top_variable.csv   (89 x 3,000)
#   data/intermediate_public/RNA_top_variable.csv    (reviewer copy)
#
# Uses the existing decoy-aware Salmon index (GENCODE v50) already on disk,
# so step 00 is NOT rerun. FASTQs are re-downloaded from SRA (PRJNA983231),
# quantified one run at a time, and deleted after quantification to save space.
#
# Usage (from the repository root):
#   conda activate hiv_integration      # needs sra-tools, fastp, salmon, R (tximport, DESeq2)
#   bash scripts/preprocessing/R0_recover_rna_matrix.sh  2>&1 | tee logs_R0_recover.txt
# Runtime: download-bound (~460 GB from SRA); 4 runs in parallel by default
# (set PAR=6 bash ... for more). Safe to stop and restart: finished
# runs are skipped.
# =============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ"
RUNINFO="data/metadata/PRJNA983231_SraRunInfo.csv"
FASTQ="data/raw/cocomo158_rna/fastq"
QUANT="data/interim/rna_quantification"
INDEX="data/raw/reference/salmon_index_gencode_v50"
THREADS="$(nproc)"
mkdir -p "$FASTQ" "$QUANT" data/processed/integrated data/interim/rna_qc/trimmed

for tool in prefetch fasterq-dump fastp salmon Rscript; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found in PATH"; exit 1; }
done
index_ok(){ { [ -f "$INDEX/versionInfo.json" ] && [ -f "$INDEX/refInfo.json" ]; } || \
            { [ -f "$INDEX/info.json" ] && [ -f "$INDEX/index.ssi" ] && [ -f "$INDEX/refseq.bin" ]; }; }
if ! index_ok; then
  echo "Salmon index at $INDEX is missing or incomplete -> rebuilding (one time, ~1-2 h, ~25 GB RAM)"
  rm -rf "$INDEX"
  bash scripts/preprocessing/00_build_salmon_index.sh
  index_ok || { echo "ERROR: index build failed"; exit 1; }
fi

# Run accessions from the SRA run table (column 'Run')
RUNS=$(python3 - "$RUNINFO" <<'PY'
import csv,sys
print(" ".join(r["Run"] for r in csv.DictReader(open(sys.argv[1])) if r.get("Run","").startswith("SRR")))
PY
)
echo "Runs listed in run table: $(echo $RUNS | wc -w)"

PAR="${PAR:-4}"                         # runs processed concurrently
TPR=$(( THREADS / PAR )); [ "$TPR" -lt 2 ] && TPR=2
# scratch for trimmed reads: a fast local disk if it has >=150 GB free, else the project disk
WORK="${WORK:-}"
if [ -z "$WORK" ]; then
  free_tmp=$(df -BG --output=avail "${TMPDIR:-/tmp}" | tail -1 | tr -dc 0-9)
  if [ "${free_tmp:-0}" -ge 150 ]; then WORK="${TMPDIR:-/tmp}/hiv_rna_work"; else WORK="data/interim/rna_qc/trimmed"; fi
fi
mkdir -p "$WORK"; echo "scratch for trimmed reads: $WORK"
export FASTQ QUANT INDEX TPR WORK
process_run(){
  set -euo pipefail
  SRR="$1"; T="$WORK"
  if [ -f "$QUANT/$SRR/quant.sf" ]; then echo "[skip] $SRR already quantified"; return 0; fi
  echo "=== $SRR : download ==="
  if ! prefetch "$SRR" -O "$FASTQ" --max-size 100G > "$FASTQ/$SRR.prefetch.log" 2>&1; then
     echo "WARNING: download failed for $SRR (the original study also lost 1 of 96)"; return 0; fi
  echo "=== $SRR : stream fasterq-dump -> fastp ==="
  # reads are streamed (interleaved) straight into fastp: no raw FASTQ on disk,
  # no separate compression pass. fastp settings are unchanged (default adapter
  # detection, paired-end mode); output written with light compression.
  fasterq-dump "$FASTQ/$SRR" --split-spot --stdout --skip-technical --threads 4 \
               -t "$T/tmp_$SRR" 2> "$T/${SRR}.fasterq.log" | \
  fastp --stdin --interleaved_in \
        --out1 "$T/${SRR}_1.trim.fastq.gz" --out2 "$T/${SRR}_2.trim.fastq.gz" --compression 1 \
        --json "$T/${SRR}.fastp.json" --html "$T/${SRR}.fastp.html" --thread "$TPR" 2> "$T/${SRR}.fastp.log"
  rm -rf "$FASTQ/$SRR" "$T/tmp_$SRR"
  echo "=== $SRR : salmon ==="
  salmon quant --index "$INDEX" --libType A \
        --mates1 "$T/${SRR}_1.trim.fastq.gz" --mates2 "$T/${SRR}_2.trim.fastq.gz" \
        --validateMappings --gcBias --seqBias --threads "$TPR" --output "$QUANT/$SRR" > "$QUANT/$SRR.salmon.log" 2>&1
  mkdir -p data/interim/rna_qc/fastp_reports
  mv "$T/${SRR}.fastp.json" "$T/${SRR}.fastp.html" data/interim/rna_qc/fastp_reports/ 2>/dev/null || true
  rm -f "$T/${SRR}"_?.trim.fastq.gz
  echo "=== $SRR : done ($(grep -o '"percent_mapped": [0-9.]*' "$QUANT/$SRR/aux_info/meta_info.json" 2>/dev/null)) ==="
}
export -f process_run
# a stale lock from an interrupted prefetch blocks a restart
rm -f "$FASTQ"/*/*.sra.lock
echo $RUNS | tr ' ' '\n' | xargs -P "$PAR" -I{} bash -c 'process_run {}'
echo "Quantified runs: $(ls "$QUANT"/*/quant.sf 2>/dev/null | wc -l)"

echo "=== tximport -> counts ==="; Rscript scripts/preprocessing/02_build_count_matrix.R
echo "=== filter + VST ===";       Rscript scripts/preprocessing/03_normalize_qc.R

echo "=== top-3,000 variable genes for the 89 linked participants ==="
Rscript - <<'RS'
link <- read.csv("data/processed/master_patient_linkage.csv", stringsAsFactors = FALSE)
vst  <- as.matrix(read.csv("data/processed/transcriptomics_vst.csv", row.names = 1, check.names = FALSE))
vst  <- vst[, colnames(vst) %in% link$Run, drop = FALSE]
colnames(vst) <- setNames(as.character(link$COCOMO_ID), link$Run)[colnames(vst)]
ids  <- as.character(link$COCOMO_ID)
stopifnot(all(ids %in% colnames(vst)))
v    <- apply(vst[, ids], 1, var)
top  <- names(sort(v, decreasing = TRUE))[1:3000]
rna  <- t(vst[top, ids])
write.csv(rna, "data/processed/integrated/RNA_top_variable.csv")
write.csv(rna, "data/intermediate_public/RNA_top_variable.csv")
cat("RNA_top_variable:", nrow(rna), "x", ncol(rna), "\n")
RS
sha256sum data/processed/transcriptomics_vst.csv data/processed/integrated/RNA_top_variable.csv | tee data/processed/RNA_checksums.sha256
echo "Recovery complete."
