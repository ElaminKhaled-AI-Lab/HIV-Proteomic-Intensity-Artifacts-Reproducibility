#!/usr/bin/env Rscript
# =============================================================================
# R06_rna_differential_power.R
# Transcriptome versus the corrected partition and versus METS, with a power
# statement (Reviewer 1 comment 9). limma on the 3,000-gene VST matrix used in
# the integration, plus DESeq2 on the full count matrix when available.
# Reports: number of FDR-significant genes, Storey pi0 (proportion of true
# nulls) from the p-value histogram, and the minimal detectable log2 fold
# change at 80% power for the observed group sizes and residual SD.
# =============================================================================
suppressMessages({library(limma); library(data.table)})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
ROOT <- normalizePath(file.path(here, "..", ".."))
TAB <- file.path(ROOT, "results", "revision", "tables"); FIG <- file.path(ROOT, "results", "revision", "figure_data")
rna_f <- c(file.path(ROOT, "data/processed/integrated/RNA_top_variable.csv"),
           file.path(ROOT, "data/intermediate_public/RNA_top_variable.csv"))
rna_f <- rna_f[file.exists(rna_f)][1]
R <- as.matrix(read.csv(rna_f, row.names = 1, check.names = FALSE))
parts <- read.csv(file.path(TAB, "R05_partitions_all_arms.csv"), row.names = 1)
clin <- fread(file.path(ROOT, "data/intermediate_public/outcomes.csv"))
ids <- rownames(parts); R <- R[ids, ]
mets <- clin$METS[match(ids, as.character(clin$COCOMO_ID))]
pi0 <- function(p, lambda = 0.5) min(1, mean(p > lambda) / (1 - lambda))
res <- list()
for (cmp in c("primary_partition", "METS")) {
  g <- if (cmp == "METS") mets else parts$shift_adjusted
  d <- model.matrix(~ factor(g))
  fit <- eBayes(lmFit(t(R), d))
  tt <- topTable(fit, coef = 2, number = Inf, sort.by = "none")
  sdres <- median(sqrt(fit$s2.post))
  n1 <- sum(g == 1); n0 <- sum(g == 0)
  mdd <- power.t.test(n = 2 / (1 / n1 + 1 / n0), sd = sdres, sig.level = 0.05 / ncol(R), power = 0.8)$delta
  fwrite(data.table(gene = rownames(tt), tt), file.path(TAB, sprintf("R06_limma_%s.csv", cmp)))
  fwrite(data.table(p = tt$P.Value), file.path(FIG, sprintf("FR9_rna_pvalues_%s.csv", cmp)))
  res[[cmp]] <- data.table(comparison = cmp, n_high = n1, n_low = n0, genes = ncol(R),
                           n_FDR05 = sum(tt$adj.P.Val < 0.05), pi0 = pi0(tt$P.Value),
                           median_residual_sd = sdres, min_detectable_log2FC_80pct_bonferroni = mdd)
}
# DESeq2 on full counts if present (all ~20k filtered genes)
cf <- file.path(ROOT, "data/processed/transcriptomics_counts.csv")
if (file.exists(cf) && requireNamespace("DESeq2", quietly = TRUE)) {
  link <- fread(file.path(ROOT, "data/processed/master_patient_linkage.csv"))
  cts <- as.matrix(read.csv(cf, row.names = 1, check.names = FALSE))
  cts <- cts[, link$Run[match(ids, as.character(link$COCOMO_ID))]]
  cts <- cts[rowSums(cts >= 10) >= 0.25 * ncol(cts), ]
  for (cmp in c("primary_partition", "METS")) {
    g <- factor(if (cmp == "METS") mets else parts$shift_adjusted)
    dds <- DESeq2::DESeqDataSetFromMatrix(cts, data.frame(g = g), ~ g)
    r <- as.data.frame(DESeq2::results(DESeq2::DESeq(dds, quiet = TRUE)))
    fwrite(data.table(gene = rownames(r), r), file.path(TAB, sprintf("R06_DESeq2_%s.csv", cmp)))
    res[[paste0(cmp, "_DESeq2")]] <- data.table(comparison = paste0(cmp, " (DESeq2, all genes)"),
      genes = nrow(r), n_FDR05 = sum(r$padj < 0.05, na.rm = TRUE), pi0 = pi0(na.omit(r$pvalue)))
  }
}
out <- rbindlist(res, fill = TRUE); fwrite(out, file.path(TAB, "R06_rna_summary.csv")); print(out)
