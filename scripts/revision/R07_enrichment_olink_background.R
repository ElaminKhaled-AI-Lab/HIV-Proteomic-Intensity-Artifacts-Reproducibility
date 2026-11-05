#!/usr/bin/env Rscript
# =============================================================================
# R07_enrichment_olink_background.R
# Pathway over-representation and rank-based enrichment with the ASSAYED
# Olink universe as background (Reviewer 1 minor comment 1; Reviewer 3).
#
# Gene sets : MSigDB v7.5.1 via msigdbr (Homo sapiens): KEGG (C2:CP:KEGG),
#             Reactome (C2:CP:REACTOME), GO Biological Process (C5:GO:BP)
# Universe  : unique gene symbols of the Olink Explore 3072 assays analysed
#             (2,923 assays; replicate assays collapsed). A second run uses
#             only assays with >= 50% of samples above LOD.
# Sets      : intersected with the universe; kept if 10 <= size <= 500 after
#             intersection.
# ORA       : one-sided hypergeometric test; BH within each library.
# Ranked    : fgsea (multilevel) on the signed t-like statistic of the
#             protein ~ METS + age + sex model (no cut-off dependence).
# Inputs    : results/revision/tables/R05_protein_METS_<arm>.csv
#             results/revision/tables/R05_cluster_differences_<arm>.csv
# Outputs   : results/revision/tables/R07_*.csv
# =============================================================================
suppressMessages({library(msigdbr); library(fgsea); library(data.table)})
set.seed(20261030)
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
ROOT <- normalizePath(file.path(here, "..", ".."))
TAB <- file.path(ROOT, "results", "revision", "tables")
FIG <- file.path(ROOT, "results", "revision", "figure_data")

load_sets <- function() {
  spec <- list(KEGG = c("C2", "CP:KEGG"), Reactome = c("C2", "CP:REACTOME"), GO_BP = c("C5", "GO:BP"))
  out <- list()
  for (lib in names(spec)) {
    d <- msigdbr(species = "Homo sapiens", category = spec[[lib]][1], subcategory = spec[[lib]][2])
    out[[lib]] <- split(d$gene_symbol, d$gs_name)
  }
  out
}
SETS <- load_sets()

ora <- function(hits, universe, lib, label) {
  hits <- intersect(unique(hits), universe)
  N <- length(universe); n <- length(hits)
  res <- rbindlist(lapply(names(SETS[[lib]]), function(s) {
    g <- intersect(unique(SETS[[lib]][[s]]), universe); K <- length(g)
    if (K < 10 || K > 500) return(NULL)
    k <- length(intersect(hits, g))
    data.table(term = s, k = k, n = n, K = K, N = N,
               p = phyper(k - 1, K, N - K, n, lower.tail = FALSE),
               genes = paste(sort(intersect(hits, g)), collapse = ";"))
  }))
  res[, q := p.adjust(p, "BH")][, library := lib][, list_name := label]
  res[order(p)]
}

run_arm <- function(arm, lod = FALSE) {
  pm <- fread(file.path(TAB, sprintf("R05_protein_METS_%s.csv", arm)))
  cd <- fread(file.path(TAB, sprintf("R05_cluster_differences_%s.csv", arm)))
  ann <- fread(file.path(TAB, "R01_protein_shift_loading.csv"))
  setnames(ann, 1, "protein")
  universe <- unique(pm$protein)
  if (lod) universe <- intersect(universe, ann[frac_above_LOD >= 0.5, protein])
  tag <- if (lod) paste0(arm, "_LOD50") else arm
  mapped <- sum(universe %in% unique(unlist(SETS)))
  out <- list()
  lists <- list(METS_up = pm[q < 0.05 & beta_METS > 0, protein],
                METS_down = pm[q < 0.05 & beta_METS < 0, protein],
                cluster_higher = cd[q < 0.05 & diff_high_minus_low > 0, protein],
                cluster_lower = cd[q < 0.05 & diff_high_minus_low < 0, protein])
  for (nm in names(lists)) for (lib in names(SETS))
    if (length(intersect(lists[[nm]], universe)) >= 5) out[[paste(nm, lib)]] <- ora(lists[[nm]], universe, lib, nm)
  ora_all <- rbindlist(out)
  ora_all[, arm := tag][, universe_size := length(universe)][, universe_mapped_to_any_set := mapped]
  fwrite(ora_all, file.path(TAB, sprintf("R07_ORA_%s.csv", tag)))
  # ranked enrichment on METS statistic
  stat <- with(pm[protein %in% universe], setNames(sign(beta_METS) * qnorm(p / 2, lower.tail = FALSE), protein))
  gs <- list()
  for (lib in names(SETS)) {
    s <- lapply(SETS[[lib]], function(g) intersect(g, universe))
    s <- s[lengths(s) >= 10 & lengths(s) <= 500]
    f <- fgsea(pathways = s, stats = stat, minSize = 10, maxSize = 500, eps = 0)
    f[, library := lib]; gs[[lib]] <- f
  }
  g <- rbindlist(gs); g[, padj_within_library := p.adjust(pval, "BH"), by = library]
  g[, leadingEdge := sapply(leadingEdge, paste, collapse = ";")][, arm := tag]
  fwrite(g[order(pval)], file.path(TAB, sprintf("R07_fgsea_METS_%s.csv", tag)))
  cat(sprintf("%s: universe %d (mapped %d) | ORA terms q<0.05: %d | fgsea q<0.05: %d\n", tag,
              length(universe), mapped, ora_all[q < 0.05, .N], g[padj_within_library < 0.05, .N]))
  invisible(list(ora = ora_all, gsea = g))
}

arms <- c("shift_adjusted", "none", "median_centred", "block_median", "quantile",
          "rank_inverse_normal", "remove_PC1")
res <- lapply(arms, run_arm)
names(res) <- arms
prim_lod <- run_arm("shift_adjusted", lod = TRUE)

# robustness of the terms reported in the submitted manuscript
key <- c("KEGG_LIPID_AND_ATHEROSCLEROSIS", "REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION",
         "REACTOME_HEMOSTASIS", "KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION",
         "REACTOME_IMMUNOREGULATORY_INTERACTIONS_BETWEEN_A_LYMPHOID_AND_A_NON_LYMPHOID_CELL",
         "KEGG_CELL_ADHESION_MOLECULES_CAMS")
rob <- rbindlist(lapply(c(arms, "shift_adjusted_LOD50"), function(a) {
  r <- if (a == "shift_adjusted_LOD50") prim_lod else res[[a]]
  rbind(r$ora[term %in% key, .(arm = a, analysis = paste0("ORA_", list_name), term, k, K, q)],
        r$gsea[pathway %in% key, .(arm = a, analysis = "fgsea_METS", term = pathway, k = size, K = size,
                                   q = padj_within_library, NES = NES)], fill = TRUE)
}), fill = TRUE)
fwrite(rob, file.path(TAB, "R07_key_term_robustness.csv"))
fwrite(rob, file.path(FIG, "FR7_key_term_robustness.csv"))
fwrite(res[["shift_adjusted"]]$ora[q < 0.05][order(q)][, head(.SD, 12), by = .(list_name)],
       file.path(FIG, "FR7b_top_terms_primary.csv"))
cat("done\n")
