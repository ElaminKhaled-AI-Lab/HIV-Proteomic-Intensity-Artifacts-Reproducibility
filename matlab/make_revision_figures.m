function make_revision_figures(repoRoot)
%MAKE_REVISION_FIGURES  Build every revised/new figure panel of the
%Proteomes revision from results/revision/figure_data/*.csv.
%   make_revision_figures('/path/to/HIV-Proteomic-Intensity-Artifacts-Reproducibility')
% Output: <repo>/results/revision/figures_matlab/<FigN>_<letter>_<slug>.{png,pdf,fig}
% One panel = one figure object = one file set. Assemble plates in a vector editor.
if nargin < 1, repoRoot = fileparts(fileparts(mfilename('fullpath'))); end
addpath(fileparts(mfilename('fullpath')));
D = fullfile(repoRoot, 'results', 'revision', 'figure_data');
T = fullfile(repoRoot, 'results', 'revision', 'tables');
O = fullfile(repoRoot, 'results', 'revision', 'figures_matlab');
S = hivStyle('house');
rd = @(f) readtable(fullfile(D, f), 'VariableNamingRule', 'preserve', 'TextType', 'string');
rt = @(f) readtable(fullfile(T, f), 'VariableNamingRule', 'preserve', 'TextType', 'string');

jobs = {@() fig1(S, rd, O), @() fig2(S, rd, rt, O), @() fig3(S, rd, O), @() fig4(S, rd, O), ...
        @() fig5(S, rd, rt, O), @() fig6(S, rd, O), @() fig7(S, rd, rt, O), @() fig8(S, rd, O), ...
        @() fig9(S, rd, O), @() figS1(S, rd, O)};
names = {'Figure 1', 'Figure 2', 'Figure 3', 'Figure 4', 'Figure 5', 'Figure 6', 'Figure 7', ...
         'Figure 8', 'Figure 9', 'Figure S1'};
failed = {};
for k = 1:numel(jobs)
    try
        jobs{k}();
    catch ME
        close all force;
        failed{end + 1} = sprintf('%s: %s (line %d of %s)', names{k}, ME.message, ...
                                  ME.stack(1).line, ME.stack(1).name); %#ok<AGROW>
    end
end
if ~isempty(failed)
    fprintf(2, '\n%d figure(s) failed. Please report this list:\n', numel(failed));
    fprintf(2, '  %s\n', failed{:});
end
fprintf('All panels written to %s\n', O);
end

% =====================================================================
function fig2(S, rd, rt, O)
% (A) nested AUROC per method with corrected 95% CI
B = rd('FR2a_nested_benchmark.csv');
B = sortrows(B, 'auc_mean_fold', 'ascend');
[f, ax] = hivAxes(S, 'a', 'Nested Cross-Validated Discrimination of Metabolic Syndrome', ...
                  'AUROC (outer folds; corrected 95% CI)', '');
f.Position(3:4) = [30 22];
n = height(B);
xline(ax, 0.5, '--', 'Color', S.grey, 'LineWidth', 2);
for i = 1:n
    isV = contains(B.method(i), 'VAE'); col = S.c(1 + isV, :);
    plot(ax, [B.auc_ci_low(i) B.auc_ci_high(i)], [i i], '-', 'Color', col, 'LineWidth', 3);
    plot(ax, B.auc_mean_fold(i), i, 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', S.ms + 3);
end
set(ax, 'YTick', 1:n, 'YTickLabel', B.method, 'YLim', [0.4 n + 0.6], 'XLim', [0.3 1.0]);
hivExport(f, 'Fig2_a_nested_benchmark', O, S); close(f);

% (B) paired per-fold differences versus PCA (concatenated)
F = rt('R02_per_fold_auroc.csv');
ref = F(F.method == "PCA (concatenated)", :);
show = ["PoE-VAE (unanchored, tuned)", "PoE-VAE (SNF-anchored, lambda = 1)", "Per-layer SVD", ...
        "Elastic net (early concatenation)", "PCA (concatenated), shift-adjusted proteomics"];
[f, ax] = hivAxes(S, 'b', 'Paired Per-Fold Difference Versus PCA', '', '\DeltaAUROC (Method - PCA)');
f.Position(3:4) = [30 20];
yline(ax, 0, '--', 'Color', S.grey, 'LineWidth', 2);
for i = 1:numel(show)
    m = F(F.method == show(i), :);
    [~, ia, ib] = intersect(m.repeat * 10 + m.fold, ref.repeat * 10 + ref.fold);
    hivPoints(ax, i, m.auc_test(ia) - ref.auc_test(ib), S.c(mod(i - 1, 6) + 1, :), S);
end
set(ax, 'XTick', 1:numel(show), 'XTickLabel', ...
    {'VAE (Tuned)', 'VAE (SNF Anchor)', 'Per-Layer SVD', 'Elastic Net', 'PCA (Shift-Adj.)'}, ...
    'XTickLabelRotation', 30, 'XLim', [0.4 numel(show) + 0.6]);
hivExport(f, 'Fig2_b_paired_differences', O, S); close(f);

% (C) learning curve
L = rd('FR2c_learning_curve.csv');
[f, ax] = hivAxes(S, 'c', 'Sample-Size Dependence', 'Participants (n)', 'Cross-Validated AUROC');
yline(ax, 0.5, '--', 'Color', S.grey, 'LineWidth', 2);
meth = unique(L.method, 'stable'); h = gobjects(numel(meth), 1);
for i = 1:numel(meth)
    m = L(L.method == meth(i), :);
    fill(ax, [m.n; flipud(m.n)], [m.p2_5; flipud(m.p97_5)], S.c(i, :), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    h(i) = plot(ax, m.n, m.mean, '-o', 'Color', S.c(i, :), 'LineWidth', S.lw, ...
                'MarkerFaceColor', S.c(i, :), 'MarkerSize', S.ms);
end
legend(h, meth, 'Location', 'southeast', 'Box', 'off', 'FontSize', S.legend);
set(ax, 'XLim', [35 92], 'YLim', [0.4 1]);
hivExport(f, 'Fig2_c_learning_curve', O, S); close(f);
end

% =====================================================================
function fig3(S, rd, O)
% (A) per-sample median NPX by initial and corrected partition
M = rd('FR5d_sample_medians_by_partition.csv');
[f, ax] = hivAxes(S, 'a', 'Per-Sample Median Protein Level', '', 'Median NPX');
hivPoints(ax, 1, M.median_NPX_vendor(M.initial_group == 0), S.down, S);
hivPoints(ax, 2, M.median_NPX_vendor(M.initial_group == 1), S.up, S);
hivPoints(ax, 3.5, M.median_NPX_primary(M.primary_group == 0), S.down, S);
hivPoints(ax, 4.5, M.median_NPX_primary(M.primary_group == 1), S.up, S);
set(ax, 'XTick', [1 2 3.5 4.5], 'XTickLabel', ...
    {sprintf('Lower (n=%d)', sum(M.initial_group == 0)), sprintf('Higher (n=%d)', sum(M.initial_group == 1)), ...
     sprintf('Lower (n=%d)', sum(M.primary_group == 0)), sprintf('Higher (n=%d)', sum(M.primary_group == 1))}, ...
    'XTickLabelRotation', 25, 'XLim', [0.4 5.1]);
yl = ylim(ax);
text(ax, 1.5, yl(2), 'Initial Partition', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', S.annot);
text(ax, 4.0, yl(2), 'Corrected Partition', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', S.annot);
hivExport(f, 'Fig3_a_sample_medians', O, S); close(f);

% (B,C,D) volcano plots, two colours by direction (Reviewer 3)
vols = {'FR5a_volcano_initial.csv', 'b', 'Initial Partition (Vendor NPX)', 'Fig3_b_volcano_initial';
        'FR5b_volcano_primary.csv', 'c', 'Corrected Partition (Shift-Adjusted)', 'Fig3_c_volcano_corrected';
        'FR5c_volcano_median_centred.csv', 'd', 'Corrected Partition (Median-Centred)', 'Fig3_d_volcano_median'};
for k = 1:size(vols, 1)
    V = rd(vols{k, 1});
    [f, ax] = hivAxes(S, vols{k, 2}, vols{k, 3}, 'Difference in NPX (Higher - Lower Group)', ...
                      '-log_{10} FDR');
    yy = -log10(max(V.q, 1e-300));
    ns = V.direction == "ns"; hi = V.direction == "higher"; lo = V.direction == "lower";
    scatter(ax, V.diff_high_minus_low(ns), yy(ns), 18, S.grey, 'filled', 'MarkerFaceAlpha', 0.5);
    h1 = scatter(ax, V.diff_high_minus_low(hi), yy(hi), 26, S.up, 'filled', 'MarkerFaceAlpha', 0.8);
    h2 = scatter(ax, V.diff_high_minus_low(lo), yy(lo), 26, S.down, 'filled', 'MarkerFaceAlpha', 0.8);
    yline(ax, -log10(0.05), '--k', 'LineWidth', 1.5); xline(ax, 0, ':', 'Color', S.grey, 'LineWidth', 1.5);
    legend([h1 h2], {sprintf('Higher in Higher-Risk Group (n = %d)', sum(hi)), ...
                     sprintf('Lower in Higher-Risk Group (n = %d)', sum(lo))}, ...
           'Location', 'northwest', 'Box', 'off', 'FontSize', S.legend);
    hivExport(f, vols{k, 4}, O, S); close(f);
end

% (E) layer-agreement ARI
A = rd('FR5e_layer_ARI.csv');
[f, ax] = hivAxes(S, 'e', 'Which Layer Defines the Partition', '', 'Adjusted Rand Index');
b = bar(ax, [A.ARI_initial A.ARI_primary], 'grouped', 'LineWidth', 1.5);
b(1).FaceColor = S.c(8, :); b(2).FaceColor = S.c(1, :);
yline(ax, 0.7, '--', 'Dominance Threshold', 'LineWidth', 1.5, 'FontSize', S.annot, 'LabelHorizontalAlignment', 'left');
yline(ax, 0.2, ':', 'Survival Threshold', 'LineWidth', 1.5, 'FontSize', S.annot, 'LabelHorizontalAlignment', 'left');
set(ax, 'XTick', [1 2], 'XTickLabel', {'Proteome Only', 'Without Proteome'}, ...
    'XTickLabelRotation', 0, 'XLim', [0.4 2.6], 'YLim', [-0.1 1]);
xlabel(ax, 'Layers Clustered', 'FontName', S.font, 'FontWeight', S.weight);
legend(b, {'Initial', 'Corrected'}, 'Location', 'northeast', 'Box', 'off', 'FontSize', S.legend);
hivExport(f, 'Fig3_e_layer_ARI', O, S); close(f);
end

% =====================================================================
function fig4(S, rd, O)
M = rd('FR1a_sample_shift_by_plate.csv');
% (A) shift by sample plate
[f, ax] = hivAxes(S, 'a', 'Global Intensity Score by Sample Plate', '', 'Per-Sample Median NPX');
hivPoints(ax, 1, M.shift(M.sample_plate == 1), S.c(1, :), S);
hivPoints(ax, 2, M.shift(M.sample_plate == 2), S.c(3, :), S);
set(ax, 'XTick', [1 2], 'XTickLabel', {sprintf('Plate 1 (n=%d)', sum(M.sample_plate == 1)), ...
    sprintf('Plate 2 (n=%d)', sum(M.sample_plate == 2))}, 'XLim', [0.4 2.6]);
hivExport(f, 'Fig4_a_shift_by_plate', O, S); close(f);

% (B) concordance of the per-sample median across the eight Explore blocks
Bc = readtable(fullfile(fileparts(fileparts(O)), 'revision', 'figure_data', 'FR1b_block_concordance.csv'), ...
               'VariableNamingRule', 'preserve', 'ReadRowNames', true);
[f, ax] = hivAxes(S, 'b', 'Concordance Across Explore Blocks', '', '');
f.Position(3:4) = [26 22];
imagesc(ax, Bc.Variables); axis(ax, 'square'); colormap(ax, parula); caxis(ax, [0 1]);
cb = colorbar(ax); cb.Label.String = 'Pearson r'; cb.FontSize = S.tick;
lab = strrep(Bc.Properties.VariableNames, '_', ' ');
set(ax, 'XTick', 1:numel(lab), 'XTickLabel', lab, 'YTick', 1:numel(lab), 'YTickLabel', lab, ...
    'XTickLabelRotation', 45, 'YDir', 'reverse', 'FontSize', S.tick - 4);
hivExport(f, 'Fig4_b_block_concordance', O, S); close(f);

% (C) loading of each protein on the score by subcellular location
L = rd('FR1c_loading_by_location.csv');
locs = ["intracellular", "membrane", "secreted"];
[f, ax] = hivAxes(S, 'c', 'Protein Loading by Subcellular Location', '', 'Spearman \rho With Score');
yline(ax, 0, '--', 'Color', S.grey, 'LineWidth', 2);
for i = 1:3
    hivPoints(ax, i, L.rho_with_shift(L.location == locs(i)), S.c(i, :), S, 0.3);
end
set(ax, 'XTick', 1:3, 'XTickLabel', {sprintf('Intracellular (n=%d)', sum(L.location == locs(1))), ...
    sprintf('Membrane (n=%d)', sum(L.location == locs(2))), sprintf('Secreted (n=%d)', sum(L.location == locs(3)))}, ...
    'XTickLabelRotation', 20, 'XLim', [0.4 3.6]);
hivExport(f, 'Fig4_c_loading_by_location', O, S); close(f);

% (D) score versus fraction of assays above LOD; (E) versus platelet score
pairs = {'frac_above_LOD', 'd', 'Fraction of Assays Above LOD', 'Fig4_d_shift_vs_LOD';
         'platelet_score', 'e', 'Platelet-Release Score (Mean z)', 'Fig4_e_shift_vs_platelet'};
for k = 1:2
    x = M.(pairs{k, 1});
    [f, ax] = hivAxes(S, pairs{k, 2}, '', pairs{k, 3}, 'Per-Sample Median NPX');
    scatter(ax, x, M.shift, S.ms^2, S.c(1, :), 'filled', 'MarkerEdgeColor', 'w');
    p = polyfit(x, M.shift, 1); xx = linspace(min(x), max(x), 50);
    plot(ax, xx, polyval(p, xx), '-', 'Color', S.c(2, :), 'LineWidth', S.lw);
    rho = corr(x, M.shift, 'Type', 'Spearman');
    title(ax, sprintf('(%s) Spearman \\rho = %.2f (n = %d)', upper(pairs{k, 2}), rho, numel(x)), ...
          'FontSize', S.title);
    hivExport(f, pairs{k, 4}, O, S); close(f);
end
end

% =====================================================================
function fig5(S, rd, rt, O)
% (A) flag rates at n = 89 across scenarios
F = rd('FR3a_flag_rates.csv');
F = F(F.n == 89, :);
lab = F.scenario + " " + string(F.param);
lab = strrep(strrep(strrep(lab, " NaN", ""), " None", ""), "_", " ");
Mx = [F.f1 F.f2 F.f3 F.f4 F.all4 F.ari_truth];
[f, ax] = hivAxes(S, 'a', 'Diagnostic Flag Rates in Simulation (n = 89)', '', '');
f.Position(3:4) = [32 26];
imagesc(ax, Mx); colormap(ax, flipud(gray)); caxis(ax, [0 1]);
cb = colorbar(ax); cb.Label.String = 'Rate / ARI'; cb.FontSize = S.tick;
for i = 1:size(Mx, 1)
    for j = 1:size(Mx, 2)
        text(ax, j, i, sprintf('%.2f', Mx(i, j)), 'HorizontalAlignment', 'center', ...
             'Color', [1 1 1] * (Mx(i, j) < 0.5), 'FontSize', S.annot - 4, 'FontWeight', 'bold');
    end
end
set(ax, 'XTick', 1:6, 'XTickLabel', {'Global Shift', 'Imbalance', 'Survival', 'Dominance', 'All Four', 'ARI Truth'}, ...
    'XTickLabelRotation', 35, 'YTick', 1:numel(lab), 'YTickLabel', lab, 'YDir', 'reverse', 'FontSize', S.tick - 4);
hivExport(f, 'Fig5_a_simulation_flag_rates', O, S); close(f);

% (B) threshold sweeps, technical (1x) vs true multilayer biology vs proteome-wide biology
W = rd('FR3b_threshold_sweep.csv');
diags = ["global_shift_p", "imbalance", "survival_ARI_below", "dominance_ARI_above"];
nice = {'Global Shift (p Threshold)', 'Imbalance (Fraction)', 'Survival (ARI Below)', 'Dominance (ARI Above)'};
scen = ["technical|(1, 0.0)", "bio_multilayer|None", "bio_global|0.9", "null|None"];
sl = {'Technical (1\times)', 'Shared Biology', 'Proteome-Wide Biology', 'No Structure'};
for k = 1:4
    [f, ax] = hivAxes(S, char('b' + k - 1), nice{k}, 'Threshold', 'Flag Rate');
    h = gobjects(4, 1);
    for s = 1:4
        m = W(W.diagnostic == diags(k) & W.scenario == scen(s), :);
        x = m.threshold; if k == 1, x = -log10(x); end
        h(s) = plot(ax, x, m.rate, '-o', 'Color', S.c(s, :), 'LineWidth', S.lw, 'MarkerFaceColor', S.c(s, :), 'MarkerSize', S.ms);
    end
    if k == 1, xlabel(ax, '-log_{10} p Threshold'); end
    ylim(ax, [-0.03 1.03]);
    legend(h, sl, 'Location', 'best', 'Box', 'off', 'FontSize', S.legend - 2);
    hivExport(f, sprintf('Fig5_%s_threshold_%s', char('b' + k - 1), diags(k)), O, S); close(f);
end

% (F) operating characteristics of the flag-count rules
OC = rt('R04_operating_characteristics.csv');
[f, ax] = hivAxes(S, 'f', 'Operating Characteristics', 'Flags Required', 'Rate');
rules = unique(OC.rule, 'stable'); ns = unique(OC.n);
h = gobjects(0);
for i = 1:numel(ns)
    m = OC(OC.n == ns(i), :);
    h(end + 1) = plot(ax, 1:numel(rules), m.sensitivity, '-o', 'Color', S.c(i, :), 'LineWidth', S.lw, 'MarkerFaceColor', S.c(i, :), 'MarkerSize', S.ms); %#ok<AGROW>
    plot(ax, 1:numel(rules), m.false_flag_rate_when_biology_recovered, '--s', 'Color', S.c(i, :), 'LineWidth', S.lw, 'MarkerSize', S.ms);
end
set(ax, 'XTick', 1:numel(rules), 'XTickLabel', {'\geq2', '\geq3', '4'}, 'YLim', [-0.03 1.03], 'XLim', [0.7 3.3]);
legend(h, compose('n = %d', ns), 'Location', 'west', 'Box', 'off', 'FontSize', S.legend);
text(ax, 1.05, 0.08, 'Solid: Sensitivity   Dashed: False Flags With Real Biology', 'FontSize', S.annot - 4);
hivExport(f, 'Fig5_f_operating_characteristics', O, S); close(f);
end

% =====================================================================
function fig6(S, rd, O)
A = rd('FR4_normalization_arms.csv');
lab = strrep(A.arm, '_', ' ');
n = height(A);
% (A) flags per arm
[f, ax] = hivAxes(S, 'a', 'Diagnostic Flags by Normalization', '', 'Flags Raised (of 4)');
b = bar(ax, A.n_flags, 'FaceColor', S.c(2, :), 'LineWidth', 1.5);
set(ax, 'XTick', 1:n, 'XTickLabel', lab, 'XTickLabelRotation', 40, 'YLim', [0 4.3]);
hivExport(f, 'Fig6_a_flags_by_arm', O, S); close(f);
% (B) odds ratio for METS with 95% CI
[f, ax] = hivAxes(S, 'b', 'Metabolic Syndrome by Partition', '', 'Odds Ratio (95% CI, Log Scale)');
yline(ax, 1, '--', 'Color', S.grey, 'LineWidth', 2);
for i = 1:n
    plot(ax, [i i], [A.OR_lo(i) A.OR_hi(i)], '-', 'Color', S.c(1, :), 'LineWidth', 3);
    plot(ax, i, A.OR_METS(i), 'o', 'MarkerFaceColor', S.c(1, :), 'MarkerEdgeColor', 'k', 'MarkerSize', S.ms + 2);
end
set(ax, 'YScale', 'log', 'XTick', 1:n, 'XTickLabel', lab, 'XTickLabelRotation', 40, 'XLim', [0.4 n + 0.6]);
hivExport(f, 'Fig6_b_METS_OR_by_arm', O, S); close(f);
% (C) agreement with the primary arm
[f, ax] = hivAxes(S, 'c', 'Agreement With the Pre-Specified Arm', '', 'Agreement');
b = bar(ax, [A.ARI_vs_primary A.Jaccard_METS_proteins_vs_primary A.rho_METS_effects_vs_primary], 'grouped');
for k = 1:3, b(k).FaceColor = S.c(k, :); end
set(ax, 'XTick', 1:n, 'XTickLabel', lab, 'XTickLabelRotation', 40, 'YLim', [-0.1 1.05]);
legend(b, {'Partition ARI', 'Jaccard (METS Proteins)', '\rho (METS Effects)'}, 'Location', 'southoutside', ...
       'Orientation', 'horizontal', 'Box', 'off', 'FontSize', S.legend - 2);
hivExport(f, 'Fig6_c_agreement_with_primary', O, S); close(f);
end

% =====================================================================
function fig7(S, rd, rt, O)
E = rd('FR6c_axis_and_embedding.csv');
% (A) embedding
[f, ax] = hivAxes(S, 'a', 'Corrected Integrated Representation', 'PC1', 'PC2');
mk = {'o', '^'};
for g = 0:1
    for m = 0:1
        idx = E.group == g & E.METS == m;
        scatter(ax, E.PC1(idx), E.PC2(idx), S.ms^2 * 1.3, S.c(2 - g, :), mk{m + 1}, 'filled', 'MarkerEdgeColor', 'k');
    end
end
legend({'Lower, No MetS', 'Lower, MetS', 'Higher, No MetS', 'Higher, MetS'}, 'Location', 'best', 'Box', 'off', 'FontSize', S.legend - 2);
hivExport(f, 'Fig7_a_embedding', O, S); close(f);
% (B) stability versus Gaussian null
St = rd('FR6a_stability_subsampling.csv');
Nl = rt('R05_stability_null.csv'); Pw = rt('R05_stability_pairwise.csv');
[f, ax] = hivAxes(S, 'b', 'Subsampling Stability (80%, Full Refit)', '', 'Adjusted Rand Index');
hivPoints(ax, 1, Pw.pairwise, S.c(1, :), S, 0.25);
hivPoints(ax, 2, Nl.null_pairwise, S.grey, S, 0.25);
set(ax, 'XTick', [1 2], 'XTickLabel', {sprintf('Observed (%d pairs)', height(Pw)), ...
    sprintf('Gaussian Null (%d)', height(Nl))}, 'XLim', [0.4 2.6], 'YLim', [-0.1 1.05]);
hivExport(f, 'Fig7_b_stability', O, S); close(f);
% (C) split-half replication
H = rd('FR6b_split_half.csv');
[f, ax] = hivAxes(S, 'c', 'Split-Half Replication', 'Odds Ratio for MetS in Held-Out Half (Log)', 'Count');
histogram(ax, log(H.OR_METS_in_B(isfinite(H.OR_METS_in_B) & H.OR_METS_in_B > 0)), 25, ...
          'FaceColor', S.c(1, :), 'EdgeColor', 'w');
xline(ax, 0, '--k', 'LineWidth', 2);
xt = [0.25 0.5 1 2 4 8]; set(ax, 'XTick', log(xt), 'XTickLabel', string(xt));
hivExport(f, 'Fig7_c_split_half', O, S); close(f);
% (D) clinical forest plot (Hedges g for continuous; log OR for binary)
Cc = rd('FR6d_clinical_effects.csv');
Cc = Cc(Cc.type ~= "categorical", :);
est = Cc.estimate; lo = Cc.ci_low; hi = Cc.ci_high;
isb = Cc.type == "binary"; est(isb) = log(est(isb)); lo(isb) = log(lo(isb)); hi(isb) = log(hi(isb));
[f, ax] = hivAxes(S, 'd', 'Clinical Differences (Higher - Lower Group)', 'Hedges g  |  log Odds Ratio', '');
f.Position(3:4) = [28 28];
xline(ax, 0, '--', 'Color', S.grey, 'LineWidth', 2);
k = height(Cc);
mc = Cc.METS_component;
if isstring(mc) || iscellstr(mc), mc = strcmpi(string(mc), "true"); end
for i = 1:k
    col = S.c(1 + double(mc(i)), :);
    plot(ax, [lo(i) hi(i)], [i i], '-', 'Color', col, 'LineWidth', 3);
    plot(ax, est(i), i, 's', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', S.ms);
    text(ax, max(hi) + 0.15, i, sprintf('q = %.2f', Cc.q_BH_all_clinical(i)), 'FontSize', S.annot - 6);
end
set(ax, 'YTick', 1:k, 'YTickLabel', Cc.variable + " (" + string(Cc.n_high) + "/" + string(Cc.n_low) + ")", ...
    'YDir', 'reverse', 'YLim', [0.4 k + 0.6], 'FontSize', S.tick - 4);
hivExport(f, 'Fig7_d_clinical_forest', O, S); close(f);
% (E) heat map, top 40 proteins of the corrected partition
Hm = readtable(fullfile(fileparts(O), 'figure_data', 'FR6e_heatmap_top40_primary.csv'), 'VariableNamingRule', 'preserve');
g = Hm.group; X = Hm{:, 3:end}';
[f, ax] = hivAxes(S, 'e', 'Top 40 Proteins, Corrected Partition (Descriptive)', 'Participants', '');
f.Position(3:4) = [32 28];
imagesc(ax, max(min(X, 3), -3)); colormap(ax, redblue_local(256)); caxis(ax, [-3 3]);
cb = colorbar(ax); cb.Label.String = 'z-Score';
xline(ax, sum(g == 1) + 0.5, 'k-', 'LineWidth', 3);
set(ax, 'YTick', 1:size(X, 1), 'YTickLabel', Hm.Properties.VariableNames(3:end), 'XTick', [], 'FontSize', S.tick - 10, 'YDir', 'reverse');
hivExport(f, 'Fig7_e_heatmap_top40', O, S); close(f);
end

% =====================================================================
function fig8(S, rd, O)
R = rd('FR7_key_term_robustness.csv');
R = R(R.analysis == "fgsea_METS", :);
terms = unique(R.term, 'stable'); arms = unique(R.arm, 'stable');
Q = nan(numel(terms), numel(arms));
for i = 1:numel(terms)
    for j = 1:numel(arms)
        m = R(R.term == terms(i) & R.arm == arms(j), :);
        if ~isempty(m), Q(i, j) = -log10(m.q(1)); end
    end
end
[f, ax] = hivAxes(S, 'a', 'Robustness of Reported Pathways Across Normalizations', '', '');
f.Position(3:4) = [34 22];
imagesc(ax, Q, 'AlphaData', ~isnan(Q)); colormap(ax, parula); cb = colorbar(ax); cb.Label.String = '-log_{10} q';
tl = regexprep(terms, '^(KEGG|REACTOME)_', ''); tl = strrep(lower(tl), '_', ' ');
set(ax, 'YTick', 1:numel(terms), 'YTickLabel', tl, 'XTick', 1:numel(arms), 'XTickLabel', strrep(arms, '_', ' '), ...
    'XTickLabelRotation', 40, 'FontSize', S.tick - 8, 'YDir', 'reverse');
hivExport(f, 'Fig8_a_pathway_robustness', O, S); close(f);
end

% =====================================================================
function fig9(S, rd, O)
M = rd('FR8_missing_modality.csv');
meth = unique(M.method, 'stable');
[f, ax] = hivAxes(S, '', 'Missing-Modality Benchmark (Nested)', 'Participants With One Layer Removed (%)', 'Cross-Validated AUROC');
yline(ax, 0.5, '--', 'Color', S.grey, 'LineWidth', 2);
h = gobjects(numel(meth), 1);
for i = 1:numel(meth)
    m = M(M.method == meth(i), :);
    x = 100 * m.missing_frac;
    fill(ax, [x; flipud(x)], [m.p2_5; flipud(m.p97_5)], S.c(i, :), 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    h(i) = plot(ax, x, m.mean, '-o', 'Color', S.c(i, :), 'LineWidth', S.lw, 'MarkerFaceColor', S.c(i, :), 'MarkerSize', S.ms);
end
legend(h, meth, 'Location', 'southwest', 'Box', 'off', 'FontSize', S.legend - 2);
set(ax, 'XLim', [-2 52], 'YLim', [0.4 1]);
hivExport(f, 'Fig9_missing_modality', O, S); close(f);
end

function figS1(S, rd, O)
M = rd('FRS1_mapping_rates.csv');
[f, ax] = hivAxes(S, '', 'RNA-Seq Mapping Rates', 'Selective-Alignment Mapping Rate (%)', 'Libraries (n)');
histogram(ax, M.mapping_rate, 'BinWidth', 0.5, 'FaceColor', S.c(1, :), 'EdgeColor', 'w', 'LineWidth', 1.5);
xline(ax, median(M.mapping_rate), '--', 'Color', S.grey, 'LineWidth', 2);
text(ax, median(M.mapping_rate) + 0.1, max(ylim(ax)) * 0.92, sprintf('Median %.1f%%', median(M.mapping_rate)), ...
     'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', S.legend);
hivExport(f, 'FigS1_mapping_rates', O, S); close(f);
end

function fig1(S, rd, O)
% (A) study design and identifier bridge (schematic)
f = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 44 20], 'PaperPositionMode', 'auto');
ax = axes(f, 'Position', [0 0 1 1]); axis(ax, [0 44 0 20]); axis(ax, 'off'); hold(ax, 'on');
cols = {'i. Public Data Sources', 'ii. Identifier Bridge', 'iii. Matched Cohort', 'iv. Analysis Flow'};
x0 = [0.5 11.5 22.5 33.0]; w = [10 10 9.5 10.5];
fc = [0.87 0.92 0.98; 0.88 0.96 0.88; 0.99 0.91 0.84; 1.00 0.96 0.83];
for k = 1:4
    rectangle(ax, 'Position', [x0(k) 0.5 w(k) 18.5], 'Curvature', 0.05, 'FaceColor', fc(k, :), ...
              'EdgeColor', fc(k, :) * 0.6, 'LineWidth', 2);
    text(ax, x0(k) + w(k) / 2, 18.2, cols{k}, 'HorizontalAlignment', 'center', 'FontSize', S.axis - 4, ...
         'FontWeight', 'bold', 'FontName', S.font);
end
box = @(x, y, ww, hh, t1, t2) drawBox(ax, S, x, y, ww, hh, t1, t2);
src = {'RNA Sequencing', 'NCBI SRA PRJNA983231'; 'Olink Explore 3072', 'Vendor NPX Export + Plate Layout'; ...
       'Metabolomics', 'COCOMO Public Repository'; 'Clinical Data', 'COCOMO Public Repository'};
for k = 1:4, box(1.0, 14.0 - (k - 1) * 3.6, 9, 2.8, src{k, 1}, src{k, 2}); end
br = {'SRR Run Accession', 'SRA Run Table'; 'P20109\_nnn Library Code', 'LibraryName Field'; ...
      'User Field Minus Run Suffix', 'Clinical Table'; 'COCOMO\_ID', 'Proteome, Metabolome Index'};
for k = 1:4
    box(12.0, 14.0 - (k - 1) * 3.6, 9, 2.8, br{k, 1}, br{k, 2});
    if k < 4, quiver(ax, 16.5, 14.0 - (k - 1) * 3.6, 0, -0.75, 0, 'Color', [0.2 0.5 0.2], 'LineWidth', 2.5, 'MaxHeadSize', 2); end
end
lay = {'Transcriptomics', '3,000 Genes'; 'Olink Proteomics', '2,923 Assays'; 'Metabolomics', '877 Analytes'};
for k = 1:3, box(23.0, 14.0 - (k - 1) * 3.6, 8.5, 2.8, lay{k, 1}, lay{k, 2}); end
rectangle(ax, 'Position', [23.0 1.6 8.5 3.6], 'Curvature', 0.25, 'FaceColor', [1 0.85 0.7], 'EdgeColor', [0.6 0.3 0.1], 'LineWidth', 2);
text(ax, 27.25, 4.2, 'Complete Four-Layer Cohort', 'HorizontalAlignment', 'center', 'FontSize', S.legend - 5, 'FontWeight', 'bold', 'FontName', S.font);
text(ax, 27.25, 2.6, 'n = 89', 'HorizontalAlignment', 'center', 'FontSize', S.axis, 'FontWeight', 'bold', 'FontName', S.font);
steps = {'1. Nested Benchmark', '2. Partition and Layer Checks', '3. Origin of Global Component', ...
         '4. Simulation of Checks', '5. Normalization Sensitivity', '6. Cluster Existence and Axis', ...
         '7. Protein and Pathway Associations'};
for k = 1:7
    y = 15.6 - (k - 1) * 2.25;
    rectangle(ax, 'Position', [33.5 y 9.5 1.7], 'Curvature', 0.3, 'FaceColor', 'w', 'EdgeColor', [0.55 0.45 0.1], 'LineWidth', 1.5);
    text(ax, 38.25, y + 0.85, steps{k}, 'HorizontalAlignment', 'center', 'FontSize', S.legend - 6, 'FontWeight', 'bold', 'FontName', S.font);
end
text(ax, 0.6, 19.6, '(A) Study Design and Identifier Reconstruction', 'FontSize', S.title - 6, 'FontWeight', 'bold', 'FontName', S.font);
hivExport(f, 'Fig1_a_study_design', O, S); close(f);

% (B) cohort assembly
C0 = rd('FR0_cohort_flow.csv');
[f, ax] = hivAxes(S, 'b', 'Cohort Assembly', '', 'Number of Runs or Participants');
b = bar(ax, C0.n, 0.6, 'FaceColor', S.c(1, :), 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 1:height(C0), C0.n + 2, string(C0.n), 'HorizontalAlignment', 'center', 'FontSize', S.legend, ...
     'FontWeight', 'bold', 'FontName', S.font);
set(ax, 'XTick', 1:height(C0), 'XTickLabel', C0.stage, 'XTickLabelRotation', 25, 'YLim', [0 110]);
hivExport(f, 'Fig1_b_cohort_assembly', O, S); close(f);
end

function drawBox(ax, S, x, y, ww, hh, t1, t2)
rectangle(ax, 'Position', [x y ww hh], 'Curvature', 0.25, 'FaceColor', 'w', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5);
text(ax, x + ww / 2, y + hh * 0.66, t1, 'HorizontalAlignment', 'center', 'FontSize', S.legend - 4, ...
     'FontWeight', 'bold', 'FontName', S.font);
text(ax, x + ww / 2, y + hh * 0.27, t2, 'HorizontalAlignment', 'center', 'FontSize', S.legend - 7, ...
     'FontWeight', 'bold', 'FontName', S.font);
end

function c = redblue_local(n)
h = floor(n / 2);
c = [linspace(0, 1, h)' linspace(0.45, 1, h)' linspace(0.70, 1, h)'; ...
     linspace(1, 0.84, n - h)' linspace(1, 0.37, n - h)' linspace(1, 0, n - h)'];
end
