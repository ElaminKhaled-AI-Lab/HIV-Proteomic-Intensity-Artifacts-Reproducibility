function hivExport(fig, name, outdir, S)
%HIVEXPORT  Write <name>.png (raster, S.dpi), .pdf (vector) and .fig.
if ~exist(outdir, 'dir'), mkdir(outdir); end
stem = fullfile(outdir, name);
% enforce font on every text object (legends, colorbars, text annotations)
set(findall(fig, '-property', 'FontName'), 'FontName', S.font);
set(findall(fig, '-property', 'FontWeight'), 'FontWeight', S.weight);
drawnow;
exportgraphics(fig, [stem '.png'], 'Resolution', S.dpi, 'ContentType', 'image');
exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector');
savefig(fig, [stem '.fig']);
% compliance note: effective size after reduction to a 183 mm column
w_mm = S.size_cm(1) * 10;
eff = S.axis * 183 / w_mm;
if eff > 7
    fprintf('[note] %s: axis titles ~%.1f pt at 183 mm width (MDPI-friendly; above Nature 5-7 pt).\n', name, eff);
end
fprintf('exported %s (.png %d dpi, .pdf vector, .fig)\n', stem, S.dpi);
end
