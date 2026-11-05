function hivPoints(ax, x, y, col, S, jitter)
%HIVPOINTS  Jittered univariate scatter with a box (median, IQR) behind it.
if nargin < 6, jitter = 0.18; end
y = y(:); y = y(~isnan(y));
if isempty(y), return; end
q = prctile(y, [25 50 75]);
patch(ax, x + [-0.28 0.28 0.28 -0.28], [q(1) q(1) q(3) q(3)], col, ...
      'FaceAlpha', 0.18, 'EdgeColor', col, 'LineWidth', 2);
plot(ax, x + [-0.28 0.28], [q(2) q(2)], '-', 'Color', col, 'LineWidth', 3.5);
rng(1);
xx = x + (rand(size(y)) - 0.5) * 2 * jitter;
scatter(ax, xx, y, S.ms^2 / 1.6, col, 'filled', 'MarkerFaceAlpha', 0.75, ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
end
