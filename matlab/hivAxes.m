function [fig, ax] = hivAxes(S, letter, ttl, xl, yl)
%HIVAXES  One styled panel in its own figure (one panel = one file).
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 S.size_cm], ...
             'PaperPositionMode', 'auto', 'Visible', 'on');
ax = axes(fig);
hold(ax, 'on');
set(ax, 'FontName', S.font, 'FontWeight', S.weight, 'FontSize', S.tick, ...
    'LineWidth', 2, 'TickDir', 'out', 'Box', 'off', 'Layer', 'top', ...
    'TitleFontSizeMultiplier', S.title / S.tick, ...
    'LabelFontSizeMultiplier', S.axis / S.tick);
if ~isempty(ttl)
    if ~isempty(letter)
        title(ax, sprintf('(%s) %s', upper(letter), hivTitleCase(ttl)), ...
              'FontName', S.font, 'FontWeight', S.weight);
    else
        title(ax, hivTitleCase(ttl), 'FontName', S.font, 'FontWeight', S.weight);
    end
end
if ~isempty(xl), xlabel(ax, hivTitleCase(xl), 'FontName', S.font, 'FontWeight', S.weight); end
if ~isempty(yl), ylabel(ax, hivTitleCase(yl), 'FontName', S.font, 'FontWeight', S.weight); end
end
