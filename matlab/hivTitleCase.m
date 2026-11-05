function out = hivTitleCase(str)
%HIVTITLECASE  Title Case with minor words lower-case; words containing an
%internal capital or digit (gene symbols, units, abbreviations) untouched.
minor = {'a','an','the','and','but','or','nor','for','so','yet','at','by', ...
         'in','of','on','to','up','as','from','into','over','with','via', ...
         'per','vs','than','that'};
w = strsplit(char(str), ' ');
for i = 1:numel(w)
    s = w{i};
    if isempty(s), continue; end
    if any(isstrprop(s(2:end), 'upper')) || any(isstrprop(s, 'digit')), continue; end
    if i > 1 && i < numel(w) && any(strcmpi(s, minor))
        w{i} = lower(s);
    else
        w{i} = [upper(s(1)) s(2:end)];
    end
end
out = strjoin(w, ' ');
end
