function S = hivStyle(preset)
%HIVSTYLE  House figure style for the HIV proteomic-artifact revision.
%   S = hivStyle('house')  Arial bold; titles 30 pt, axis titles 25 pt,
%   ticks 22 pt; PNG at 1500 dpi + vector PDF + editable FIG.
%   Edit the numbers here once to change every panel.
if nargin < 1, preset = 'house'; end
S.font = 'Arial';
S.weight = 'bold';
switch lower(preset)
    case 'house'
        S.title = 30; S.axis = 25; S.tick = 22; S.legend = 20; S.annot = 20;
        S.size_cm = [22 18];
    case 'nature'
        S.title = 8; S.axis = 7; S.tick = 6; S.legend = 6; S.annot = 6;
        S.size_cm = [8.9 7.5];
    otherwise
        error('unknown preset %s', preset);
end
S.dpi = 1500;
S.lw = 2.5; S.ms = 9;
% Okabe-Ito (colour-vision-deficiency safe); first two maximally distinct
S.c = [0.00 0.45 0.70;   % blue
       0.84 0.37 0.00;   % vermillion
       0.00 0.62 0.45;   % bluish green
       0.80 0.47 0.65;   % reddish purple
       0.90 0.62 0.00;   % orange
       0.34 0.71 0.91;   % sky blue
       0.95 0.90 0.25;   % yellow
       0.40 0.40 0.40];  % grey
S.grey = [0.70 0.70 0.70];
S.up = S.c(2,:);   % higher in higher-risk group / positive
S.down = S.c(1,:); % lower in higher-risk group / negative
end
