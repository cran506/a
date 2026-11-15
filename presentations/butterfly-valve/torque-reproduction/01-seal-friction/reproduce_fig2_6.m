%% Reproduce the published linear curves in thesis Fig. 2.6
% Method: published endpoints + stated linearity, NOT an independent
% numerical integration of equations (2.7)-(2.23).
% Scope: disc/seat seal friction at the instant of opening/closing.
% p is the paper's medium pressure in MPa; torque is in N*m.
% No CFD, contact FEA, shaft-bearing friction or packing friction included.
% Requires MATLAB R2020a+ for exportgraphics. No additional toolbox required.
clear; clc; close all;

%% Inputs
p_MPa = linspace(0, 1, 101).';
k_open  = 840.92;  % N*m/MPa, published opening torque at 1 MPa
k_close = 174.64;  % N*m/MPa, published closing torque at 1 MPa
p_query_MPa = 0.45;
% These slopes are NOT coefficients of friction.

%% Published-result reconstruction
Mf1_Nm = k_open  .* p_MPa;
Mf2_Nm = k_close .* p_MPa;
query_open_Nm  = k_open  * p_query_MPa;
query_close_Nm = k_close * p_query_MPa;
assert(abs(Mf1_Nm(end) - 840.92) < 1e-9);
assert(abs(Mf2_Nm(end) - 174.64) < 1e-9);
assert(Mf1_Nm(1) == 0 && Mf2_Nm(1) == 0);
fprintf('At p = %.3f MPa:\n', p_query_MPa);
fprintf('  Opening seal friction torque = %.3f N*m\n', query_open_Nm);
fprintf('  Closing seal friction torque = %.3f N*m\n', query_close_Nm);

%% Figure
fig = figure('Color', 'w', 'Position', [100 100 820 580]);
ax = axes(fig); hold(ax, 'on');
plot(ax, p_MPa, Mf1_Nm, 'r-', 'LineWidth', 1.8);
plot(ax, p_MPa, Mf2_Nm, '-.', 'Color', [0.15 0.65 0.2], 'LineWidth', 1.8);
xlim(ax, [0 1.1]); ylim(ax, [0 1000]);
xticks(ax, 0:0.2:1); yticks(ax, 0:200:1000);
xlabel(ax, 'Medium pressure p (MPa)');
ylabel(ax, 'Seal friction torque (N\cdotm)', 'Interpreter', 'tex');
title(ax, {'Fig. 2.6: disc-seat seal friction torque', ...
    'Published-endpoint reconstruction; not independent model validation'});
legend(ax, {'M_{f1}: opening', 'M_{f2}: closing'}, ...
    'Location', 'northwest', 'Interpreter', 'tex');
text(ax, 0.74, 880, '(1, 840.92)', 'FontSize', 11);
text(ax, 0.74, 215, '(1, 174.64)', 'FontSize', 11);
text(ax, 0.03, 40, '(0, 0)', 'FontSize', 11);
plot(ax, 1, k_open, 'ro', 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
plot(ax, 1, k_close, 'o', 'Color', [0.15 0.65 0.2], ...
    'MarkerFaceColor', [0.15 0.65 0.2], 'HandleVisibility', 'off');
set(ax, 'FontSize', 12, 'LineWidth', 1, 'Box', 'on');
grid(ax, 'on'); ax.GridAlpha = 0.15;

%% Save alongside the script
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
out_dir = fullfile(script_dir, 'matlab_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
T = table(p_MPa, Mf1_Nm, Mf2_Nm, 'VariableNames', ...
    {'Pressure_MPa', 'OpeningFriction_Nm', 'ClosingFriction_Nm'});
writetable(T, fullfile(out_dir, 'fig2_6_data.csv'));
exportgraphics(fig, fullfile(out_dir, 'fig2_6.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(out_dir, 'fig2_6.pdf'), 'ContentType', 'vector');
savefig(fig, fullfile(out_dir, 'fig2_6.fig'));
fprintf('Results saved to: %s\n', out_dir);
% Zero-pressure torque is zero here only because the paper idealizes away
% preload and residual friction. Do not generalize this to a real valve.
