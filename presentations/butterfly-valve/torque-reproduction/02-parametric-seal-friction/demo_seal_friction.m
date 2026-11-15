%% Parameterized seal friction model -- edit these parameters, then Run
% Keep seal_friction_model.m in the same folder.
% MATLAB R2020a+ (exportgraphics); no additional toolbox.
clear;clc;close all;
script_dir=fileparts(mfilename('fullpath'));
if isempty(script_dir),script_dir=pwd;end
addpath(script_dir);

%% User inputs: lengths mm, angles degrees
par.A_mm=113.49;       % major SEMIAXIS, not diameter
par.B_mm=113.05;       % minor SEMIAXIS
par.R0_mm=117.5;      % cone reference-base radius
par.theta_half_deg=10; % HALF cone angle (full apex angle/2)
par.tilt_deg=5;       % angular offset of cone
par.b_mm=15;          % disc axial thickness
par.c_mm=30;          % axial shaft offset
par.e_mm=15;          % signed radial shaft offset
par.mu=0.05;          % DEMONSTRATION ASSUMPTION, not verified thesis value!
par.normal_projection='local'; % default literal local-angle interpretation
p_MPa=linspace(0,1,101);
warning('Demo mu=0.05 is assumed. Replace it with a justified material/contact value.');

%% Integrate the model; no published torque endpoint appears here
out=seal_friction_model(par,p_MPa);
disp(out.geometry);
fprintf('Computed opening slope: %.6f N*m/MPa\n',out.slopes_Nm_per_MPa(1));
fprintf('Computed closing slope: %.6f N*m/MPa\n',out.slopes_Nm_per_MPa(2));
q=seal_friction_model(par,0.45);disp(q.table);

%% Plot pressure-dependent friction, not torque versus opening angle
fig=figure('Color','w','Position',[100 100 850 580]);
plot(out.table.p_MPa,out.table.opening_Nm,'r-','LineWidth',1.8);hold on;
plot(out.table.p_MPa,out.table.closing_Nm,'g-.','LineWidth',1.8);
xlabel('Medium pressure p (MPa)');ylabel('Seal friction torque (N\cdotm)');
title({'Parameterized seal friction integral model', ...
    sprintf('mu=%.3f (example assumption); normal projection: %s',par.mu,par.normal_projection)}, ...
    'Interpreter','none');
legend('Opening M_{f1}','Closing M_{f2}','Location','northwest');
grid on;box on;set(gca,'FontSize',12);
outdir=fullfile(script_dir,'model_results');
if ~exist(outdir,'dir'),mkdir(outdir);end
writetable(out.table,fullfile(outdir,'friction_components.csv'));
save(fullfile(outdir,'model_results.mat'),'par','out');
exportgraphics(fig,fullfile(outdir,'friction_model.png'),'Resolution',300);
exportgraphics(fig,fullfile(outdir,'friction_model.pdf'),'ContentType','vector');
savefig(fig,fullfile(outdir,'friction_model.fig'));
fprintf('Saved to %s\n',outdir);
