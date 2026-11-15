%% 密封面摩擦力矩参数化模型——中文版运行示例
% 将本文件与 seal_friction_model.m 放在同一文件夹，再点击“运行”。
% MATLAB R2020a及以上；不需要额外工具箱；请使用UTF-8编码保存。
% 本示例仍保留原模型的补全解释，不代表已完成原论文的唯一复原。
clear;clc;close all;
script_dir=fileparts(mfilename('fullpath'));
if isempty(script_dir),script_dir=pwd;end
addpath(script_dir);

%% 一、修改蝶阀参数：长度单位毫米，角度单位度
par.A_mm=113.49;        % 中性面椭圆长半轴，不是直径
par.B_mm=113.05;        % 中性面椭圆短半轴
par.R0_mm=117.5;        % 密封圆锥参考底面半径，不等于长短半轴
par.theta_half_deg=10;  % 圆锥半角；若给的是顶角，需先除以2
par.tilt_deg=5;         % 圆锥角偏心，不是蝶板开启角度
par.b_mm=15;            % 蝶板轴向厚度
par.c_mm=30;            % 轴向偏心距
par.e_mm=15;            % 径向偏心距，带符号
par.mu=0.05;            % 仅为演示假设，请替换为有依据的等效摩擦系数

% 法向线载荷投影解释：保留英文选项，以兼容核心函数。
% 'local'：采用局部方向角；线载荷随位置变化。
% 'uniform_tilt'：采用角偏心；线载荷为常量。
% 这两个选项用于明确论文的符号歧义，不代表已验证的两种接触模型。
par.normal_projection='local';
p_MPa=linspace(0,1,101); % 介质压力范围，单位MPa；这里不是开度数组
warning('示例摩擦系数mu=0.05为假设值，请按材料、涂层、润滑及试验依据修改。');

%% 二、数值积分：计算四项分量与开启、关闭摩擦力矩
out=seal_friction_model(par,p_MPa);
fprintf('\n===== 中间几何与积分参数 =====\n');
fprintf('椭圆半焦距 C = %.6f 毫米\n',out.geometry.C_mm);
fprintf('锥顶至中性面的距离参数 h = %.6f 毫米\n',out.geometry.h_mm);
fprintf('积分分割角 theta1 = %.6f 度\n',out.geometry.theta1_deg);
fprintf('上端局部方向角 alpha1 = %.6f 度\n',out.geometry.alpha1_deg);
fprintf('积分量 k1 = %.9f 米\n',out.geometry.k1_m);
fprintf('积分量 k2 = %.9f 米\n',out.geometry.k2_m);
fprintf('法向线载荷分母 = %.9f 米\n',out.geometry.normal_load_denominator_m);
fprintf('\n===== 力矩随压力变化的斜率 =====\n');
fprintf('开启摩擦力矩斜率：%.6f N·m/MPa\n',out.slopes_Nm_per_MPa(1));
fprintf('关闭摩擦力矩斜率：%.6f N·m/MPa\n',out.slopes_Nm_per_MPa(2));

%% 三、查询一个指定压力下的结果
query_p_MPa=0.45; % 可自行修改
query_out=seal_friction_model(par,query_p_MPa);
% 核心函数保留英文列名；显示和导出时替换为中文表头。
cn_names={'压力_MPa','分量M1_Nm','分量M2_Nm','分量M3_Nm', ...
    '分量M4_Nm','开启摩擦力矩_Nm','关闭摩擦力矩_Nm'};
query_table=query_out.table;
query_table.Properties.VariableNames=cn_names;
fprintf('\n===== 指定压力下的四项分量与合成力矩 =====\n');
disp(query_table);

%% 四、绘图：横轴是压力，不是蝶板开度
if strcmp(par.normal_projection,'local')
    mode_cn='局部方向角解释';
else
    mode_cn='均匀线载荷解释';
end
fig=figure('Color','w','Position',[100 100 900 600]);
plot(out.table.p_MPa,out.table.opening_Nm,'r-','LineWidth',1.8);hold on;
plot(out.table.p_MPa,out.table.closing_Nm,'g-.','LineWidth',1.8);
xlabel('介质压力 p / MPa');
ylabel('密封面摩擦力矩 / (N·m)');
title({'密封面摩擦力矩参数化积分模型', ...
    sprintf('摩擦系数μ=%.3f（示例假设）；%s',par.mu,mode_cn)},'Interpreter','none');
legend('开启摩擦力矩 M_{f1}','关闭摩擦力矩 M_{f2}','Location','northwest');
grid on;box on;set(gca,'FontSize',12);

%% 五、导出中文表头数据及图片
outdir=fullfile(script_dir,'model_results_cn');
if ~exist(outdir,'dir'),mkdir(outdir);end
result_table=out.table;
result_table.Properties.VariableNames=cn_names;
writetable(result_table,fullfile(outdir,'摩擦力矩各分量.csv'),'Encoding','UTF-8');
save(fullfile(outdir,'摩擦模型结果.mat'),'par','out','result_table','query_table');
exportgraphics(fig,fullfile(outdir,'摩擦力矩曲线.png'),'Resolution',300);
exportgraphics(fig,fullfile(outdir,'摩擦力矩曲线.pdf'),'ContentType','vector');
savefig(fig,fullfile(outdir,'摩擦力矩曲线.fig'));
fprintf('\n计算结果已保存至：%s\n',outdir);
% 说明：不包含预紧、夹砂、接触变形、轴承、填料、静水和动水力矩。
