function out = seal_friction_model(par, p_MPa)
% 密封面摩擦力矩参数化模型——按论文式（2.7）至（2.23）进行解释性实现。
% 本程序通过数值积分计算，不使用论文端点拟合；不需要额外工具箱。
% 原文角度与极点定义存在歧义，补全解释详见“模型说明.md”。
%
% 输入结构体 par 的字段（长度：毫米；角度：度）：
% A_mm、B_mm：中性面椭圆的长、短半轴，必须满足 A>=B>0，不是直径。
% R0_mm：密封圆锥的参考底面半径，不是蝶板半径。
% b_mm：蝶板轴向厚度。
% c_mm、e_mm：轴向、径向偏心距，沿用论文的符号约定。
% theta_half_deg：圆锥半角；若已知圆锥顶角，需除以2后输入。
% tilt_deg：圆锥角偏心，与局部摩擦方向角区分。
% mu：等效密封摩擦系数，需要用户根据材料和接触条件提供。
% normal_projection：法向载荷投影解释，可选：
%   'local'：默认；式（2.11）采用式（2.20）的局部角，线载荷随位置变化。
%   'uniform_tilt'：式（2.11）采用角偏心，得到均匀线载荷。
% 两种设置仅用于明确原文符号歧义，均非已经验证的接触模型。
% p_MPa：非负压力数组，单位MPa，沿用论文的“介质压力”定义。
%
% 内部统一使用：长度m、压力Pa、线载荷N/m、力矩N·m。
% 输出 out.table：压力及四项分量、开启与关闭摩擦力矩。
% 输出 out.geometry：中间几何量；out.assumptions：本实现的解释假设。
% 英文函数名、字段名和MATLAB关键字保留，以兼容原调用接口。

fields = {'A_mm','B_mm','R0_mm','b_mm','c_mm','e_mm', ...
          'theta_half_deg','tilt_deg','mu'};
for k = 1:numel(fields)
    assert(isfield(par,fields{k}), '缺少输入参数：%s',fields{k});
    validateattributes(par.(fields{k}), {'numeric'}, ...
        {'real','finite','scalar'}, mfilename,fields{k});
end
validateattributes(p_MPa, {'numeric'}, {'real','finite','vector','nonnegative'});
assert(par.A_mm >= par.B_mm && par.B_mm > 0, '长短半轴必须满足 A >= B > 0。');
assert(par.R0_mm > 0 && par.b_mm >= 0 && par.c_mm >= 0, ...
    '必须满足 R0>0、b>=0、c>=0。');
assert(par.mu >= 0, '摩擦系数 mu 必须为非负数。');
assert(par.theta_half_deg > 0 && par.theta_half_deg < 90, ...
    '圆锥半角必须大于0度且小于90度。');
assert(abs(par.e_mm) < par.A_mm, '径向偏心的绝对值必须小于椭圆长半轴。');
assert(par.theta_half_deg+par.tilt_deg > 0 && ...
       par.theta_half_deg+par.tilt_deg < 90, ...
    '当前分支要求：圆锥半角与角偏心之和大于0度且小于90度。');
if ~isfield(par,'normal_projection'), par.normal_projection = 'local'; end
mode = validatestring(par.normal_projection, {'local','uniform_tilt'});

A=par.A_mm*1e-3; B=par.B_mm*1e-3;
R0=par.R0_mm*1e-3; b=par.b_mm*1e-3;
c=par.c_mm*1e-3; e=par.e_mm*1e-3;
theta=par.theta_half_deg*pi/180; gamma=par.tilt_deg*pi/180; mu=par.mu;
C=sqrt(max(0,A^2-B^2));                          % 式2.16
h=R0*cos(theta+gamma)/sin(theta)-b/2;           % 式2.15
assert(h>0, '几何参数无效：计算得到 h<=0。');

% 补全解释：采用以下焦点F1为极点的椭圆极坐标关系，角度从长轴正向计量。
% 此处补充了节选中未明确列出的rho_beta表达式。
% 不据此认定椭圆焦点F1与锥顶投影O1重合。
rho=@(beta) B^2 ./ (A-C*cos(beta));
% 在椭圆中心坐标中，以y=-e为分界，即rho*cos(beta)=C-e。
ct=(C-e)*A/(B^2+C*(C-e));
assert(abs(ct)<=1+1e-12, '无法得到有效的积分分割角 theta1，请检查几何与偏心参数。');
theta1=acos(max(-1,min(1,ct)));

% 保留原文式2.17和2.18，包括R0*sin(beta)。
% 中间投影量可以带符号，在式2.18中进行平方。
K=@(beta) rho(beta).*cos(beta)-A-C+h*tan(theta+gamma);
L=@(beta) hypot(K(beta), R0*sin(beta));
cos_chi=@(beta) h ./ hypot(h,L(beta));          % 式2.20
sin_chi=@(beta) L(beta) ./ hypot(h,L(beta));    % 式2.19
% alpha1：上端T1点的局部摩擦方向角，该点取beta=0。
ca1=cos_chi(0); sa1=sin_chi(0);

opts={'RelTol',1e-9,'AbsTol',1e-13};
k1=integral(rho,0,pi,opts{:});                 % 式2.12
k2=integral(rho,0,theta1,opts{:});              % 式2.13
D=2*((sin(theta)-mu*cos(theta))*k1 ...
    +2*mu*cos(theta)*k2);                      % 式2.11 的分母
assert(D>0 && D>1e-10*k1, ...
    '法向线载荷分母非正或接近奇异，请核实参数与模型适用范围。');
if strcmp(mode,'local')
    projection=cos_chi;
else
    projection=@(beta) cos(gamma)+zeros(size(beta));
end
% q_per_Pa的单位是m；乘以压力Pa后，得到单位为N/m的法向线载荷。
q_per_Pa=@(beta) pi*A*B.*projection(beta)./D;

I1=integral(@(z) 2*mu.*q_per_Pa(z).*ca1.* ...
    (rho(z).*cos(z)-(C-e)).*rho(z),0,theta1,opts{:});
I2=integral(@(z) 2*mu.*q_per_Pa(z).*ca1.* ...
    ((C-e)-rho(z).*cos(z)).*rho(z),theta1,pi,opts{:});
I3=integral(@(z) 2*mu.*q_per_Pa(z).*sa1.* ...
    cos_chi(z).*c.*rho(z),0,theta1,opts{:});
I4=integral(@(z) 2*mu.*q_per_Pa(z).*sa1.* ...
    cos_chi(z).*c.*rho(z),theta1,pi,opts{:});
coef=[I1,I2,I3,I4];                            % 单位压力对应的力矩系数，单位N·m/Pa
p=p_MPa(:); components=(p*1e6)*coef;
opening=components*[1;1;1;-1];                 % 式2.22
closing=components*[1;-1;-1;1];                % 式2.23
% 保留力矩符号，不自动将负值改成正的阻力矩。
if any(opening < -1e-9) || any(closing < -1e-9)
    warning('seal:modelBranch', ['合成力矩出现负值，请检查几何、', ...
        '符号约定与模型适用范围；程序没有取绝对值掩盖该结果。']);
end
out.table=table(p,components(:,1),components(:,2),components(:,3), ...
    components(:,4),opening,closing,'VariableNames', ...
    {'p_MPa','M1_Nm','M2_Nm','M3_Nm','M4_Nm','opening_Nm','closing_Nm'});
out.slopes_Nm_per_MPa=1e6*[coef*[1;1;1;-1],coef*[1;-1;-1;1]];
out.geometry=struct('C_mm',C*1e3,'h_mm',h*1e3, ...
    'theta1_deg',theta1*180/pi,'alpha1_deg',atan2(sa1,ca1)*180/pi, ...
    'k1_m',k1,'k2_m',k2,'normal_load_denominator_m',D);
out.parameters=par;
out.assumptions={ ...
    'rho采用以下焦点为极点的椭圆关系；论文中的极点标记存在冲突', ...
    'alpha1按T1点的局部方向角解释，用式2.19至2.20求得', ...
    ['法向线载荷投影解释选项：', mode], ...
    '式2.18保留原文R0*sin(beta)，没有替换为rho*sin(beta)', ...
    '不包含预紧、颗粒、变形、动水及轴承摩擦力矩', ...
    '这是含补全解释的实现，不是已核实的原作者源代码'};
end
