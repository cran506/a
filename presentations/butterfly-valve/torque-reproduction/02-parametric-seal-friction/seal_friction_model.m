function out = seal_friction_model(par, p_MPa)
%SEAL_FRICTION_MODEL Parameterized interpretation of thesis Eqs. 2.7-2.23.
% Not a fit to published endpoint torques. No toolbox required.
% IMPORTANT: the supplied thesis has inconsistent angle/pole definitions.
% This implementation exposes its closure assumptions; see MODEL_NOTES.md.
%
% Required par fields (lengths in mm, angles in degrees):
% A_mm,B_mm : ellipse major/minor semiaxes (A>=B>0), NOT diameters
% R0_mm    : cone reference-base radius, NOT disc radius
% b_mm     : disc axial thickness
% c_mm,e_mm: axial/radial offsets using thesis sign convention
% theta_half_deg: cone half angle; divide a full apex angle by 2
% tilt_deg : cone angular offset, distinguished from local friction angle
% mu       : equivalent seal friction coefficient, supplied by user
% Optional par.normal_projection: 'local' (default) or 'uniform_tilt'
% 'local': literal Eq.2.11 cos(alpha) uses local angle from Eq.2.20;
%          the resulting N(beta) is NOT uniform.
% 'uniform_tilt': interpret Eq.2.11 cos(alpha) as cos(tilt); N is uniform.
% These are interpretations, NOT two validated physical models.
% p_MPa: nonnegative vector of the paper's medium pressure, in MPa.
%
% All internal lengths use metres; pressure Pa; line load N/m; torques N*m.
% Output closure assumptions are returned in out.assumptions.

fields = {'A_mm','B_mm','R0_mm','b_mm','c_mm','e_mm', ...
          'theta_half_deg','tilt_deg','mu'};
for k = 1:numel(fields)
    assert(isfield(par,fields{k}), 'Missing parameter: %s',fields{k});
    validateattributes(par.(fields{k}), {'numeric'}, ...
        {'real','finite','scalar'}, mfilename,fields{k});
end
validateattributes(p_MPa, {'numeric'}, {'real','finite','vector','nonnegative'});
assert(par.A_mm >= par.B_mm && par.B_mm > 0, 'Require A >= B > 0.');
assert(par.R0_mm > 0 && par.b_mm >= 0 && par.c_mm >= 0, ...
    'Require R0>0, b>=0, c>=0.');
assert(par.mu >= 0, 'mu must be nonnegative.');
assert(par.theta_half_deg > 0 && par.theta_half_deg < 90, ...
    'Cone half angle must lie between 0 and 90 degrees.');
assert(abs(par.e_mm) < par.A_mm, 'Shaft radial offset must lie inside ellipse.');
assert(par.theta_half_deg+par.tilt_deg > 0 && ...
       par.theta_half_deg+par.tilt_deg < 90, ...
    'This branch requires 0 < theta_half + tilt < 90 degrees.');
if ~isfield(par,'normal_projection'), par.normal_projection = 'local'; end
mode = validatestring(par.normal_projection, {'local','uniform_tilt'});

A=par.A_mm*1e-3; B=par.B_mm*1e-3;
R0=par.R0_mm*1e-3; b=par.b_mm*1e-3;
c=par.c_mm*1e-3; e=par.e_mm*1e-3;
theta=par.theta_half_deg*pi/180; gamma=par.tilt_deg*pi/180; mu=par.mu;
C=sqrt(max(0,A^2-B^2));                          % Eq.2.16
h=R0*cos(theta+gamma)/sin(theta)-b/2;           % Eq.2.15
assert(h>0, 'Invalid geometry: h<=0.');

% Closure: polar ellipse referenced to its LOWER FOCUS F1, angle measured
% from +major-axis direction. This supplies rho_beta absent as an explicit
% equation in the excerpt. It does NOT assert F1 equals cone projection O1.
rho=@(beta) B^2 ./ (A-C*cos(beta));
% Split at y=-e in ellipse-centred coordinates: rho*cos(beta)=C-e.
ct=(C-e)*A/(B^2+C*(C-e));
assert(abs(ct)<=1+1e-12, 'No valid split angle theta1.');
theta1=acos(max(-1,min(1,ct)));

% Retain the printed Eq.2.17 and Eq.2.18 (including R0*sin(beta)).
% Signed intermediate projection is squared in Eq.2.18.
K=@(beta) rho(beta).*cos(beta)-A-C+h*tan(theta+gamma);
L=@(beta) hypot(K(beta), R0*sin(beta));
cos_chi=@(beta) h ./ hypot(h,L(beta));          % Eq.2.20
sin_chi=@(beta) L(beta) ./ hypot(h,L(beta));    % Eq.2.19
% alpha1: local friction-direction angle at T1, beta=0.
ca1=cos_chi(0); sa1=sin_chi(0);

opts={'RelTol',1e-9,'AbsTol',1e-13};
k1=integral(rho,0,pi,opts{:});                 % Eq.2.12
k2=integral(rho,0,theta1,opts{:});              % Eq.2.13
D=2*((sin(theta)-mu*cos(theta))*k1 ...
    +2*mu*cos(theta)*k2);                      % Eq.2.11 denominator
assert(D>0 && D>1e-10*k1, ...
    'Normal-load denominator nonpositive/nearly singular: review parameters.');
if strcmp(mode,'local')
    projection=cos_chi;
else
    projection=@(beta) cos(gamma)+zeros(size(beta));
end
% q_per_Pa has units m: q(beta,p)=q_per_Pa(beta)*p_Pa in N/m.
q_per_Pa=@(beta) pi*A*B.*projection(beta)./D;

I1=integral(@(z) 2*mu.*q_per_Pa(z).*ca1.* ...
    (rho(z).*cos(z)-(C-e)).*rho(z),0,theta1,opts{:});
I2=integral(@(z) 2*mu.*q_per_Pa(z).*ca1.* ...
    ((C-e)-rho(z).*cos(z)).*rho(z),theta1,pi,opts{:});
I3=integral(@(z) 2*mu.*q_per_Pa(z).*sa1.* ...
    cos_chi(z).*c.*rho(z),0,theta1,opts{:});
I4=integral(@(z) 2*mu.*q_per_Pa(z).*sa1.* ...
    cos_chi(z).*c.*rho(z),theta1,pi,opts{:});
coef=[I1,I2,I3,I4];                            % N*m/Pa
p=p_MPa(:); components=(p*1e6)*coef;
opening=components*[1;1;1;-1];                 % Eq.2.22
closing=components*[1;-1;-1;1];                % Eq.2.23
% Keep the signs. Negative results are not silently turned into resistance.
if any(opening < -1e-9) || any(closing < -1e-9)
    warning('seal:modelBranch', ['Negative assembled torque: review geometry, ', ...
        'sign convention and model applicability; no abs() has been applied.']);
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
    'rho uses lower-focus polar ellipse; thesis pole labels conflict', ...
    'alpha1 evaluated at T1 using local-angle equations 2.19-2.20', ...
    ['normal projection interpretation: ', mode], ...
    'Eq.2.18 retains printed R0*sin(beta), not rho*sin(beta)', ...
    'no preload, particles, deformation, hydrodynamic or bearing torque', ...
    'interpretation implementation, not authenticated author source code'};
end
