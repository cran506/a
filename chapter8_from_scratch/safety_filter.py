"""Exact sampled HOCBF teaching filter for x'=v, v'=k*u-c*v (mm, s, V).

This reduced-order model is an explicit assumption, not the four-state hydraulic
pressure ODE.  A first-order valve lag or the pressure states change relative
order; the present proof must not be transferred to those plants unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from scipy.optimize import brentq
import numpy as np

class SafetyInfeasibleError(RuntimeError):
    """The certified initial set or actuator-constrained QP is infeasible."""

@dataclass(frozen=True)
class FilterDiagnostics:
    proposed_v: float
    applied_v: float
    lower_v: float
    upper_v: float
    physical_barrier: float
    upper_distance_mm: float
    lower_distance_mm: float
    upper_psi1_mm_s: float
    lower_psi1_mm_s: float
    intervened: bool
    model: str = 'second_order_velocity_surrogate'

class HOCBFSafetyFilter:
    """Minimize (u-u_rl)^2/2 subject to a two-sided sampled HOCBF.

    kappa1,kappa2 > 0 produce psi1=h_dot+kappa1*h and
    psi2=psi1_dot+kappa2*psi1 >= 0.  The QP enforces psi2 at
    *every time* inside the zero-order-hold interval, using exact flow.
    Initial h >= 0 and psi1 >= 0 are required for both travel boundaries.
    Infeasible states raise; no command is falsely labelled safe.
    """
    def __init__(self, x_min=0.0, x_max=100.0, u_min=-10.0, u_max=10.0,
                 margin=1.2, dt=0.001, k_gain=1850.0, damp_c=14.0,
                 kappa1=24.0, kappa2=24.0):
        args=[x_min,x_max,u_min,u_max,margin,dt,k_gain,damp_c,kappa1,kappa2]
        if not all(math.isfinite(float(x)) for x in args):
            raise ValueError('All filter settings must be finite')
        if not (x_max>x_min and 0<=2*margin<x_max-x_min and u_max>u_min):
            raise ValueError('Invalid physical limits or margin')
        if min(dt,k_gain,damp_c,kappa1,kappa2)<=0:
            raise ValueError('Time, gain, damping and kappa values must be positive')
        self.x_min=float(x_min);self.x_max=float(x_max)
        self.u_min=float(u_min);self.u_max=float(u_max)
        self.margin=float(margin);self.dt=float(dt)
        self.k_gain=float(k_gain);self.damp_c=float(damp_c)
        self.kappa1=float(kappa1);self.kappa2=float(kappa2)
        self.alpha1=self.kappa1+self.kappa2
        self.alpha2=self.kappa1*self.kappa2
        self.last_info=None

    def compute_barrier_h(self,x_p):
        span=self.x_max-self.x_min
        return ((self.x_max-x_p)*(x_p-self.x_min))/(span*span)

    def _bound(self,h,v,lower=False):
        # psi2(t) = C(t) - D(t)*u for the upper boundary;
        # the lower bound is u >= C_lower(t)/D(t). D(t)>0.
        c=self.damp_c;k=self.k_gain;alpha=self.alpha1;beta=self.alpha2
        d=c-alpha
        D0=1-d/c-beta/(c*c);D1=beta/c;D2=d/c+beta/(c*c)
        C0=-beta*(h+v/c) if lower else beta*(h-v/c)
        C2=(d+beta/c)*v
        E0=-c*C2*D0-C2*D1+c*C0*D2
        E1=-c*C2*D1
        def derivative_numerator(t):
            return math.exp(-c*t)*(E0+E1*t)-C0*D1
        # The derivative numerator has at most one turning point. Its
        # monotone subintervals therefore isolate every stationary ratio.
        nodes=[0.0,self.dt]
        if E1 != 0.0:
            turning=(E1-c*E0)/(c*E1)
            if 0.0<turning<self.dt:nodes.append(turning)
        nodes.sort(); candidates=nodes.copy()
        for left,right in zip(nodes[:-1],nodes[1:]):
            fl=derivative_numerator(left);fr=derivative_numerator(right)
            if fl*fr<0:
                candidates.append(brentq(derivative_numerator,left,right,xtol=1e-15))
        def ratio(t):
            decay=math.exp(-c*t)
            den=k*(D0+D1*t+D2*decay)
            if den<=0:raise SafetyInfeasibleError('Invalid ZOH CBF input coefficient')
            return (C0+C2*decay)/den
        values=[ratio(t) for t in candidates]
        return max(values) if lower else min(values)

    def filter_action(self,x_p,v_p_mms,u_rl,k_gain=None,damp_c=None):
        """Return the scalar QP projection; optional model arguments are checked.

        The legacy teaching interface accepts k_gain/damp_c here. Changing
        only the safety model while retaining another propagation model would
        invalidate the certificate, so configure both through the constructor.
        """
        if k_gain is not None and not math.isclose(float(k_gain),self.k_gain):
            raise ValueError('k_gain must match the declared propagation model')
        if damp_c is not None and not math.isclose(float(damp_c),self.damp_c):
            raise ValueError('damp_c must match the declared propagation model')
        x=float(x_p);v=float(v_p_mms);proposed=float(u_rl)
        if not all(math.isfinite(y) for y in (x,v,proposed)):
            raise ValueError('State and action must be finite')
        hu=self.x_max-self.margin-x
        hl=x-self.x_min-self.margin
        pu=self.kappa1*hu-v;pl=self.kappa1*hl+v
        # Tiny floating roundoff at the asymptotic boundary is tolerated;
        # no physical state or input is clipped to fabricate invariance.
        if min(hu,hl,pu,pl)<-1e-8:
            raise SafetyInfeasibleError('State outside certified HOCBF initial set')
        lo=max(self.u_min,self._bound(hl,v,lower=True))
        hi=min(self.u_max,self._bound(hu,v))
        if lo>hi:
            raise SafetyInfeasibleError(f'No admissible action: lower={lo:g}, upper={hi:g}')
        applied=float(np.clip(proposed,lo,hi))
        self.last_info=FilterDiagnostics(proposed,applied,lo,hi,
            self.compute_barrier_h(x),hu,hl,pu,pl,abs(applied-proposed)>1e-10)
        return applied

    @staticmethod
    def _affine_exponential_min(a0,a1,a2,c,duration):
        """Minimum of a0+a1*t+a2*exp(-c*t) on a closed interval."""
        times=[0.0,float(duration)]
        if a2 != 0.0:
            ratio=a1/(c*a2)
            if ratio>0.0:
                stationary=-math.log(ratio)/c
                if 0.0<stationary<duration:times.append(stationary)
        return min(a0+a1*t+a2*math.exp(-c*t) for t in times)

    def interval_diagnostics(self,x_p,v_p_mms,u,dt=None):
        """Analytic interval minima, without clipping the physical trajectory."""
        duration=self.dt if dt is None else float(dt)
        if duration<0:raise ValueError('Duration must be nonnegative')
        x=float(x_p);v=float(v_p_mms);u=float(u)
        c=self.damp_c;k=self.k_gain;kap=self.kappa1
        hu=self.x_max-self.margin-x;hl=x-self.x_min-self.margin
        times=[0.0,duration]
        v_inf=k*u/c;v_transient=v-v_inf
        if v_transient != 0.0:
            ratio=-v_inf/v_transient
            if ratio>0.0:
                stationary=-math.log(ratio)/c
                if 0.0<stationary<duration:times.append(stationary)
        positions=[self.state_transition(x,v,u,t)[0] for t in times]
        min_h=min(self.compute_barrier_h(pos) for pos in positions)
        min_hu=self.x_max-self.margin-max(positions)
        min_hl=min(positions)-self.x_min-self.margin
        min_psi_up=self._affine_exponential_min(
            kap*hu-kap*v_transient/c-v_inf,
            -kap*v_inf,(kap/c-1)*v_transient,c,duration)
        min_psi_low=self._affine_exponential_min(
            kap*hl+kap*v_transient/c+v_inf,
            kap*v_inf,(1-kap/c)*v_transient,c,duration)
        alpha=self.alpha1;beta=self.alpha2;d=c-alpha
        d0=1-d/c-beta/(c*c);d1=beta/c;d2=d/c+beta/(c*c)
        c2=(d+beta/c)*v
        min_psi2_up=self._affine_exponential_min(
            beta*(hu-v/c)-k*d0*u,-k*d1*u,c2-k*d2*u,c,duration)
        min_psi2_low=self._affine_exponential_min(
            beta*(hl+v/c)+k*d0*u,k*d1*u,-c2+k*d2*u,c,duration)
        return {'h_mechanical_min':float(min_h),
                'h_upper_mm_min':float(min_hu),'h_lower_mm_min':float(min_hl),
                'psi1_upper_mm_s_min':float(min_psi_up),
                'psi1_lower_mm_s_min':float(min_psi_low),
                'psi2_upper_mm_s2_min':float(min_psi2_up),
                'psi2_lower_mm_s2_min':float(min_psi2_low)}

    def state_transition(self,x_p,v_p_mms,u,dt=None):
        """Exact solution of the declared surrogate under a held command."""
        duration=self.dt if dt is None else float(dt)
        if duration<0:raise ValueError('Duration must be nonnegative')
        c=self.damp_c;k=self.k_gain
        decay=math.exp(-c*duration)
        phi=-math.expm1(-c*duration)/c
        x_next=x_p+phi*v_p_mms+(k/c)*(duration-phi)*u
        v_next=decay*v_p_mms+k*phi*u
        return float(x_next),float(v_next)
