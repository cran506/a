"""Reproducible two-state HOCBF demonstration, separate from hydraulic SAC.

The supplied textbook filter assumes x_dot=v, v_dot=1850*u-14*v in mm/s
units. This script tests exactly that model, not the four pressure/position
hydraulic states used to train SAC. No impact trajectory is fabricated:
unprotected motion terminates at its first mechanical-boundary crossing.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from safety_filter import HOCBFSafetyFilter, SafetyInfeasibleError


def validate_filter_numerically(seed=8103,n_cases=128):
    """Independent DOP853 dense-flow check of random, feasible QP projections."""
    rng=np.random.default_rng(seed)
    filt=HOCBFSafetyFilter()
    cases=0;attempts=0
    max_flow_error=0.0;max_qp_violation=0.0
    min_dense_h=math.inf;min_dense_psi1=math.inf;min_dense_psi2=math.inf
    max_analytic_min_error=0.0
    while cases<n_cases and attempts<10*n_cases:
        attempts+=1
        x=float(rng.uniform(1.21,98.79))
        hl=x-1.2;hu=98.8-x
        v=float(rng.uniform(-24*hl,24*hu))
        proposed=float(rng.uniform(-10,10))
        try:u=filt.filter_action(x,v,proposed)
        except SafetyInfeasibleError:continue
        diagnostic=filt.last_info
        max_qp_violation=max(max_qp_violation,diagnostic.lower_v-u,u-diagnostic.upper_v,0.0)
        sol=solve_ivp(lambda _t,z:[z[1],1850*u-14*z[1]],
            [0,filt.dt],[x,v],method='DOP853',rtol=1e-11,atol=1e-11,dense_output=True)
        if not sol.success:raise RuntimeError('Independent HOCBF validation integration failed')
        times=np.linspace(0,filt.dt,101)
        states=sol.sol(times)
        exact=np.array([filt.state_transition(x,v,u,t) for t in times]).T
        max_flow_error=max(max_flow_error,float(np.max(np.abs(states-exact))))
        hu=98.8-states[0];hl=states[0]-1.2
        psi1u=24*hu-states[1];psi1l=24*hl+states[1]
        psi2u=576*hu-34*states[1]-1850*u
        psi2l=576*hl+34*states[1]+1850*u
        hs=(100-states[0])*states[0]/10000
        min_dense_h=min(min_dense_h,float(hs.min()))
        min_dense_psi1=min(min_dense_psi1,float(psi1u.min()),float(psi1l.min()))
        min_dense_psi2=min(min_dense_psi2,float(psi2u.min()),float(psi2l.min()))
        analytic=filt.interval_diagnostics(x,v,u)
        # The dense grid is an independent numerical lower-resolution check;
        # analytic extrema can lie between its nodes, so require one-sided agreement.
        for name,dense in [('psi1_upper_mm_s_min',psi1u),('psi1_lower_mm_s_min',psi1l),
                           ('psi2_upper_mm_s2_min',psi2u),('psi2_lower_mm_s2_min',psi2l)]:
            max_analytic_min_error=max(max_analytic_min_error,analytic[name]-float(dense.min()))
        cases+=1
    reject_bad_initial=False
    try:filt.filter_action(99.0,0.0,8.5)
    except SafetyInfeasibleError:reject_bad_initial=True
    reject_bad_psi=False
    try:filt.filter_action(98.0,100.0,8.5)
    except SafetyInfeasibleError:reject_bad_psi=True
    reject_empty_qp=False
    try:HOCBFSafetyFilter(u_min=-.01,u_max=.01).filter_action(50.0,1000.0,8.5)
    except SafetyInfeasibleError:reject_empty_qp=True
    checks={
        'all_requested_random_cases_checked':cases==n_cases,
        'independent_flow_matches_exact_zoh':max_flow_error<1e-7,
        'all_qp_projections_feasible':max_qp_violation<1e-10,
        'dense_mechanical_barriers_nonnegative':min_dense_h>=-1e-10,
        'dense_initial_set_conditions_preserved':min_dense_psi1>=-1e-7,
        'dense_hocbf_inequalities_preserved':min_dense_psi2>=-1e-6,
        'analytic_minima_do_not_overestimate_dense_minima':max_analytic_min_error<1e-6,
        'invalid_inner_set_is_rejected':reject_bad_initial,
        'invalid_high_order_initial_state_is_rejected':reject_bad_psi,
        'infeasible_actuator_intersection_is_rejected':reject_empty_qp,
    }
    return {'seed':seed,'requested_cases':n_cases,'feasible_cases_checked':cases,
        'random_state_attempts':attempts,'max_flow_error_mm_or_mm_s':max_flow_error,
        'max_qp_violation_v':max_qp_violation,
        'min_dense_physical_barrier':min_dense_h,
        'min_dense_psi1_mm_s':min_dense_psi1,'min_dense_psi2_mm_s2':min_dense_psi2,
        'max_analytic_minimum_overestimate':max_analytic_min_error,
        'checks':checks,'passed':bool(all(checks.values()))}


def run_safety_experiment(output_dir,duration=1.0):
    """Save raw trajectory and verification for the declared sampled model.

    Returns {'summary': ..., 'data': {column: np.ndarray}, 'csv_path': ...,
    'summary_path': ...}. An exact event row is inserted at the unprotected
    collision time; it is not an extra control update. Subsequent unprotected
    states are NaN, explicitly denoting an experiment terminated at impact.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    filt=HOCBFSafetyFilter()
    dt=filt.dt;n_steps=int(round(float(duration)/dt))
    if n_steps<1 or not math.isclose(n_steps*dt,duration,abs_tol=1e-12):
        raise ValueError('duration must be a positive integer multiple of 1 ms')
    initial_x=20.0;initial_v=0.0;proposed=8.5
    def unsafe_state(t):return filt.state_transition(initial_x,initial_v,proposed,t)
    collision_time=None
    if unsafe_state(duration)[0]>=100.0:
        collision_time=float(brentq(lambda t:unsafe_state(t)[0]-100.0,0,duration,xtol=1e-14))
    rows=[];x=initial_x;v=initial_v
    interval_minima={name:math.inf for name in (
        'h_mechanical_min','h_upper_mm_min','h_lower_mm_min',
        'psi1_upper_mm_s_min','psi1_lower_mm_s_min',
        'psi2_upper_mm_s2_min','psi2_lower_mm_s2_min')}
    def make_row(t,xs,vs,u,diagnostic,interval,control_sample,event=False):
        if collision_time is None or t<=collision_time+1e-12:
            xu,vu=unsafe_state(t)
            if event:xu=100.0  # value at the exact root, not trajectory clipping
        else:xu=vu=float('nan')
        return {'time_s':float(t),'x_safe_mm':float(xs),'v_safe_mm_s':float(vs),
            'u_rl_v':proposed,'u_safe_v':float(u),'x_unsafe_mm':float(xu),
            'v_unsafe_mm_s':float(vu),'h_mechanical':filt.compute_barrier_h(xs),
            'h_upper_mm':98.8-xs,'h_lower_mm':xs-1.2,
            'psi1_upper_mm_s':24*(98.8-xs)-vs,'psi1_lower_mm_s':24*(xs-1.2)+vs,
            'qp_lower_v':diagnostic.lower_v if diagnostic else float('nan'),
            'qp_upper_v':diagnostic.upper_v if diagnostic else float('nan'),
            'intervened':float(abs(u-proposed)>1e-10),
            'is_control_sample':float(control_sample),'unsafe_collision_event':float(event),
            **{'interval_'+name:interval[name] if interval else float('nan') for name in interval_minima}}
    for i in range(n_steps+1):
        t=i*dt;u=filt.filter_action(x,v,proposed,k_gain=1850.0,damp_c=14.0)
        diag=filt.last_info
        interval=filt.interval_diagnostics(x,v,u,dt if i<n_steps else 0.0)
        if i<n_steps:
            for name,value in interval.items():interval_minima[name]=min(interval_minima[name],value)
        event_here=collision_time is not None and abs(t-collision_time)<1e-12
        rows.append(make_row(t,x,v,u,diag,interval,True,event_here))
        if i<n_steps:
            if collision_time is not None and t+1e-12<collision_time<t+dt-1e-12:
                xe,ve=filt.state_transition(x,v,u,collision_time-t)
                rows.append(make_row(collision_time,xe,ve,u,None,None,False,True))
            x,v=filt.state_transition(x,v,u)
    fields=list(rows[0])
    csv_path=out/'safety.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    data={name:np.array([row[name] for row in rows],dtype=float) for name in fields}
    main=data['is_control_sample']==1.0
    def first_time(mask):
        times=data['time_s'][main][mask]
        return float(times[0]) if times.size else None
    numerical=validate_filter_numerically()
    command=data['u_safe_v'][main]
    safe_times=data['time_s'][main]
    safe_positions=data['x_safe_mm'][main]
    checks={
        'all_safe_commands_inside_voltage_limits':bool(np.max(np.abs(command))<=10.0+1e-12),
        'entire_zoh_intervals_inside_physical_limits':interval_minima['h_mechanical_min']>=-1e-10,
        'entire_zoh_intervals_inside_guard_limits':min(interval_minima['h_upper_mm_min'],interval_minima['h_lower_mm_min'])>=-1e-8,
        'entire_zoh_intervals_preserve_psi1':min(interval_minima['psi1_upper_mm_s_min'],interval_minima['psi1_lower_mm_s_min'])>=-1e-7,
        'entire_zoh_intervals_satisfy_hocbf':min(interval_minima['psi2_upper_mm_s2_min'],interval_minima['psi2_lower_mm_s2_min'])>=-1e-6,
        'unprotected_collision_recorded':collision_time is not None,
        'independent_random_flow_checks_passed':numerical['passed'],
    }
    summary={
        'experiment':'sampled_two_state_hocbf_safety_demo',
        'model_scope':'x_dot=v; v_dot=1850*u-14*v; x in mm, v in mm/s. This is an explicit second-order surrogate, not the four-state hydraulic pressure ODE.',
        'relative_order_warning':'Position has relative degree 3 in the four-state pressure model, or 4 when a first-order valve state is added. This second-order certificate does not transfer to either plant.',
        'physical_limit_mm':[0.0,100.0],'guard_limit_mm':[1.2,98.8],
        'k_gain_mm_s2_per_v':1850.0,'damping_per_s':14.0,'sample_time_s':dt,
        'kappa1_per_s':24.0,'kappa2_per_s':24.0,'alpha_sum_per_s':48.0,
        'alpha_product_per_s2':576.0,
        'reference_coefficient_note':'The supplied alpha_product=580 gives discriminant 48^2-4*580=-16; use the factorable kappa1=kappa2=24 product 576 for the stated recursive HOCBF proof.',
        'duration_s':duration,'initial_x_mm':initial_x,'initial_v_mm_s':initial_v,
        'proposed_action_v':proposed,'first_intervention_s':first_time(command<proposed-1e-10),
        'minimum_safe_voltage_v':float(command.min()),
        'minimum_safe_voltage_time_s':float(safe_times[np.argmin(command)]),
        'safe_final_position_mm':float(safe_positions[-1]),
        'safe_final_velocity_mm_s':float(data['v_safe_mm_s'][main][-1]),
        'safe_maximum_position_mm':float(safe_positions.max()),
        'safe_position_at_0_25_s_mm':float(np.interp(.25,safe_times,safe_positions)) if duration>=.25 else None,
        'unsafe_collision_time_s':collision_time,'unsafe_collision_x_mm':100.0 if collision_time is not None else None,
        'unsafe_trajectory_convention':'Terminate at the exact first x=100 mm event; subsequent states are NaN. The inserted event row does not change the sampled safe control.',
        'interval_minima':{name:float(value) for name,value in interval_minima.items()},
        'independent_random_validation':numerical,'checks':checks,'passed':bool(all(checks.values())),
        'reference_numeric_difference':'Computed minimum control is approximately -0.30 V, not the illustrative -3.1 V; the plant approaches 98.8 mm asymptotically. No signal is spliced to the reference figure.'}
    summary_path=out/'safety_summary.json'
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if not summary['passed']:
        raise RuntimeError(f'Safety verification failed: {summary_path}')
    return {'summary':summary,'data':data,'csv_path':str(csv_path),'summary_path':str(summary_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--duration',type=float,default=1.0)
    args=parser.parse_args()
    result=run_safety_experiment(args.output_dir,args.duration)
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
