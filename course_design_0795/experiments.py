"""统一真实被控对象/参考/初态/故障/采样时钟的确定性控制比较。"""
from pathlib import Path
from dataclasses import asdict,replace
import csv,json,time,platform
import numpy as np
from physics import Parameters,HydraulicTwin,trajectory,effective_voltage
from controllers import IncrementalPID,ModelAssistedController,LTVMPC


def calculate_metrics(rows,fault_time=.5,initial_u=0.):
    e=np.array([row['error_mm'] for row in rows]);u=np.array([row['u_v'] for row in rows]);t=np.array([row['time_s'] for row in rows])
    window=(t>=fault_time-1e-12)&(t<=fault_time+.15+1e-12)
    p1=max(row['p1_max_mpa'] for row in rows);p2=max(row['p2_max_mpa'] for row in rows)
    saturation=np.mean([row['saturation'] for row in rows])
    return dict(rmse_mm=float(np.sqrt(np.mean(e*e))),disturbance_mae_mm=float(np.max(abs(e[window]))) if window.any() else None,
                maximum_absolute_error_mm=float(np.max(abs(e))),TV_V=float(np.sum(abs(np.diff(np.r_[initial_u,u])))),
                p1_peak_mpa=p1,p2_peak_mpa=p2,p_peak_mpa=max(p1,p2),
                minimum_pressure_mpa=min(min(row['p1_min_mpa'],row['p2_min_mpa']) for row in rows),
                pressure_safe=bool(max(p1,p2)<=21+1e-6),saturation_fraction=float(saturation),
                final_error_mm=float(e[-1]),samples=len(rows))


def run_scenario(root,identified,scenario='extreme',horizon=30):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    stress=scenario=='braking';duration=.20 if stress else 1.2
    move=.8;distance=0. if stress else .08;origin=.08 if stress else .04
    fault_time=.5;dt=.001;times=np.arange(int(round(duration/dt))+1)*dt
    refs=trajectory(times,distance,move,origin)
    policies=['PID','ModelAssisted','LTV_MPC']
    if stress:policies=['PID','ModelAssisted','InputOnlyMPC','LTV_MPC']
    result={};all_rows=[]
    for policy in policies:
        truth=Parameters();plant=HydraulicTwin(truth);plant.reset([origin,0.,truth.Pr,truth.Pr])
        if policy=='PID':controller=IncrementalPID()
        elif policy=='ModelAssisted':controller=ModelAssistedController(identified)
        else:controller=LTVMPC(identified,dt,horizon,policy!='InputOnlyMPC',move,distance,origin)
        if stress:
            initial,cruise_voltage=plant.equilibrium(.7,origin);plant.reset(initial)
            if isinstance(controller,IncrementalPID):controller.u0=cruise_voltage
            if hasattr(controller,'previous_w'):controller.previous_w=float(effective_voltage(cruise_voltage))
            if hasattr(controller,'observer'):
                controller.observer.previous_v=.7;controller.observer.initialized=True
        rows=[];begin=time.perf_counter();failure=None
        initial_u=cruise_voltage if stress else 0.
        ext=np.array([origin,origin,plant.state[2],plant.state[2],plant.state[3],plant.state[3]]);last_ext=ext.copy();last_u=0.
        for k,t in enumerate(times):
            load=5000. if scenario=='extreme' and t>=fault_time-1e-12 else 0.
            ratio=.5 if scenario=='extreme' and t>=fault_time-1e-12 else 1.
            if scenario=='extreme' and k==int(fault_time/dt):plant.set_beta_ratio(.5)
            y=plant.state.copy();control_start=time.perf_counter()
            try:u=controller.update(refs[k],y,t)
            except RuntimeError as error:failure=str(error);break
            elapsed=time.perf_counter()-control_start
            diag=controller.last
            row=dict(policy=policy,scenario=scenario,time_s=float(t),reference_m=float(refs[k,0]),reference_v_m_s=float(refs[k,1]),
                     x_m=float(y[0]),v_m_s=float(y[1]),p1_pa=float(y[2]),p2_pa=float(y[3]),
                     error_mm=float((refs[k,0]-y[0])*1000),u_v=float(u),load_n=load,beta_ratio=ratio,
                     p1_min_mpa=float(last_ext[2]/1e6),p1_max_mpa=float(last_ext[3]/1e6),
                     p2_min_mpa=float(last_ext[4]/1e6),p2_max_mpa=float(last_ext[5]/1e6),
                     saturation=int(diag.get('saturation',False)),estimated_load_n=float(diag.get('estimated_load_n',0.)),
                     control_seconds=elapsed,qp_seconds=float(diag.get('qp_seconds',0.)),
                     qp_status=str(diag.get('qp_status','not_applicable')),qp_iterations=int(diag.get('qp_iterations',0)),
                     qp_dual_residual=float(diag.get('qp_dual_residual',0.)),qp_residual=float(diag.get('qp_residual',0.)),predicted_pmax_mpa=float(diag.get('predicted_max_pressure_mpa',0.)))
            rows.append(row)
            if hasattr(controller,'last_matrices') and k in [0,500,60]:
                np.savez(root/f'{scenario}_{policy}_matrices_k{k:04d}.npz',**controller.last_matrices)
            if k==len(times)-1:break
            try:_,last_ext=plant.step(u,load)
            except RuntimeError as error:failure=str(error);break
        metrics=calculate_metrics(rows,fault_time,initial_u);metrics.update(completed=bool(len(rows)==len(times) and failure is None),
                                                                 failure=failure,initial_voltage_v=float(initial_u),qp_approximate_steps=sum(1 for r in rows if r['qp_status']=='maximum iterations reached'),wall_seconds=time.perf_counter()-begin,
                                                                 mean_control_ms=float(np.mean([r['control_seconds'] for r in rows])*1000),
                                                                 p95_control_ms=float(np.quantile([r['control_seconds'] for r in rows],.95)*1000))
        result[policy]=metrics;all_rows.extend(rows)
        print(scenario,policy,metrics,flush=True)
    with (root/f'{scenario}_traces.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (root/f'{scenario}_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
