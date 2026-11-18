"""独立Radau重放；指标重算；压力约束矩阵、随机种子和完整时钟核验。"""
from pathlib import Path
import csv,json,time
import numpy as np
from scipy.integrate import solve_ivp
from physics import Parameters,rhs_effective,effective_voltage
from experiments import calculate_metrics


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def verify(root,seed=795):
    root=Path(root);summary=json.loads((root/'identification_summary.json').read_text())
    checks={'student_seed_matches':summary['seed_integer']==seed,'global_numpy_seed_matches_student':summary['global_numpy_seed_word']==seed,'beta_identified_within_2percent':abs(summary['beta_relative_error'])<.02,
            'Ct_identified_within_2percent':abs(summary['Ct_relative_error'])<.02,
            'held_out_position_rmse_below_0_05mm':summary['independent_validation_rmse_si'][0]<5e-5}
    results={};start=time.perf_counter()
    for scenario in ['nominal','extreme','braking']:
        records=read(root/f'{scenario}_traces.csv');metrics=json.loads((root/f'{scenario}_metrics.json').read_text())
        for policy in metrics:
            group=[r for r in records if r['policy']==policy]
            initial=np.array([float(group[0][k]) for k in ['x_m','v_m_s','p1_pa','p2_pa']]);state=initial.copy()
            p=Parameters().vector();max_error=np.zeros(4);reference_pmax=max(state[2:])/1e6
            for i,row in enumerate(group):
                saved=np.array([float(row[k]) for k in ['x_m','v_m_s','p1_pa','p2_pa']])
                max_error=np.maximum(max_error,abs(saved-state))
                if i==len(group)-1:break
                pv=p.copy();pv[21]=float(row['beta_ratio']);load=float(row['load_n']);w=effective_voltage(float(row['u_v']))
                sol=solve_ivp(lambda t,y:rhs_effective(y,w,load,pv),(0.,.001),state,
                              method='Radau',rtol=3e-8,atol=[1e-11,1e-10,.002,.002],max_step=.000125)
                if not sol.success:raise RuntimeError(sol.message)
                state=sol.y[:,-1];reference_pmax=max(reference_pmax,float(sol.y[2:].max()/1e6))
            # 每组按自己的保存动作重放，所有策略均可复核，不调用控制器。
            tolerance=np.array([1e-6,3e-4,1e4,1e4])
            passed=bool(np.all(max_error<=tolerance))
            checks[f'{scenario}/{policy}/independent_radau_replay']=passed
            if scenario in ['nominal','extreme']:
                checks[f'{scenario}/{policy}/full_common_clock']=len(group)==1201 and all(abs(float(r['time_s'])-.001*i)<1e-10 for i,r in enumerate(group))
            numeric=[]
            for row in group:
                item={k:(v if k in ['policy','scenario','qp_status'] else float(v)) for k,v in row.items()}
                numeric.append(item)
            recomputed=calculate_metrics(numeric,initial_u=metrics[policy]['initial_voltage_v'])
            checks[f'{scenario}/{policy}/metrics_recompute']=all(abs(recomputed[k]-metrics[policy][k])<1e-8 for k in ['rmse_mm','TV_V','p_peak_mpa'])
            results[f'{scenario}/{policy}']=dict(max_state_error_si=max_error.tolist(),tolerance_si=tolerance.tolist(),
                                                saved_substep_pmax_mpa=metrics[policy]['p_peak_mpa'],
                                                independent_radau_pmax_mpa=reference_pmax,passed=passed,complete=metrics[policy]['completed'])
            print('Radau replay',scenario,policy,max_error,flush=True)
    for path in sorted(root.glob('*LTV_MPC_matrices_*.npz')):
        with np.load(path) as data:
            residual=max(float(np.max(data['l']-data['A']@data['solution'])),float(np.max(data['A']@data['solution']-data['upper'])),0.)
            n=len(data['solution']);indices=[4*i+j for i in range(n) for j in [2,3]]
            checks[path.stem+'/pressure_rows_correct']=bool(np.allclose(data['A'][-2*n:],data['Phi'][indices]))
            checks[path.stem+'/qp_feasible']=residual<=.002
    brake=json.loads((root/'braking_metrics.json').read_text())
    checks['braking/constrained_mpc_completed']=brake['LTV_MPC']['completed']
    checks['braking/constrained_mpc_below_21MPa']=brake['LTV_MPC']['p_peak_mpa']<=21
    checks['braking/input_only_exceeds_21MPa']=brake['InputOnlyMPC']['p_peak_mpa']>21
    result=dict(checks=checks,all_checks_passed=all(checks.values()),replays=results,wall_seconds=time.perf_counter()-start,
                safety_scope='sample-time QP pressure upper limits and independently checked trajectories; no robust invariant-set theorem or physical hardware certification',
                reference_state_tolerances='1 um position, 0.3 mm/s velocity, 0.01 MPa pressure; per-run actual errors disclosed',
                identification_data_origin='synthetic, ground truth and noisy observations both saved',
                exclusion='SAC/HOCBF/PHM/RUL not implemented by explicit user scope')
    (root/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['all_checks_passed']:raise AssertionError('验证失败项: '+str([k for k,v in checks.items() if not v]))
    return result
