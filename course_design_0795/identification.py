"""第二章：学号种子、独立合成观测、无量纲DE反演beta及Ct。"""
from dataclasses import replace,asdict
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from physics import Parameters,simulate_inputs
from filtering import CascadeFilter


def decode(theta):return 1e9*(.5+1.3*float(theta[0])),10**(-12.5+1.5*float(theta[1]))


def generate_dataset(seed,validation=False):
    np.random.seed(seed)
    rng=np.random.default_rng(seed if not validation else seed+1)
    t=np.arange(1601)*.001
    if validation:
        u=2.1*np.sin(2*np.pi*2.3*t[:-1])+.9*np.sin(2*np.pi*13*t[:-1])+.45*np.cos(2*np.pi*37*t[:-1])
    else:
        u=2.4*np.sin(2*np.pi*1.5*t[:-1])+1.3*np.sin(2*np.pi*8*t[:-1])+.8*np.sin(2*np.pi*23*t[:-1])+.4*np.sin(2*np.pi*47*t[:-1])
        u[(t[:-1]>.5)&(t[:-1]<.65)]=0.;u[(t[:-1]>1.3)&(t[:-1]<1.45)]=0.
    truth=Parameters();initial=np.array([.12,0.,1e5,1e5]);loads=np.zeros(len(u))
    clean=simulate_inputs(u,loads,truth.vector(),initial)
    measured=clean[:,[0,2,3]]+rng.normal(size=(len(clean),3))*np.array([2e-5,2e4,2e4])
    spikes=rng.random((len(clean),3))<.008
    measured+=spikes*rng.choice([-1,1],(len(clean),3))*np.array([.001,7e5,7e5])
    return t,u,loads,initial,clean,measured


def identify(root,seed=795,maxiter=35,popsize=12):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);np.random.seed(seed)
    t,u,loads,initial,clean,measured=generate_dataset(seed)
    filter_=CascadeFilter();target=filter_(measured);truth_filtered=filter_(clean[:,[0,2,3]])
    scale=np.array([.005,2e6,2e6]);history=[];evaluations=0;start=time.perf_counter()
    def objective(theta):
        nonlocal evaluations
        evaluations+=1;beta,ct=decode(theta)
        p=replace(Parameters(),beta=beta,Ct=ct)
        predicted=simulate_inputs(u,loads,p.vector(),initial)
        if not np.isfinite(predicted).all() or np.max(abs(predicted[:,2:]))>1e9:return 1e12
        # 同一因果滤波作用于模拟与观测，避免把滤波时延误辨识成刚度变化。
        modeled=filter_(predicted[:,[0,2,3]])
        residual=(modeled[20:]-target[20:])/scale
        return float(np.mean(residual*residual))
    def callback(theta,convergence):
        beta,ct=decode(theta);loss=objective(theta)
        history.append(dict(generation=len(history)+1,beta_pa=beta,Ct_m3_s_pa=ct,loss=loss))
        if len(history)%10==0:print('DE',history[-1],flush=True)
        return False
    result=differential_evolution(objective,[(0.,1.),(0.,1.)],seed=seed,popsize=popsize,
                                  maxiter=maxiter,tol=1e-7,atol=1e-10,polish=True,callback=callback,workers=1)
    beta,ct=decode(result.x);identified=replace(Parameters(),beta=beta,Ct=ct)
    tv,uv,lv,iv,validation_clean,validation_measured=generate_dataset(seed,validation=True)
    prediction=simulate_inputs(uv,lv,identified.vector(),iv)
    validation_error=np.sqrt(np.mean((prediction-validation_clean)**2,axis=0))
    filtered_rmse=np.sqrt(np.mean((target[20:]-truth_filtered[20:])**2,axis=0))
    raw_rmse=np.sqrt(np.mean((measured[20:]-clean[20:,[0,2,3]])**2,axis=0))
    pressure_trace=np.c_[t,clean[:,0],clean[:,2:4],measured,target,truth_filtered]
    np.savetxt(root/'identification_signals.csv',pressure_trace,delimiter=',',
               header='time_s,true_x_m,true_p1_pa,true_p2_pa,raw_x_m,raw_p1_pa,raw_p2_pa,filtered_x_m,filtered_p1_pa,filtered_p2_pa,filtered_true_x_m,filtered_true_p1_pa,filtered_true_p2_pa',comments='')
    np.savez(root/'identification_dataset.npz',t=t,u=u,loads=loads,initial=initial,clean=clean,measured=measured,
             validation_t=tv,validation_u=uv,validation_initial=iv,validation_clean=validation_clean,validation_prediction=prediction)
    import csv
    with (root/'de_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    summary=dict(seed_integer=seed,global_numpy_seed_word=int(np.random.get_state()[1][0]),student_id_last_four=f'{seed:04d}',truth=asdict(Parameters()),identified=asdict(identified),
                 beta_relative_error=(beta/Parameters().beta-1),Ct_relative_error=(ct/Parameters().Ct-1),
                 objective=float(result.fun),optimizer_success=bool(result.success),optimizer_message=str(result.message),
                 generations=int(result.nit),objective_evaluations=evaluations,wall_seconds=time.perf_counter()-start,
                 raw_noise_rmse=raw_rmse.tolist(),filtered_noise_rmse_against_causally_filtered_truth=filtered_rmse.tolist(),
                 independent_validation_rmse_si=validation_error.tolist(),
                 measurement_origin='independently generated synthetic observations, not physical bench measurements',
                 filter=dict(window=5,cutoff_hz=120,order=2,causal=True,model_and_observation_filtered_identically=True),
                 normalized_bounds=dict(beta_pa=[.5e9,1.8e9],Ct_m3_s_pa=[10**-12.5,1e-11]))
    (root/'identification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return identified,summary
