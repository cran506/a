"""固定剪枝掩码下的局部输入雅可比蒸馏；教师为最终 FP32 神经 actor。

这只是导出校准阶段的浮点稀疏网络拟合；没有对整数舍入操作冒充求导。
最终部署仍由独立 INT8/INT32 内核执行，并须另行检查实际闭环。
"""
from __future__ import annotations
import copy
import time
import numpy as np
import torch
from torch import nn


def _network(weights,biases):
    modules=[]
    for index,(weight,bias) in enumerate(zip(weights,biases)):
        layer=nn.Linear(weight.shape[1],weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(np.asarray(weight,dtype=np.float32)))
            layer.bias.copy_(torch.from_numpy(np.asarray(bias,dtype=np.float32)))
        modules.append(layer)
        if index<len(weights)-1:modules.append(nn.ReLU())
    modules.append(nn.Tanh())
    return nn.Sequential(*modules)


def consolidate_edge(edge,cal,steps=8000,batch_size=256,seed=2029):
    """Mutate only the already pruned weights/biases; return auditable statistics."""
    started=time.perf_counter();torch.set_num_threads(1)
    if steps<1 or batch_size<1:raise ValueError('steps and batch_size must be positive')
    cal=np.asarray(cal,dtype=np.float32)
    if cal.ndim!=2 or cal.shape[1]!=5 or not np.isfinite(cal).all() or len(cal)==0:
        raise ValueError('Expected finite five-feature calibration observations')
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    teacher=_network(edge.weights,edge.biases).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    student=_network(edge.pruned_weights,edge.pruned_biases).train()
    layers=[module for module in student if isinstance(module,nn.Linear)]
    masks=[torch.from_numpy(np.asarray(mask,dtype=np.float32)) for mask in edge.masks]
    anchors_count=24000
    raw=np.column_stack([rng.uniform(-.015,.015,anchors_count),
                         rng.uniform(-.25,.25,anchors_count),
                         rng.uniform(-300.,300.,anchors_count),
                         rng.uniform(2.4,5.8,anchors_count),np.full(anchors_count,21.)])
    anchors=((raw-np.array([0.,0.,0.,10.,21.]))/np.array([10.,50.,500.,8.,2.])).astype(np.float32)
    x=torch.tensor(cal);near=torch.tensor(anchors)
    with torch.no_grad():labels=teacher(x);initial_mse=float(((student(x)-labels)**2).mean())
    jac_weights=torch.tensor([1.,1.,100.,2.,2.])
    reference_near=near[:512].detach().requires_grad_(True)
    reference_labels=teacher(reference_near)
    reference_jac=torch.autograd.grad(reference_labels.sum(),reference_near)[0].detach()
    reference_labels=reference_labels.detach()
    optimizer=torch.optim.Adam(student.parameters(),lr=2e-4)
    best=None;best_state=None;logs=[]
    def measure():
        with torch.no_grad():mse=float(((student(x)-labels)**2).mean())
        anchor=reference_near.detach().requires_grad_(True)
        pred=student(anchor)
        jac=torch.autograd.grad(pred.sum(),anchor)[0]
        mse_near=float(((pred-reference_labels)**2).mean().detach())
        jerr=float((((jac-reference_jac)*jac_weights)**2).mean().detach())
        return mse,mse_near,jerr
    before=measure()
    for step in range(int(steps)):
        rate=2e-5+1.8e-4*.5*(1.+np.cos(np.pi*step/max(steps-1,1)))
        for group in optimizer.param_groups:group['lr']=rate
        indices=rng.integers(0,len(cal),batch_size)
        selected=rng.integers(0,anchors_count,max(64,batch_size//2))
        local=near[selected].detach().requires_grad_(True)
        target=teacher(local)
        target_jac=torch.autograd.grad(target.sum(),local)[0].detach()
        target=target.detach()
        output=student(local)
        jac=torch.autograd.grad(output.sum(),local,create_graph=True)[0]
        wide_loss=((student(x[indices])-labels[indices])**2).mean()
        near_loss=((output-target)**2).mean()
        derivative_loss=(((jac-target_jac)*jac_weights)**2).mean()
        loss=wide_loss+30.*near_loss+2.*derivative_loss
        optimizer.zero_grad(set_to_none=True);loss.backward()
        for layer,mask in zip(layers,masks):layer.weight.grad.mul_(mask)
        optimizer.step()
        with torch.no_grad():
            for layer,mask in zip(layers,masks):layer.weight.mul_(mask)
        if (step+1)%1000==0 or step==steps-1:
            mse,near_mse,jac_mse=measure()
            record=dict(step=step+1,calibration_action_mse=mse,unloaded_anchor_action_mse=near_mse,
                        unloaded_weighted_jacobian_mse=jac_mse,learning_rate=rate)
            logs.append(record)
            objective=mse+30.*near_mse+2.*jac_mse
            if best is None or objective<best:
                best=objective;best_state=copy.deepcopy(student.state_dict());selected_step=step+1
    student.load_state_dict(best_state)
    edge.pruned_weights=[layer.weight.detach().numpy().copy() for layer in layers]
    edge.pruned_biases=[layer.bias.detach().numpy().copy() for layer in layers]
    for weight,mask in zip(edge.pruned_weights,edge.masks):
        if np.any(weight[~mask]!=0.):raise AssertionError('Pruning mask changed during consolidation')
    edge.integer_layers=edge._build_integer_layers(cal)
    final=measure()
    return dict(method='Fixed-mask sparse action and input-Jacobian distillation from final deployed FP32 neural actor',
        steps=int(steps),batch_size=int(batch_size),seed=seed,environment_interactions=0,
        calibration_observation_count=len(cal),unloaded_regression_anchors=anchors_count,
        synthetic_anchors_are_not_physical_trajectories=True,teacher='Full final FP32 neural actor; no physical teacher at this stage',
        raw_anchor_ranges=dict(position_error_mm=[-.015,.015],velocity_error_mms=[-.25,.25],
                               acceleration_error_mms2=[-300.,300.],load_pressure_mpa=[2.4,5.8],supply_pressure_mpa=[21.,21.]),
        loss='calibration_action_MSE+30*unloaded_action_MSE+2*mean(((jac_student-jac_full)*[1,1,100,2,2])^2)',
        selection_metric='calibration and synthetic unloaded-anchor joint objective only',
        selected_step=selected_step,initial_action_mse=initial_mse,final_action_mse=final[0],
        initial_unloaded_anchor_action_mse=before[1],final_unloaded_anchor_action_mse=final[1],
        initial_unloaded_weighted_jacobian_mse=before[2],final_unloaded_weighted_jacobian_mse=final[2],
        unchanged_pruning_mask=True,integer_rounding_differentiated=False,
        wall_seconds=time.perf_counter()-started,history=logs)
