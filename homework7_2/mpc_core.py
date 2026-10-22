"""Affine LTV hydraulic MPC with incremental voltage decisions.

Prediction units are [relative position mm, velocity m/s, P1 MPa, P2 MPa,
valve voltage V].  The augmented sixth state is the previous applied command.
Only position and velocity are tracked.  Pressure constraints refer to the
nonlinear model prediction; they do not prove safety of a mismatched plant.
"""
from __future__ import annotations

import math
import time
import numpy as np
import scipy.sparse as sparse
import osqp


class MPCSolveError(RuntimeError):
    """The controller could not produce an independently validated command."""
    def __init__(self,message,diagnostics=None):
        super().__init__(message)
        self.diagnostics=diagnostics


def augment_incremental_model(ad, bd, cd):
    """z_next=Ad*z+Bd*u+cd, u=u_previous+delta_u; single input only."""
    ad=np.asarray(ad,dtype=float)
    bd=np.asarray(bd,dtype=float).reshape(-1)
    cd=np.asarray(cd,dtype=float).reshape(-1)
    n=len(cd)
    if ad.shape!=(n,n) or bd.shape!=(n,) or not all(
            np.all(np.isfinite(value)) for value in (ad,bd,cd)):
        raise ValueError('Invalid affine state model dimensions or coefficients')
    aa=np.zeros((n+1,n+1));aa[:n,:n]=ad;aa[:n,n]=bd;aa[n,n]=1.
    ba=np.r_[bd,1.]
    ca=np.r_[cd,0.]
    return aa,ba,ca


def condense_affine(models):
    """Return F, Phi, b for future states 1..N of a time-varying affine model.

    Each item is (A_i, B_i, c_i), with one scalar control increment.  Stacking
    is time-major; Xi=F*xi_initial+Phi*delta_U+b includes every affine offset.
    """
    if not models:
        raise ValueError('The prediction horizon must contain a model')
    n=np.asarray(models[0][0]).shape[0];h=len(models)
    free=np.eye(n);response=np.zeros((n,h));offset=np.zeros(n)
    f,p,b=[],[],[]
    for i,(a,control,c) in enumerate(models):
        a=np.asarray(a,dtype=float);control=np.asarray(control,dtype=float).reshape(-1)
        c=np.asarray(c,dtype=float).reshape(-1)
        if a.shape!=(n,n) or control.shape!=(n,) or c.shape!=(n,) or not all(
                np.all(np.isfinite(value)) for value in (a,control,c)):
            raise ValueError('Inconsistent affine prediction model')
        free=a@free
        response=a@response;response[:,i]+=control
        offset=a@offset+c
        f.append(free.copy());p.append(response.copy());b.append(offset.copy())
    return np.vstack(f),np.vstack(p),np.concatenate(b)


def _dense_csc(matrix,upper=False):
    """Keep a fixed sparsity pattern, including coefficients currently zero."""
    matrix=np.asarray(matrix,dtype=float)
    if upper:
        rows,columns=np.triu_indices(matrix.shape[0])
    else:
        rows,columns=np.indices(matrix.shape);rows=rows.ravel();columns=columns.ravel()
    return sparse.csc_matrix((matrix[rows,columns],(rows,columns)),shape=matrix.shape)


class HydraulicMPC:
    """Successive local QPs, OSQP warm starts, and nonlinear feasibility checks.

    Default full mode constrains P2 <= p_safe-pressure_margin.  P1 upper and
    both lower limits are explicit opt-ins, rather than an undocumented extra
    constraint.  Increment and voltage bounds remain hard in both MPC modes.
    A retained plan is used only after a fresh nonlinear feasibility check;
    OSQP failures themselves raise MPCSolveError rather than silently holding.
    """
    def __init__(self,horizon=20,dt=.001,sys_params=None,
                 enable_pressure_constraint=True,model=None,*,
                 q_position=50000.,q_velocity=50.,move_weight=.01,Q=None,W=None,
                 u_min=-10.,u_max=10.,delta_u_max=4.5,p_safe=21.,
                 pressure_margin=.5,constrain_p1=False,pressure_lower=None,
                 pressure_upper=None,sequential_iterations=3,min_iterations=2,
                 eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,
                 primal_tolerance=1e-5,dual_tolerance=2e-5,
                 nonlinear_tolerance=2e-6,bootstrap_error_mm=.02,
                 linearization_reserve_mpa=.001,polishing=False):
        if not isinstance(polishing,(bool,np.bool_)):
            raise ValueError('polishing must be a boolean')
        if int(horizon)!=horizon or horizon<1:
            raise ValueError('horizon must be a positive integer')
        if Q is not None:
            q=np.asarray(Q,dtype=float)
            if q.ndim==2:
                if q.shape[0]!=q.shape[1] or not np.allclose(q,np.diag(np.diag(q))):
                    raise ValueError('This tracking API accepts diagonal Q only')
                q=np.diag(q)
            if q.ndim!=1 or len(q) not in (2,5,6) or np.any(q[2:]!=0.):
                raise ValueError('Q must track position/velocity, not pressure or input history')
            q_position,q_velocity=q[:2]
        if W is not None:
            w=np.asarray(W,dtype=float)
            if w.size!=1:
                raise ValueError('W is a positive scalar incremental-input weight')
            move_weight=float(w.ravel()[0])
        scalars=(dt,q_position,q_velocity,move_weight,u_min,u_max,delta_u_max,p_safe,
                 pressure_margin,eps_abs,eps_rel,primal_tolerance,dual_tolerance,
                 nonlinear_tolerance,bootstrap_error_mm,linearization_reserve_mpa)
        if (not all(math.isfinite(v) for v in scalars) or dt<=0
                or min(q_position,q_velocity)<0 or move_weight<=0 or u_min>=u_max
                or u_min < -10. or u_max>10. or delta_u_max<=0 or p_safe<=0
                or pressure_margin<0 or pressure_margin>=p_safe
                or min(eps_abs,eps_rel,primal_tolerance,dual_tolerance,nonlinear_tolerance)<=0
                or min(bootstrap_error_mm,linearization_reserve_mpa)<0 or int(sequential_iterations)!=sequential_iterations
                or int(min_iterations)!=min_iterations
                or not 1<=min_iterations<=sequential_iterations
                or int(max_iter)!=max_iter or max_iter<1):
            raise ValueError('Invalid MPC weights, bounds, tolerances, or iterations')
        if model is None:
            from hydraulic_plant import GreyBoxModel
            model=GreyBoxModel(config=sys_params)
        self.model=model
        self.horizon,self.dt=int(horizon),float(dt)
        self.nx_physical,self.nx_augmented=5,6
        self.enable_pressure_constraint=bool(enable_pressure_constraint)
        self.q_position,self.q_velocity=float(q_position),float(q_velocity)
        self.move_weight=float(move_weight)
        self.Q=np.diag([self.q_position,self.q_velocity]);self.W=self.move_weight
        # A common objective factor leaves the optimizer unchanged.  Keeping
        # the largest stage weight at 100 avoids near-zero move curvature.
        self.objective_scale=max(self.q_position,self.q_velocity,self.move_weight)/100.
        self.u_min,self.u_max,self.delta_u_max=float(u_min),float(u_max),float(delta_u_max)
        self.p_safe,self.pressure_margin=float(p_safe),float(pressure_margin)
        self.sequential_iterations,self.min_iterations=int(sequential_iterations),int(min_iterations)
        self.eps_abs,self.eps_rel,self.max_iter=float(eps_abs),float(eps_rel),int(max_iter)
        # OSQP may print an informational no-active-set message even with
        # verbose=False.  Default off matches the supplied reference solver;
        # residual validation and the secondary method remain mandatory.
        self.polishing=bool(polishing)
        self.primal_tolerance,self.dual_tolerance=float(primal_tolerance),float(dual_tolerance)
        # Sequential projection can propagate one per-row increment residual
        # through N command entries.  Physical applied bounds are still exact.
        self.input_correction_budget_v=(self.horizon*self.primal_tolerance
            +16*np.finfo(float).eps*max(abs(self.u_min),abs(self.u_max)))
        self.nonlinear_tolerance=float(nonlinear_tolerance)
        self.bootstrap_error_mm=float(bootstrap_error_mm)
        # Extra inward QP margin addresses curvature at an active LTV bound.
        # Nonlinear acceptance still uses the original requested pressure cap.
        self.linearization_reserve_mpa=float(linearization_reserve_mpa)
        self.pressure_lower=self._pressure_pair(pressure_lower,default=(None,None))
        default_upper=(self.p_safe if constrain_p1 else None,self.p_safe)
        self.pressure_upper=self._pressure_pair(pressure_upper,default=default_upper)
        self.pressure_bounds=[]
        if self.enable_pressure_constraint:
            for chamber,(lower,upper) in enumerate(zip(self.pressure_lower,self.pressure_upper)):
                upper=None if upper is None else upper-self.pressure_margin
                if lower is not None and upper is not None and lower>=upper:
                    raise ValueError('Pressure margin makes the bound empty')
                if lower is not None or upper is not None:
                    self.pressure_bounds.append((2+chamber,lower,upper))
        self._sum=np.tril(np.ones((self.horizon,self.horizon)))
        self.reset()

    @staticmethod
    def _pressure_pair(value,default):
        if value is None:
            return tuple(default)
        if np.isscalar(value):
            value=(value,value)
        if len(value)!=2:
            raise ValueError('Pressure limits require (P1, P2), in MPa')
        result=tuple(None if v is None else float(v) for v in value)
        if any(v is not None and (not math.isfinite(v) or v<0.) for v in result):
            raise ValueError('Pressure limits must be finite and nonnegative')
        return result

    def reset(self):
        self.nominal_sequence=None
        self.last={}
        self._solver=None
        self._previous_applied=None
        self._last_du=None

    def _reference(self,reference):
        r=np.asarray(reference,dtype=float)
        if r.ndim==0:
            r=np.tile([float(r),0.],(self.horizon,1))
        elif r.ndim==1:
            if len(r) in (2,5,6):
                r=np.tile(r[:2],(self.horizon,1))
            elif len(r)==self.horizon:
                r=np.column_stack([r,np.zeros(self.horizon)])
            else:
                raise ValueError('Reference needs one constant state or N future outputs')
        elif r.ndim==2 and r.shape[0]==self.horizon and r.shape[1] in (1,2,5,6):
            r=np.column_stack([r[:,0],np.zeros(self.horizon)]) if r.shape[1]==1 else r[:,:2]
        else:
            raise ValueError('Reference shape must be (N,1), (N,2), (N,5), or (N,6)')
        if not np.all(np.isfinite(r)):
            raise ValueError('Reference must be finite')
        return r

    def _project_inputs(self,sequence,u_previous):
        result=[];previous=float(u_previous)
        for value in sequence:
            value=float(np.clip(value,max(self.u_min,previous-self.delta_u_max),
                                min(self.u_max,previous+self.delta_u_max)))
            result.append(value);previous=value
        return np.asarray(result)

    def _warm_plan(self,z,u_previous,reference):
        if self.nominal_sequence is None or (self._previous_applied is not None and
                abs(self._previous_applied-u_previous)>self.primal_tolerance):
            warm=np.full(self.horizon,u_previous)
        else:
            warm=np.r_[self.nominal_sequence[1:],self.nominal_sequence[-1]]
        warm=self._project_inputs(warm,u_previous)
        nominal=warm.copy()
        parameters=getattr(self.model,'parameters',{})
        deadzone=10*parameters.get('deadzone',.05)
        error=float(reference[0,0]-z[0])
        if np.max(abs(nominal))<=deadzone+1e-8 and abs(error)>self.bootstrap_error_mm:
            # A local deadzone Jacobian at zero cannot predict breakaway.
            # Seed a declared active branch; the optimizer still enforces all
            # bounds, and the candidate is checked with the nonlinear model.
            nominal=self._project_inputs(np.full(self.horizon,
                self.u_max if error>0 else self.u_min),u_previous)
        return nominal,warm

    def _rollout(self,z,sequence):
        predicted=np.asarray(self.model.rollout(z.copy(),sequence.copy(),self.dt),dtype=float)
        if predicted.shape!=(self.horizon,self.nx_physical) or not np.all(np.isfinite(predicted)):
            raise MPCSolveError('Nonlinear rollout returned invalid future state dimensions or values')
        return predicted

    def _prediction(self,z,nominal):
        predicted=self._rollout(z,nominal)
        starts=np.vstack([z,predicted[:-1]])
        models=[augment_incremental_model(*self.model.linearize_discrete(state,float(u),self.dt))
                for state,u in zip(starts,nominal)]
        f,phi,b=condense_affine(models)
        if f.shape!=(self.horizon*6,6):
            raise MPCSolveError('The grey-box model must contain five physical states')
        return f,phi,b

    def _qp(self,z,u_previous,reference,nominal):
        f,phi,b=self._prediction(z,nominal)
        initial=np.r_[z,u_previous]
        free=f@initial+b
        track=np.array([6*i+j for i in range(self.horizon) for j in (0,1)])
        response=phi[track];offset=free[track]-reference.ravel()
        q=np.tile([self.q_position,self.q_velocity],self.horizon)/self.objective_scale
        h=2*(response.T@(q[:,None]*response)+self.move_weight/self.objective_scale*np.eye(self.horizon))
        h=(h+h.T)/2
        g=2*response.T@(q*offset)
        matrices=[np.eye(self.horizon),self._sum]
        lower=[np.full(self.horizon,-self.delta_u_max),np.full(self.horizon,self.u_min-u_previous)]
        upper=[np.full(self.horizon,self.delta_u_max),np.full(self.horizon,self.u_max-u_previous)]
        # Pressure rows are time-major, with absolute measured pressure and
        # the affine offset retained in the free response.
        if self.pressure_bounds:
            rows=[];lo=[];hi=[]
            for i in range(self.horizon):
                for state,minimum,maximum in self.pressure_bounds:
                    row=6*i+state
                    rows.append(row)
                    lo.append(-np.inf if minimum is None else minimum+self.linearization_reserve_mpa-free[row])
                    hi.append(np.inf if maximum is None else maximum-self.linearization_reserve_mpa-free[row])
            matrices.append(phi[rows]);lower.append(np.asarray(lo));upper.append(np.asarray(hi))
        return h,g,np.vstack(matrices),np.concatenate(lower),np.concatenate(upper),free,phi

    def _solve_qp(self,h,g,a,lower,upper,warm):
        p_csc=_dense_csc(h,upper=True);a_csc=_dense_csc(a)
        polish=self.polishing
        if self._solver is None:
            self._solver=osqp.OSQP()
            self._solver.setup(P=p_csc,q=g,A=a_csc,l=lower,u=upper,verbose=False,
                warm_starting=True,polishing=polish,eps_abs=self.eps_abs,eps_rel=self.eps_rel,
                max_iter=self.max_iter,adaptive_rho=True,adaptive_rho_interval=25,check_termination=25)
        else:
            self._solver.update(Px=p_csc.data,q=g,Ax=a_csc.data,l=lower,u=upper)
            self._solver.update_settings(polishing=polish)
        self._solver.warm_start(x=warm)
        result=self._solver.solve(raise_error=False)
        status=str(result.info.status)
        diagnostics=dict(osqp_status=status,status_value=int(result.info.status_val),
                         osqp_iterations=int(result.info.iter),
                         primal_residual=float(result.info.prim_res),
                         dual_residual=float(result.info.dual_res),
                         solver_time_ms=1000*float(result.info.solve_time))
        reason=None
        if result.info.status_val not in (1,2) or result.x is None or result.y is None or not (
                np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.y))):
            reason=f'OSQP rejected: {status}'
        else:
            decision=np.asarray(result.x)
            mapped=a@decision
            violation=float(max(0.,np.max(lower-mapped),np.max(mapped-upper)))
            stationarity=float(np.max(abs(h@decision+g+a.T@np.asarray(result.y))))
            diagnostics.update(linear_constraint_violation=violation,kkt_stationarity=stationarity)
            if violation>self.primal_tolerance or stationarity>self.dual_tolerance*(1+np.max(abs(g))):
                reason='OSQP failed independent residual checks'
            else:
                diagnostics['qp_solver']='osqp'
                diagnostics['fallback_solver_used']=False
                return decision,diagnostics
        # A second numerical method solves exactly the same QP.  The failed
        # OSQP status and residuals remain recorded, never relabelled solved.
        diagnostics['osqp_failure_reason']=reason
        return self._solve_clarabel(h,g,a,lower,upper,diagnostics)

    def _solve_clarabel(self,h,g,a,lower,upper,primary):
        start=time.perf_counter()
        try:
            import clarabel
            has_upper=np.flatnonzero(np.isfinite(upper))
            has_lower=np.flatnonzero(np.isfinite(lower))
            inequalities=np.vstack([a[has_upper],-a[has_lower]])
            limits=np.r_[upper[has_upper],-lower[has_lower]]
            settings=clarabel.DefaultSettings()
            settings.verbose=False;settings.max_iter=200;settings.max_threads=1
            settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
            solver=clarabel.DefaultSolver(_dense_csc(h,upper=True),g,
                sparse.csc_matrix(inequalities),limits,
                [clarabel.NonnegativeConeT(len(limits))],settings)
            solution=solver.solve()
            fallback_status=str(solution.status)
            info=dict(primary,qp_solver='clarabel',fallback_solver_used=True,
                      fallback_solver_status=fallback_status,
                      fallback_iterations=int(solution.iterations),
                      fallback_solver_time_ms=1000*float(solution.solve_time))
            info['solver_time_ms']+=info['fallback_solver_time_ms']
            if fallback_status not in ('Solved','AlmostSolved'):
                raise MPCSolveError(f'Both QP methods rejected: OSQP={primary["osqp_status"]}, Clarabel={fallback_status}',info)
            decision=np.asarray(solution.x);cone_dual=np.asarray(solution.z)
            if not np.all(np.isfinite(decision)) or not np.all(np.isfinite(cone_dual)):
                raise MPCSolveError('Clarabel returned nonfinite primal/dual values',info)
            multiplier=np.zeros(len(lower))
            multiplier[has_upper]+=cone_dual[:len(has_upper)]
            multiplier[has_lower]-=cone_dual[len(has_upper):]
            mapped=a@decision
            violation=float(max(0.,np.max(lower-mapped),np.max(mapped-upper)))
            stationarity=float(np.max(abs(h@decision+g+a.T@multiplier)))
            slack=limits-inequalities@decision
            complementarity=float(np.max(abs(slack*cone_dual)))
            info.update(linear_constraint_violation=violation,kkt_stationarity=stationarity,
                        fallback_complementarity=complementarity,
                        fallback_primal_residual=float(solution.r_prim),
                        fallback_dual_residual=float(solution.r_dual))
            if (violation>self.primal_tolerance or np.min(cone_dual)<-self.dual_tolerance
                    or stationarity>self.dual_tolerance*(1+np.max(abs(g)))):
                raise MPCSolveError('Clarabel failed independent primal/dual residual checks',info)
            return decision,info
        except MPCSolveError:
            raise
        except Exception as exc:
            info=dict(primary,fallback_solver_used=True,qp_solver='clarabel',
                      fallback_solver_error=str(exc),
                      fallback_solver_time_ms=1000*(time.perf_counter()-start))
            raise MPCSolveError(f'QP fallback failed: {exc}',info) from exc

    def _nonlinear_validation(self,z,sequence,u_previous):
        predicted=self._rollout(z,sequence)
        delta=np.diff(np.r_[u_previous,sequence])
        input_violation=float(max(0.,np.max(sequence-self.u_max),np.max(self.u_min-sequence),
                                  np.max(abs(delta)-self.delta_u_max)))
        pressure_violation=0.
        for index,minimum,maximum in self.pressure_bounds:
            if minimum is not None:
                pressure_violation=max(pressure_violation,float(np.max(minimum-predicted[:,index])))
            if maximum is not None:
                pressure_violation=max(pressure_violation,float(np.max(predicted[:,index]-maximum)))
        valid=(input_violation<=self.primal_tolerance
               and pressure_violation<=self.nonlinear_tolerance)
        return valid,predicted,max(0.,pressure_violation),input_violation

    def solve(self,z,u_prev,ref_traj):
        start=time.perf_counter()
        z=np.asarray(z,dtype=float).reshape(-1);u_prev=float(u_prev)
        self.last=dict(status='started',runtime_ms=0.)
        if (z.shape!=(5,) or not np.all(np.isfinite(z)) or not math.isfinite(u_prev)
                or min(z[2:4])<=0. or not self.u_min<=u_prev<=self.u_max):
            raise ValueError('Expected finite five-state feedback, positive MPa pressures, and bounded previous voltage')
        reference=self._reference(ref_traj)
        nominal,shifted=self._warm_plan(z,u_prev,reference)
        best=None;diagnostics=[];last_violation=None;raw_first=None;forced_shift=False
        try:
            for iteration in range(self.sequential_iterations):
                h,g,a,lower,upper,_,_=self._qp(z,u_prev,reference,nominal)
                warm=np.diff(np.r_[u_prev,nominal])
                try:
                    decision,info=self._solve_qp(h,g,a,lower,upper,warm)
                except MPCSolveError as exc:
                    # Solver failures are not successful QPs.  A previously
                    # accepted plan may still be usable, but only after new
                    # feedback and a fresh nonlinear constraint check.
                    if self.nominal_sequence is None:
                        raise
                    feasible,predicted,violation,input_violation=self._nonlinear_validation(z,shifted,u_prev)
                    if not feasible:
                        raise
                    info=dict(exc.diagnostics or {},solver_failure=str(exc),
                              nonlinear_pressure_violation_mpa=violation,
                              input_numerical_correction_v=0.,
                              candidate_voltage_v=float(shifted[0]))
                    info.setdefault('osqp_status','failed')
                    info.setdefault('osqp_iterations',0)
                    info.setdefault('solver_time_ms',0.)
                    info.setdefault('linear_constraint_violation',None)
                    info.setdefault('kkt_stationarity',None)
                    diagnostics.append(info)
                    errors=predicted[:,:2]-reference;du=np.diff(np.r_[u_prev,shifted])
                    cost=float(np.sum(errors**2*np.array([self.q_position,self.q_velocity]))
                               +self.move_weight*np.dot(du,du))
                    best=(cost,shifted,predicted,info,violation,input_violation)
                    forced_shift=True
                    break
                candidate=u_prev+self._sum@decision
                raw_first=float(candidate[0])
                applied=self._project_inputs(candidate,u_prev)
                correction=float(np.max(abs(candidate-applied)))
                if correction>self.input_correction_budget_v:
                    raise MPCSolveError('Input bound correction exceeded the accumulated numerical budget',
                        dict(**info,input_numerical_correction_v=correction,
                             input_correction_budget_v=self.input_correction_budget_v,
                             candidate_sequence_v=candidate.tolist(),previous_voltage_v=u_prev))
                feasible,predicted,violation,input_violation=self._nonlinear_validation(z,applied,u_prev)
                last_violation=violation
                info.update(nonlinear_pressure_violation_mpa=violation,
                            input_numerical_correction_v=correction,
                            input_correction_budget_v=self.input_correction_budget_v,
                            candidate_voltage_v=raw_first)
                diagnostics.append(info)
                if feasible:
                    errors=predicted[:,:2]-reference
                    du=np.diff(np.r_[u_prev,applied])
                    cost=float(np.sum(errors**2*np.array([self.q_position,self.q_velocity]))
                               +self.move_weight*np.dot(du,du))
                    if best is None or cost<best[0]:
                        best=(cost,applied,predicted,info,violation,input_violation)
                nominal=applied
                if feasible and iteration+1>=self.min_iterations:
                    break
            fallback=forced_shift
            if best is None:
                # This is a model-validated retained plan, never an assertion
                # that simply holding u_prev is physically safe.
                if self.nominal_sequence is not None:
                    feasible,predicted,violation,input_violation=self._nonlinear_validation(z,shifted,u_prev)
                    if feasible:
                        errors=predicted[:,:2]-reference;du=np.diff(np.r_[u_prev,shifted])
                        cost=float(np.sum(errors**2*np.array([self.q_position,self.q_velocity]))
                                   +self.move_weight*np.dot(du,du))
                        best=(cost,shifted,predicted,diagnostics[-1],violation,input_violation)
                        fallback=True
                if best is None:
                    raise MPCSolveError(f'No nonlinear-feasible pressure plan after {len(diagnostics)} local QPs; violation={last_violation} MPa')
            cost,sequence,predicted,chosen,violation,input_violation=best
            self.nominal_sequence=sequence.copy()
            self._previous_applied=float(sequence[0])
            self._last_du=np.diff(np.r_[u_prev,sequence])
            self.last=dict(status='validated_shifted_plan' if fallback else 'solved',
                osqp_status=chosen['osqp_status'],iterations=sum(d['osqp_iterations'] for d in diagnostics),
                qp_solver=chosen.get('qp_solver','osqp'),
                fallback_solver_used=chosen.get('fallback_solver_used',False),
                fallback_solver_status=chosen.get('fallback_solver_status'),
                fallback_iterations=chosen.get('fallback_iterations',0),
                solver_failure=chosen.get('solver_failure'),
                primal_tolerance=self.primal_tolerance,nonlinear_tolerance_mpa=self.nonlinear_tolerance,
                linearization_reserve_mpa=self.linearization_reserve_mpa,
                polishing=self.polishing,
                sequential_iterations=len(diagnostics),runtime_ms=1000*(time.perf_counter()-start),
                solver_time_ms=sum(d['solver_time_ms'] for d in diagnostics),
                prediction_violation=violation,prediction_violation_mpa=violation,
                linear_constraint_violation=chosen['linear_constraint_violation'],
                kkt_stationarity=chosen['kkt_stationarity'],input_violation_v=input_violation,
                actual_du=float(sequence[0]-u_prev),actual_du_v=float(sequence[0]-u_prev),
                candidate_voltage_v=float(sequence[0]) if fallback else chosen['candidate_voltage_v'],
                last_attempt_candidate_voltage_v=raw_first,voltage_v=float(sequence[0]),
                input_numerical_correction_v=0. if fallback else chosen['input_numerical_correction_v'],
                input_correction_budget_v=self.input_correction_budget_v,
                command_sequence_v=sequence.tolist(),prediction=predicted.tolist(),
                predicted_peak_p1_mpa=float(np.max(predicted[:,2])),
                predicted_peak_p2_mpa=float(np.max(predicted[:,3])),
                nonlinear_cost=cost,feasible_shifted_plan_used=fallback,
                pressure_constraints=[dict(state=index,lower=lo,upper=hi)
                                      for index,lo,hi in self.pressure_bounds],
                qp_diagnostics=diagnostics)
            return float(sequence[0])
        except Exception as exc:
            self.last.update(status='failed',runtime_ms=1000*(time.perf_counter()-start),
                             error=str(exc),prediction_violation=last_violation,
                             qp_diagnostics=diagnostics)
            if isinstance(exc,MPCSolveError) and exc.diagnostics is not None:
                self.last.update(exc.diagnostics)
                self.last['status']='failed'
            if isinstance(exc,MPCSolveError):
                raise
            raise MPCSolveError(f'MPC model or solver failed: {exc}') from exc

    def update(self,z,u_prev,ref_traj):
        """Alias for simulation loops using the chapter-six controller spelling."""
        return self.solve(z,u_prev,ref_traj)
