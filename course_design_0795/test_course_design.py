import unittest
from dataclasses import replace
import numpy as np
from scipy.signal import sosfilt,sosfilt_zi
from filtering import CascadeFilter,causal_median,sos_recursion
from physics import Parameters,HydraulicTwin,rhs_effective,advance_effective,friction,effective_voltage,inverse_deadzone,trajectory
from controllers import IncrementalPID,LTVMPC,linearize,prediction_matrices
from experiments import calculate_metrics


class FilterTests(unittest.TestCase):
    def test_constant_preserved(self):
        x=np.tile([.03,3e6,6e6],(100,1));np.testing.assert_allclose(CascadeFilter()(x),x,rtol=1e-12)
    def test_causality(self):
        rng=np.random.default_rng(795);x=rng.normal(size=(100,3));changed=x.copy();changed[70:]+=1e6
        np.testing.assert_array_equal(CascadeFilter()(x)[:70],CascadeFilter()(changed)[:70])
    def test_recursive_butterworth_against_scipy(self):
        rng=np.random.default_rng(795);x=rng.normal(size=(200,3));f=CascadeFilter();med=causal_median(x,5)
        expected=np.column_stack([sosfilt(f.sos,med[:,i],zi=sosfilt_zi(f.sos)*med[0,i])[0] for i in range(3)])
        np.testing.assert_allclose(sos_recursion(med,f.sos),expected,atol=1e-12)


class PhysicsTests(unittest.TestCase):
    def test_four_states_and_force(self):
        p=Parameters().vector();y=np.array([.08,.03,3e6,5e6]);a=rhs_effective(y,.6,0,p);b=rhs_effective(y,.6,5000,p)
        self.assertEqual(a.shape,(4,));self.assertAlmostEqual(a[1]-b[1],5000/p[5])
    def test_deadzone_and_inverse(self):
        for u in [-.5,-.1,0,.1,.5]:self.assertEqual(effective_voltage(u),0)
        for w in [-3.,-.2,.2,3.]:self.assertAlmostEqual(effective_voltage(inverse_deadzone(w)),w)
    def test_friction_odd_and_leakage_sign(self):
        p=Parameters().vector();self.assertAlmostEqual(friction(.03,p),-friction(-.03,p))
        y=np.array([.1,0.,5e6,3e6]);a=rhs_effective(y,0.,0,p)
        p2=replace(Parameters(),Ct=Parameters().Ct*2).vector();b=rhs_effective(y,0.,0,p2)
        self.assertLess(b[2],a[2]);self.assertGreater(b[3],a[3])
    def test_stationary_balance_and_moving_equilibrium(self):
        plant=HydraulicTwin();y,u=plant.equilibrium(.6)
        f=rhs_effective(y,effective_voltage(u),0.,plant.p)
        np.testing.assert_allclose(f[1:],0.,atol=.01)
        y0=np.array([.04,0.,plant.p[7],plant.p[7]])
        np.testing.assert_allclose(rhs_effective(y0,0,0,plant.p),0.)
    def test_substep_convergence(self):
        p=Parameters().vector();a=np.array([.08,0.,1e5,1e5]);b=a.copy()
        for i in range(60):
            w=.3*np.sin(i*.2);a,_=advance_effective(a,w,0.,p,.001,64);b,_=advance_effective(b,w,0.,p,.001,128)
        self.assertTrue(np.all(abs(a-b)<np.array([1e-8,1e-6,100.,100.])),str(a-b))
    def test_reference_boundary_conditions(self):
        r=trajectory(np.array([0.,.8,1.2]));np.testing.assert_allclose(r[:,0],[.04,.12,.12],atol=1e-12)
        np.testing.assert_allclose(r[:,1:3],0.,atol=1e-12)


class ControlTests(unittest.TestCase):
    def test_incremental_equivalence_before_saturation(self):
        pid=IncrementalPID(kp=250,ki=150,kd=.2);integral=0.;prev=0.;d=0.
        for e in np.linspace(0.,.001,30):
            integral+=150*e*.001;d=.85*d+.15*.2*(e-prev)/.001
            expected=250*e+integral+d
            self.assertAlmostEqual(pid.update([e,0,0,0],[0,0,0,0]),expected,places=12);prev=e
    def test_saturation_output_memory(self):
        pid=IncrementalPID(kp=250,ki=2000,kd=0)
        for _ in range(250):self.assertEqual(pid.update([.05,0,0,0],[0,0,0,0]),10.)
        self.assertLess(pid.update([-.001,0,0,0],[0,0,0,0]),0.)
        self.assertEqual(pid.u0,pid.last['requested_v'])
    def test_affine_constant_term_preserves_equilibrium(self):
        p=Parameters().vector();y=np.array([.08,0.,1e5,1e5]);a,b,c=linearize(y,0.,0.,p)
        z=y*np.array([1000,1,1e-6,1e-6]);np.testing.assert_allclose(a@z+c,z,atol=1e-9)
    def test_prediction_matrix_direct_recursion(self):
        a=np.eye(4)*.8;b=np.array([.1,.2,.3,.4]);c=np.arange(4)*.03;z=np.arange(4)*.2;controls=np.linspace(-1,1,8)
        f,g,o=prediction_matrices(a,b,c,8);direct=[];state=z.copy()
        for u in controls:state=a@state+b*u+c;direct.extend(state)
        np.testing.assert_allclose(f@z+g@controls+o,direct,atol=1e-12)
    def test_qp_pressure_constraints_and_input_mapping(self):
        model=LTVMPC(Parameters(),horizon=8);y=np.array([.04,0.,1e5,1e5]);u=model.update(trajectory(0.),y,0.)
        self.assertLessEqual(abs(u),10.)
        m=model.last_matrices;n=8;self.assertEqual(m['A'].shape,(5*n,n))
        actual=m['A']@m['solution'];self.assertLessEqual(np.max(actual-m['upper']),2e-3)
        self.assertLessEqual(np.max(m['l']-actual),2e-3)
        np.testing.assert_allclose(m['A'][-2*n:],m['Phi'][[4*i+j for i in range(n) for j in [2,3]]])


class MetricTests(unittest.TestCase):
    def test_metric_definitions(self):
        rows=[]
        for t,e,u in zip([.4,.5,.6],[1.,-2.,3.],[1.,-2.,2.]):
            rows.append(dict(time_s=t,error_mm=e,u_v=u,p1_max_mpa=10.,p2_max_mpa=12.,p1_min_mpa=.1,p2_min_mpa=.1,saturation=0))
        result=calculate_metrics(rows)
        self.assertAlmostEqual(result['rmse_mm'],np.sqrt(14/3));self.assertEqual(result['disturbance_mae_mm'],3.)
        self.assertEqual(result['TV_V'],8.);self.assertEqual(result['p_peak_mpa'],12.)


if __name__=='__main__':unittest.main(verbosity=2)
