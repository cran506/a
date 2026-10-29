"""独立核验物理方程、采样护栏、量化及C部署接口，不拟合教材数字。"""
import ctypes
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from scipy.integrate import solve_ivp
import drl_hydraulic_lab as lab


class PhysicsTests(unittest.TestCase):
    def test_equilibrium_and_force_balance(self):
        plant=lab.HydraulicTwin()
        for load in [0.,2500.,5000.]:
            y,u=plant.equilibrium(load=load)
            derivative=plant.rhs(y,u,load)
            self.assertLess(np.max(abs(derivative[1:])),.02)
            self.assertAlmostEqual(y[1],.05)
            self.assertTrue(.5<u<10)

    def test_load_beta_mass_and_friction_have_physical_effect(self):
        twin=lab.HydraulicTwin();y,u=twin.equilibrium()
        a=twin.rhs(y,u);b=twin.rhs(y,u,5000.)
        self.assertAlmostEqual(b[1]-a[1],-5000/twin.physics.mass)
        baseline=twin.rhs(y,1.2);soft=twin.rhs(y,1.2,beta_ratio=.5)
        np.testing.assert_allclose(soft[2:],baseline[2:]*.5)
        self.assertAlmostEqual(soft[1],baseline[1])
        heavier=lab.HydraulicTwin(replace(lab.Physics(),mass=600.))
        self.assertAlmostEqual(heavier.rhs(y,u,5000.)[1]/b[1],500/600)
        rougher=lab.HydraulicTwin(replace(lab.Physics(),Fc=650.))
        self.assertLess(rougher.rhs(y,u)[1],a[1])

    def test_rk4_against_independent_radau_through_fault(self):
        twin=lab.HydraulicTwin();state,u=twin.equilibrium();twin.reset(state);reference=state.copy()
        for i in range(80):
            load=5000. if i>=40 else 0.;beta=.5 if i>=40 else 1.
            control=u+.02*np.sin(i*.13)
            actual,_=twin.step(control,load=load,beta_ratio=beta)
            sol=solve_ivp(lambda t,y:twin.rhs(y,control,load,beta),(0.,.001),reference,
                          method='Radau',rtol=1e-10,atol=[1e-12,1e-12,1e-3,1e-3])
            self.assertTrue(sol.success);reference=sol.y[:,-1]
        self.assertTrue(np.all(abs(actual-reference)<=np.array([1e-8,1e-6,20.,20.])+2e-5*abs(reference)),str(actual-reference))

    def test_third_order_lie_derivative_against_directional_difference(self):
        plant=lab.HydraulicTwin();y,u=plant.equilibrium();u+=.04
        f=plant.rhs(y,u);p=plant.physics
        jerk=(p.A1*f[2]-p.A2*f[3]-lab.friction_gradient(y[1],plant.p)*f[1])/p.mass
        finite=(plant.rhs(y+1e-7*f,u)[1]-f[1])/1e-7
        self.assertAlmostEqual(jerk,finite,places=3)


class SafetyTests(unittest.TestCase):
    def test_safe_trajectory_and_unprotected_failure_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            result=lab.safety_experiment(folder)
            self.assertTrue(result['empirical_node_safety_pass'])
            self.assertAlmostEqual(result['protected_duration_s'],1.)
            self.assertLessEqual(result['observed_substep_max_x_mm'],100.)
            self.assertIsNotNone(result['unprotected_other_failure'])
            self.assertGreater(result['minimum_safe_voltage_v'],-10.)

    def test_initial_unsafe_state_rejected(self):
        twin=lab.HydraulicTwin();twin.state[0]=.101
        with self.assertRaises(lab.InfeasibleSafety):lab.HOCBFFilter().project(twin,0.)

    def test_transparency_for_equilibrium_and_voltage_limits(self):
        twin=lab.HydraulicTwin();y,u=twin.equilibrium();twin.reset(y);filter_=lab.HOCBFFilter()
        self.assertAlmostEqual(filter_.project(twin,u),u,places=8)
        for action in [-10.,-3.,0.,3.,10.]:
            applied=filter_.project(twin,action)
            self.assertLessEqual(abs(applied),10.)


class LearningInterfaceTests(unittest.TestCase):
    @unittest.skipIf(lab.gym is None,'Gymnasium not installed')
    def test_randomization_reproducible_and_reward_exact(self):
        env=lab.HydraulicGymEnv();obs,info=env.reset(seed=22)
        p=info['physics'];self.assertTrue(400<=p['mass']<=600)
        self.assertTrue(.6e9<=p['beta']<=1.4e9);self.assertTrue(350<=p['Fc']<=650)
        obs2,info2=env.reset(seed=22);np.testing.assert_array_equal(obs,obs2)
        self.assertEqual(p,info2['physics']);old=env.last_u
        _,reward,_,_,item=env.step(np.array([.09],np.float32))
        expected=-100*item['error_mm']**2-.1*item['applied_v']**2-(item['applied_v']-old)**2
        expected+=2. if abs(item['error_mm'])<.2 else 0.
        self.assertAlmostEqual(expected,item['base_reward'])
        self.assertAlmostEqual(reward,.01*(expected+item['terminal_penalty']))
        env.close()


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(23);self.inputs=rng.normal(0,.5,(300,5)).astype(np.float32)
        arrays={}
        for i,(no,ni) in enumerate([(64,5),(64,64),(1,64)]):
            arrays[f'W{i}']=rng.normal(0,.1,(no,ni)).astype(np.float32)
            arrays[f'b{i}']=rng.normal(0,.01,no).astype(np.float32)
        self.actor=lab.FloatActor(arrays)

    def test_affine_zero_and_error_bound(self):
        values=np.array([-.7,-.2,0.,.1,.6],np.float32)
        scale,zero=lab.qparams(values);q=lab.quantize(values,scale,zero)
        self.assertEqual(q.dtype,np.int8);self.assertEqual(q[2],zero)
        restored=(q.astype(np.int32)-zero)*scale
        self.assertLessEqual(np.max(abs(values-restored)),scale/2+1e-7)

    def test_structural_pruning_and_saved_integer_inference(self):
        pruned,edge,report=lab.compress_actor(self.actor,self.inputs)
        self.assertEqual(pruned.weights[0].shape,(19,5))
        self.assertEqual(pruned.weights[1].shape,(19,19))
        self.assertEqual(report['weight_connections_after'],475)
        self.assertTrue(np.isfinite(edge(self.inputs)).all())
        self.assertLessEqual(np.max(abs(edge(self.inputs))),1.)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'edge.npz';np.savez(target,**edge.data)
            with np.load(target) as data:loaded=lab.Int8Actor(data)
            np.testing.assert_array_equal(edge(self.inputs),loaded(self.inputs))
        for i in range(3):
            self.assertEqual(edge.data[f'QW{i}'].dtype,np.int8)
            self.assertEqual(edge.data[f'qb{i}'].dtype,np.int32)

    @unittest.skipUnless(shutil.which('gcc'),'gcc not installed')
    def test_generated_static_c_matches_numpy_integer_policy(self):
        _,edge,_=lab.compress_actor(self.actor,self.inputs)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'policy.c';library=Path(folder)/'policy.so'
            lab.export_c(edge,source)
            subprocess.run(['gcc','-O2','-std=c99','-shared','-fPIC',str(source),'-lm','-o',str(library)],check=True,capture_output=True)
            dll=ctypes.CDLL(str(library));dll.hydraulic_actor.restype=ctypes.c_float
            dll.hydraulic_actor.argtypes=[ctypes.POINTER(ctypes.c_float)]
            for obs in self.inputs[:30]:
                got=dll.hydraulic_actor(obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                self.assertAlmostEqual(got,float(edge(obs).item()),delta=2e-5)


if __name__=='__main__':unittest.main(verbosity=2)
