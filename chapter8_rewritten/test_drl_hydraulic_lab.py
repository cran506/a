"""第八章重写版独立回归测试：物理、Gym、HOCBF 与整数部署。

运行：python -m unittest discover -s . -p test_ch8_rewritten.py -v
可设 CH8_REWRITTEN_DIR 指向包含 drl_hydraulic_lab.py 和 assets/ 的交付目录。
不训练网络，不重写模型、原始训练日志或实验报告。
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE if (HERE/'drl_hydraulic_lab.py').exists() else Path('/workspace/a/chapter8_rewritten')
ROOT = Path(os.environ.get('CH8_REWRITTEN_DIR', str(DEFAULT_ROOT)))
spec = importlib.util.spec_from_file_location('chapter8_rewritten_under_test', ROOT/'drl_hydraulic_lab.py')
lab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lab
spec.loader.exec_module(lab)

# 在先前用户选定的冻结模型上生成；独立保存输入/输出以检查重写后的整数算子。
# 输入已经过课堂五维 Z-score，覆盖空载、故障触发、瞬态及重载稳态。
FIXTURE_OBS = np.array([
    [0., 0., 1.818989489943217e-15, -.8003289699554443, 0.],
    [4.727077794086654e-6, -8.741980082049849e-7, .00029927969444543123, -.8003166913986206, 0.],
    [4.73463751404779e-6, 4.657022600440541e-6, -39.999244689941406, -.8002976775169373, 0.],
    [-.0007570399902760983, -.261829137802124, -14.31003189086914, .24308279156684875, 0.],
    [-.002286806469783187, -.3275283873081207, -.7374233603477478, .7979252934455872, 0.],
    [-.00986938364803791, -.04769944027066231, 2.601865530014038, .9490389227867126, 0.],
    [-.011427167803049088, -.005321276839822531, .09081151336431503, .8478793501853943, 0.],
    [-.007646962534636259, .002255568979308009, -.0024179888423532248, .8444197773933411, 0.],
    [-.0066439262591302395, 8.801568765193224e-5, -9.524823690298945e-5, .8444083333015442, 0.],
], dtype=np.float32)
FIXTURE_INT8 = np.array([.3457437753677368, .34575068950653076, .9979103207588196,
    .9684029221534729, .8766058087348938, .7813452482223511, .6926676034927368,
    .6680147647857666, .6663572192192078], dtype=np.float32).reshape(-1, 1)
FIXTURE_FP32 = np.array([.3510074019432068, .3509936034679413, .9969027638435364,
    .9897950887680054, .9474409222602844, .8017441034317017, .6946938037872314,
    .6691369414329529, .6678094267845154], dtype=np.float32).reshape(-1, 1)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.twin = lab.HydraulicTwinCore()

    def test_explicit_parameters_override_config(self):
        twin = lab.HydraulicTwinCore(m=280., config={'m':230., 'F_c':1500.})
        self.assertEqual(twin.p['m'], 280.)
        self.assertEqual(twin.p['F_c'], 1500.)

    def test_default_parameter_copy_is_independent(self):
        p = lab.default_config()
        p['m'] = -1.
        p['state_units'][0] = 'damaged'
        self.assertGreater(lab.default_config()['m'], 0.)
        self.assertEqual(lab.default_config()['state_units'][0], 'm')

    def test_invalid_parameter_update_rolls_back(self):
        original = self.twin.parameters
        for invalid in ({'m':-1.}, {'beta_e':0.}, {'F_c':float('nan')}, {'unknown':1.}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.twin.update_physics(invalid)
                self.assertEqual(self.twin.parameters, original)
        # Native parameter vector must also remain consistent after failed update.
        control = lab.HydraulicTwinCore()
        np.testing.assert_array_equal(self.twin.step(4.)[0], control.step(4.)[0])

    def test_invalid_actions_do_not_commit_state_or_clock(self):
        state, time = self.twin.state.copy(), self.twin.t
        for args in [(11., {}), (float('nan'), {}), (4., {'dt':0}),
                     (4., {'beta_multiplier':-1}), (4., {'substeps':1.5})]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    self.twin.step(args[0], **args[1])
                np.testing.assert_array_equal(self.twin.state, state)
                self.assertEqual(self.twin.t, time)

    def test_mass_changes_force_acceleration(self):
        state = self.twin.state.copy()
        a = self.twin.rhs(state, 4., load_n=1000.)[1]
        self.twin.update_physics({'m':300.})
        self.assertAlmostEqual(self.twin.rhs(state, 4., load_n=1000.)[1], a*250./300., places=13)

    def test_friction_randomization_is_in_force_equation(self):
        state = self.twin.state.copy()
        original = self.twin.rhs(state, 4.)
        self.twin.update_physics({'F_c':1560.})
        changed = self.twin.rhs(state, 4.)
        self.assertLess(changed[1], original[1]-1.)
        np.testing.assert_array_equal(changed[2:], original[2:])

    def test_bulk_modulus_changes_pressure_continuity(self):
        state = self.twin.state.copy()
        original = self.twin.rhs(state, 4.)
        self.twin.update_physics({'beta_e':500e6})
        changed = self.twin.rhs(state, 4.)
        self.assertAlmostEqual(changed[1], original[1], places=14)
        self.assertTrue(np.all(np.abs(changed[2:]) < np.abs(original[2:])))
        self.assertTrue(np.all(np.abs(changed[2:]) > .49*np.abs(original[2:])))

    def test_equilibrium_solves_same_rhs(self):
        for velocity, load in ((.05, 0.), (.05, 5000.), (0., 0.)):
            with self.subTest(velocity=velocity, load=load):
                state, u = self.twin.equilibrium_for_velocity(velocity, load)
                rhs = self.twin.rhs(state, u, load)
                np.testing.assert_allclose(rhs[1:], [0., 0., 0.], atol=.01, rtol=0.)
                self.assertLessEqual(abs(u), 10.)

    def test_backend_validation(self):
        with self.assertRaises(ValueError):
            lab.HydraulicTwinCore(backend='invented')
        self.assertEqual(lab.HydraulicTwinCore(backend='radau').backend, 'radau')

    def test_601_sample_fault_replay_native_versus_radau(self):
        native = lab.HydraulicTwinCore()
        if native.backend != 'native':
            self.skipTest('Native compiler unavailable; Radau fallback is separately covered')
        portable = lab.HydraulicTwinCore(backend='radau')
        state, u0 = native.equilibrium_for_velocity(.05)
        native.reset(x0=state); portable.reset(x0=state)
        errors = [np.zeros(4)]
        # 601 state samples from 600 physically integrated 1 ms intervals.
        # Includes load/bulk-modulus switch and a causal overdrive/recovery command.
        for k in range(600):
            load, beta = (0., 1.) if k < 100 else (5000., .5)
            u = u0 if k < 100 else 9.1 if k < 110 else 6.68
            a, _ = native.step(u, load_n=load, beta_multiplier=beta)
            b, info = portable.step(u, load_n=load, beta_multiplier=beta)
            errors.append(np.abs(a-b))
            self.assertEqual(info['backend'], 'radau')
        self.assertEqual(len(errors), 601)
        worst = np.max(errors, axis=0)
        self.assertLess(worst[0], 1e-8)
        self.assertLess(worst[1], 1e-6)
        self.assertTrue(np.all(worst[2:] < 100.), str(worst))


class GymEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = lab.HydraulicGymEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_gymnasium_contract(self):
        from gymnasium.utils.env_checker import check_env
        check_env(self.make_env(), skip_render_check=True)

    def test_normalization_exact_values_and_dtype(self):
        env = self.make_env()
        expected = np.array([1., 1., 1., 1., 1.], np.float32)
        actual = env.normalize_obs(np.array([10., 50., 500., 18., 23.]))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        for invalid in ([0., 0.], [0.,0.,float('nan'),10.,21.]):
            with self.assertRaises(ValueError):
                env.normalize_obs(invalid)

    def test_reset_and_actions_are_reproducible_for_seed(self):
        a, b = self.make_env(), self.make_env()
        oa, ia = a.reset(seed=2026); ob, ib = b.reset(seed=2026)
        np.testing.assert_array_equal(oa, ob)
        self.assertEqual(ia, ib)
        for action in (.35, .4, .3, .32, .36):
            sa, ra, ta, xa, _ = a.step(np.array([action], np.float32))
            sb, rb, tb, xb, _ = b.step(np.array([action], np.float32))
            np.testing.assert_array_equal(sa, sb)
            self.assertEqual((ra, ta, xa), (rb, tb, xb))

    def test_domain_randomization_envelope_and_variation(self):
        env = self.make_env()
        values = []
        for seed in range(16):
            _, info = env.reset(seed=seed)
            values.append([info[k] for k in ('m','beta_e','F_c')])
        values = np.array(values)
        self.assertTrue(np.all((200. <= values[:,0]) & (values[:,0] <= 300.)))
        self.assertTrue(np.all((600e6 <= values[:,1]) & (values[:,1] <= 1400e6)))
        self.assertTrue(np.all((840. <= values[:,2]) & (values[:,2] <= 1560.)))
        self.assertTrue(np.all(np.ptp(values, axis=0) > 0.))

    def test_evaluation_disables_randomization_and_restores_nominal_parameters(self):
        env = self.make_env()
        env.reset(seed=8)
        _, info = env.reset(seed=9, options={'evaluation':True})
        self.assertEqual([info[k] for k in ('m','beta_e','F_c')], [250., 1000e6, 1200.])

    def test_reward_terms_match_textbook_and_strict_bonus_boundary(self):
        terms = lab.HydraulicGymEnv.reward_terms(.1, 3., 2.)
        self.assertEqual(terms['reward_error'], -1.0000000000000002)
        self.assertAlmostEqual(terms['reward_energy'], -.9)
        self.assertEqual(terms['reward_smooth'], -1.)
        self.assertEqual(terms['reward_bonus'], 2.)
        self.assertEqual(lab.HydraulicGymEnv.reward_terms(.2, 0., 0.)['reward_bonus'], 0.)

    def test_step_reward_records_raw_formula_and_previous_control(self):
        env = self.make_env(domain_randomization=False, fault_augmentation=False)
        _, initial = env.reset(seed=1)
        old_u = 0.
        for action in (initial['initial_voltage_v']/10., .4):
            _, reward, terminated, _, info = env.step(np.array([action],np.float32))
            terms = env.reward_terms(info['position_error_mm'], info['control_v'], old_u)
            self.assertFalse(terminated)
            self.assertEqual(info['failure_penalty'], 0.)
            self.assertAlmostEqual(reward, sum(terms.values()), places=10)
            self.assertEqual(info['base_reward'], reward)
            self.assertAlmostEqual(info['reward_smooth'], -(info['control_v']-old_u)**2)
            old_u = info['control_v']

    def test_action_clipping_and_invalid_action_atomicity(self):
        env = self.make_env(domain_randomization=False, fault_augmentation=False)
        env.reset(seed=1)
        state, count = env.twin.state.copy(), env.step_count
        for action in ([], [0.,0.], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(env.twin.state,state)
            self.assertEqual(env.step_count,count)
        _, _, _, _, info = env.step([2.])
        self.assertEqual(info['control_v'],10.)

    def test_episode_truncation_requires_reset(self):
        env = self.make_env(max_steps=1, domain_randomization=False, fault_augmentation=False)
        _, info = env.reset(seed=2)
        _, _, terminated, truncated, _ = env.step([info['initial_voltage_v']/10.])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step([0.])
        env.reset(seed=2)
        self.assertEqual(env.step_count,0)

    def test_future_fault_labels_do_not_change_prefault_observation(self):
        a = self.make_env(domain_randomization=False, fault_augmentation=False)
        b = self.make_env(domain_randomization=False, fault_augmentation=False)
        oa, ia = a.reset(seed=1, options={'fault_time':.1,'load_n':5000.,'beta_multiplier':.5})
        ob, ib = b.reset(seed=1, options={'fault_time':.2,'load_n':1000.,'beta_multiplier':.9})
        np.testing.assert_array_equal(oa,ob)
        for _ in range(10):
            oa, *_ = a.step([ia['initial_voltage_v']/10.])
            ob, *_ = b.step([ib['initial_voltage_v']/10.])
            np.testing.assert_array_equal(oa,ob)
        self.assertEqual(a.current_fault(),(0.,1.))

    def test_observation_pressure_and_error_coordinates(self):
        env = self.make_env(domain_randomization=False, fault_augmentation=False)
        env.reset(seed=2)
        x,v,p1,p2 = env.twin.state
        raw = env.raw_observation()
        self.assertAlmostEqual(raw[0],0.,places=12)
        self.assertAlmostEqual(raw[1],0.,places=12)
        self.assertAlmostEqual(raw[3],(p1-env.twin.p['A2']/env.twin.p['A1']*p2)/1e6)
        self.assertEqual(raw[4],21.)


class SafetyFilterTests(unittest.TestCase):
    def test_dimensionless_barrier_shape(self):
        filt = lab.HOCBFSafetyFilter()
        self.assertEqual(filt.compute_barrier_h(0.),0.)
        self.assertEqual(filt.compute_barrier_h(100.),0.)
        self.assertEqual(filt.compute_barrier_h(50.),.25)
        self.assertLess(filt.compute_barrier_h(101.),0.)

    def test_reference_formula_matches_supplied_code(self):
        filt=lab.HOCBFSafetyFilter()
        low,high=filt.reference_bounds(95.,100.)
        self.assertEqual(low,max(-10.,((14.-48.)*100.-580.*(95.-1.2))/1850.))
        self.assertAlmostEqual(high,min(10.,((14.-48.)*100.+580.*(98.8-95.))/1850.))
        self.assertEqual(filt.filter_action(20.,0.,0.),0.)
        self.assertIn('no sampled certificate',filt.last_info.certificate_scope)

    def test_infeasible_intersection_is_reported(self):
        filt=lab.HOCBFSafetyFilter()
        with self.assertRaises(lab.SafetyInfeasibleError):
            filt.filter_action(98.,1000.,8.5)
        with self.assertRaises(lab.SafetyInfeasibleError):
            filt.filter_action(100.,0.,8.5)

    def test_sampled_gains_and_high_order_initial_set(self):
        with self.assertRaises(ValueError):
            lab.HOCBFSafetyFilter(sampled=True)
        with self.assertRaises(ValueError):
            lab.HOCBFSafetyFilter(sampled=True,kappa1=-1.,kappa2=49.)
        filt=lab.HOCBFSafetyFilter(sampled=True,kappa1=49.,kappa2=49.)
        self.assertEqual((filt.alpha1,filt.alpha2),(98.,2401.))
        with self.assertRaises(lab.SafetyInfeasibleError):
            filt.filter_action(98.8,100.,8.5)

    def test_exact_zoh_transition_matches_closed_form(self):
        filt=lab.HOCBFSafetyFilter(k_gain=2640.)
        x,v,u,t=.02,.4,-3.,.001
        expected_v=math.exp(-14*t)*v+2.64/14.*(1-math.exp(-14*t))*u
        expected_x=x+(1-math.exp(-14*t))/14.*v+2.64/14.*(t-(1-math.exp(-14*t))/14.)*u
        actual=filt.state_transition_si(x,v,u,t)
        np.testing.assert_allclose(actual,[expected_x,expected_v],rtol=1e-13,atol=1e-15)

    def test_projected_input_satisfies_whole_hold_interval(self):
        filt=lab.HOCBFSafetyFilter(k_gain=2640.,sampled=True,kappa1=49.,kappa2=49.)
        u=filt.filter_action(70.,600.,8.5)
        minima=filt.interval_minima(.070,.600,u)
        self.assertLess(u,8.5)
        self.assertTrue(all(v>=-1e-9 for v in minima.values()),str(minima))
        self.assertTrue(filt.last_info.intervened)

    def test_one_second_safety_demo_and_raw_collision_convention(self):
        fn=getattr(lab,'run_safety_experiment',None) or lab.run_safety_demo
        result=fn()
        summary=result['summary']
        data=result.get('data',result.get('trajectory'))
        self.assertTrue(summary['passed'])
        self.assertAlmostEqual(summary['minimum_safe_voltage_v'],-3.1,delta=.1)
        self.assertAlmostEqual(summary['safe_final_position_mm'],98.8,delta=.001)
        self.assertLess(summary['safe_max_position_mm'],100.)
        self.assertAlmostEqual(summary['unsafe_first_collision_s'],.1048911493,places=8)
        self.assertTrue(np.isnan(np.asarray(data['x_raw_m'])[-1]))
        self.assertTrue(all(summary['checks'].values()))


class EdgeDeploymentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(8)
        self.calibration=rng.uniform(-1.,1.,(128,5)).astype(np.float32)
        self.actor=lab.FloatActor([np.array([[.2,-.3,.4,.1,.05]],np.float32)],
                                 [np.array([.02],np.float32)])

    def test_zero_and_fully_pruned_weights_are_valid(self):
        for w,ratio in ((np.zeros((3,4),np.float32),.7),(np.ones((3,4),np.float32),1.)):
            dq,q,scale,zero,sparse=lab.EdgeLightweightExporter.prune_and_quantize_weights(w,ratio)
            self.assertEqual(q.dtype,np.int8)
            np.testing.assert_array_equal(q,np.zeros_like(q))
            np.testing.assert_array_equal(dq,np.zeros_like(dq))
            self.assertGreater(scale,0.)
            self.assertEqual(zero,0)
            self.assertEqual(sparse,100.)

    def test_stable_tie_pruning_has_exact_count(self):
        dq,_,_,_,sparse=lab.EdgeLightweightExporter.prune_and_quantize_weights(np.ones(10),.7)
        np.testing.assert_array_equal(dq[:7],np.zeros(7))
        self.assertTrue(np.all(dq[7:] > 0.))
        self.assertEqual(sparse,70.)

    def test_affine_quantization_and_reconstruction(self):
        weights=np.array([[-2.,-.5,0.,.5,3.]],np.float32)
        dq,q,scale,zero,sparse=lab.EdgeLightweightExporter.prune_and_quantize_weights(weights,0.)
        np.testing.assert_allclose(dq,scale*(q.astype(np.float32)-zero),rtol=0.,atol=0.)
        self.assertLessEqual(np.max(np.abs(dq-weights)),scale+.000001)
        self.assertGreaterEqual(int(q.min()),-128)
        self.assertLessEqual(int(q.max()),127)
        self.assertAlmostEqual(sparse,20.)

    def test_pruning_rejects_nan_empty_and_invalid_ratios(self):
        for ratio in (-.1,1.1,float('nan'),float('inf')):
            with self.subTest(ratio=ratio),self.assertRaises(ValueError):
                lab.EdgeLightweightExporter.prune_and_quantize_weights(np.ones(5),ratio)
        for weights in (np.array([]),np.array([np.nan]),np.array([np.inf])):
            with self.assertRaises(ValueError):
                lab.EdgeLightweightExporter.prune_and_quantize_weights(weights)

    def test_float_actor_validates_topology_and_observation(self):
        with self.assertRaises(ValueError):
            lab.FloatActor([np.zeros((3,5)),np.zeros((1,2))],[np.zeros(3),np.zeros(1)])
        with self.assertRaises(ValueError):
            lab.FloatActor([np.full((1,5),np.nan)],[np.zeros(1)])
        for obs in (np.zeros(4),np.full(5,np.nan),np.zeros((2,2,5))):
            with self.assertRaises(ValueError):
                self.actor(obs)

    def test_float_actor_batch_and_roundtrip(self):
        expected=np.vstack([self.actor(row) for row in self.calibration])
        np.testing.assert_allclose(self.actor(self.calibration),expected,atol=1e-7,rtol=0.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'float.npz';self.actor.save(path)
            restored=lab.FloatActor.load(path)
            np.testing.assert_array_equal(restored(self.calibration),self.actor(self.calibration))

    def test_integer_export_uses_int8_weights_and_matches_simple_actor(self):
        integer=lab.EdgeLightweightExporter.export_integer(self.actor,self.calibration,
                    prune_ratio=0.,activation_bits=16)
        self.assertTrue(all(w.dtype==np.int8 for w in integer.weights))
        self.assertTrue(all(b.dtype==np.int32 for b in integer.biases))
        self.assertEqual(integer.metadata['activation_bits'],16)
        self.assertLess(np.max(np.abs(integer(self.calibration)-self.actor(self.calibration))),.004)
        self.assertFalse(integer.metadata['deployment_ready'])

    def test_integer_roundtrip_is_exact(self):
        integer=lab.EdgeLightweightExporter.export_integer(self.actor,self.calibration,prune_ratio=0.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'integer.npz';integer.save(path)
            restored=lab.IntegerActor.load(path)
            np.testing.assert_array_equal(restored(self.calibration),integer(self.calibration))
            for obs in (np.zeros(4),np.full(5,np.nan)):
                with self.assertRaises(ValueError):
                    restored(obs)

    def test_integer_export_rejects_bad_calibration_and_precision(self):
        for calibration in (np.array([]),np.zeros(5),np.empty((0,5)),np.full((3,5),np.nan)):
            with self.assertRaises(ValueError):
                lab.EdgeLightweightExporter.export_integer(self.actor,calibration)
        with self.assertRaises(ValueError):
            lab.EdgeLightweightExporter.export_integer(self.actor,self.calibration,activation_bits=12)

    def test_integer_loader_rejects_unsafe_dtype_and_invalid_shift(self):
        integer=lab.EdgeLightweightExporter.export_integer(self.actor,self.calibration,prune_ratio=0.)
        kwargs=dict(weights=integer.weights,biases=integer.biases,weight_scales=integer.weight_scales,
            multipliers=integer.multipliers,metadata=integer.metadata,
            input_center=integer.input_center,input_channel_scale=integer.input_channel_scale)
        bad=dict(kwargs,weights=[integer.weights[0].astype(np.float32)])
        with self.assertRaises(ValueError):
            lab.IntegerActor(**bad)
        metadata=copy.deepcopy(integer.metadata);metadata['layers'][0]['shift']=100
        with self.assertRaises(ValueError):
            lab.IntegerActor(**dict(kwargs,metadata=metadata))

    def test_integer_loader_rejects_int32_accumulation_overflow(self):
        integer=lab.EdgeLightweightExporter.export_integer(self.actor,self.calibration,prune_ratio=0.)
        with self.assertRaisesRegex(ValueError,'INT32 accumulation'):
            lab.IntegerActor(integer.weights,[np.full_like(integer.biases[0],2**31-1)],
                integer.weight_scales,integer.multipliers,integer.metadata,
                integer.input_center,integer.input_channel_scale)

    def test_integer_loader_rejects_int64_requantization_overflow(self):
        actor=lab.FloatActor([np.array([[.2]*5,[-.1]*5],np.float32),
                             np.array([[.3,.5]],np.float32)],
                            [np.zeros(2,np.float32),np.zeros(1,np.float32)])
        integer=lab.EdgeLightweightExporter.export_integer(actor,self.calibration,prune_ratio=0.)
        multipliers=[np.full_like(integer.multipliers[0],2**62),integer.multipliers[1]]
        with self.assertRaisesRegex(ValueError,'INT64 requantization'):
            lab.IntegerActor(integer.weights,integer.biases,integer.weight_scales,multipliers,
                integer.metadata,integer.input_center,integer.input_channel_scale)

    def test_frozen_original_integer_fixture(self):
        actor=lab.IntegerActor.load(ROOT/'assets'/'actor_int8.npz')
        np.testing.assert_array_equal(actor(FIXTURE_OBS),FIXTURE_INT8)

    def test_frozen_original_float_fixture(self):
        actor=lab.FloatActor.load(ROOT/'assets'/'actor_fp32.npz')
        np.testing.assert_allclose(actor(FIXTURE_OBS),FIXTURE_FP32,atol=4e-7,rtol=0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
