"""独立验证模块三的量纲、双腔反演、因果性、约束状态和参考计划。"""
import math
import os
import tempfile
import unittest
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'chapter6-mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'chapter6-cache'))
import numpy as np
from scipy.integrate import trapezoid
from controllers import DigitalPID
from model_assisted_controller import ModelAssistedController
from twin_adapter import ClosedLoopTwin,classroom_config
from trajectory import generate_scurve,generate_process_scurve
from hydraulic_digital_twin import HydraulicTwin
from media_model import AeratedHydraulicModel
from simulation import Experiment,planned_load
from friction_model import (sliding_friction,sliding_friction_slope,
                            breakaway_friction)


class CompositeControllerTests(unittest.TestCase):
    def setUp(self):
        self.p=classroom_config();self.p.update(ambient_pressure=1e5,makeup_area=.001)
        cfg=Experiment('controller_test')
        self.pid=dict(kp=cfg.kp,ki=cfg.ki,kd=cfg.kd,ts=cfg.ts,alpha=cfg.alpha,
                      integral_separation=False)

    def controller(self,**kwargs):
        return ModelAssistedController(self.pid,self.p,**kwargs)

    def test_volts_to_flow_and_deadzone_have_correct_units(self):
        c=self.controller()
        opening_v=2.-c.deadzone_v
        q_from_voltage=c.k_flow*opening_v*math.sqrt(self.p['Ps']-2.5e6)
        normalized=2./10-self.p['deadzone']
        q_from_spool=self.p['Cd']*self.p['w']*normalized*self.p['xv_max']*math.sqrt(2*(self.p['Ps']-2.5e6)/self.p['rho'])
        self.assertAlmostEqual(q_from_voltage,q_from_spool,places=14)
        self.assertAlmostEqual(c.deadzone_v,.4)

    def test_nominal_pressures_satisfy_two_port_and_force_equilibrium(self):
        c=self.controller();p1,p2=c.nominal_pressures;p=self.p
        self.assertAlmostEqual((p['Ps']-p1)/(p2-p['Pr']),(p['A1']/p['A2'])**2,places=12)
        force=p1*p['A1']-p2*p['A2']-p['ambient_pressure']*(p['A1']-p['A2'])
        self.assertAlmostEqual(force,sliding_friction(.2,self.p),places=7)

    def test_dual_chamber_inverse_matches_independent_twin_pressure_derivatives(self):
        twin=ClosedLoopTwin(config=self.p,load_force=0.,valve_tau=0.)
        for velocity,pressures,force_rate in ((.2,(17e6,1.2e6),2e6),
                                                (-.1,(2e6,17e6),-3e6),
                                                (.01,(.09e6,.08e6),1e6)):
            with self.subTest(velocity=velocity,pressures=pressures):
                c=self.controller(pressure_alpha=0.,gain_prediction_s=0.)
                c.pressure_filtered=np.array(pressures)
                linear=c._inverse_force_rate(.08,velocity,force_rate)
                voltage=linear+math.copysign(c.deadzone_v,linear)
                state=np.array([.08,velocity,*pressures])
                d=twin._dynamics(0.,state,voltage/10)
                measured=self.p['A1']*d[2]-self.p['A2']*d[3]
                self.assertAlmostEqual(measured,force_rate,delta=max(1e-5,abs(force_rate)*1e-10))

    def test_analytic_friction_rate_matches_the_sliding_branch(self):
        c=self.controller()
        for v in (.001,.03,.2,-.04):
            epsilon=1e-7
            slope=(sliding_friction(v+epsilon,self.p)-sliding_friction(v-epsilon,self.p))/(2*epsilon)
            self.assertAlmostEqual(c._friction_rate(v,.7),slope*.7,delta=max(1e-4,abs(slope)*1e-6))
        # At zero use the one-sided slip derivative, not a central difference
        # through the jump between negative and positive slip branches.
        self.assertEqual(c._friction_rate(0.,.7),self.p['sigma']*.7)

    def test_zero_reference_without_motion_intent_has_no_hidden_pulse(self):
        c=self.controller()
        self.assertEqual(c.update((0.,0.,0.),(0.,0.),(1e5,1e5)),(0.,0.,0.))

    def test_planned_jerk_produces_calculated_valve_precharge(self):
        c=self.controller(pressure_alpha=0.,enable_startup_regularization=False)
        _,ff,_=c.update((0.,0.,0.),(0.,0.),(1e5,1e5),jerk_d=8.,origin_m=.01)
        a=math.exp(-.001/.0025)
        self.assertAlmostEqual(ff,c.last['valve_target_v']/(1-a),places=12)
        self.assertGreater(c.last['valve_target_v'],c.deadzone_v)
        self.assertEqual(c.last['reference_friction_n'],self.p['Fs'])
        self.assertEqual(c.last['planned_inverse_force_n'],self.p['Fs'])
        self.assertAlmostEqual(c.previous_valve_target_v,c.last['valve_target_v'],places=12)

    def test_zero_leakage_still_allows_mass_force_precharge(self):
        self.p['C_ip']=0.
        c=self.controller(pressure_alpha=0.,enable_valve_inverse=False,enable_startup_regularization=False)
        c.update((0.,0.,.5),(0.,0.),(1e5,1e5),origin_m=.01)
        self.assertGreater(c.last['valve_target_v'],c.deadzone_v)
        self.assertGreater(c.last['linear_voltage_v'],0.)

    def test_retraction_uses_negative_inverse_and_real_second_chamber_pressure(self):
        c=self.controller(pressure_alpha=0.,gain_prediction_s=0.,enable_valve_inverse=False,
                          force_error_gain_s=0.,enable_startup_regularization=False)
        _,ff,_=c.update((0.,-.1,-.2),(0.,-.1),(2e6,17e6),jerk_d=-5.,origin_m=.08)
        self.assertLess(ff,-c.deadzone_v)
        self.assertAlmostEqual(c.last['supply_delta_pa'],self.p['Ps']-17e6)

    def test_total_saturation_tracks_feedback_headroom(self):
        c=self.controller(pressure_alpha=0.,enable_startup_regularization=False)
        total,ff,fb=c.update((.1,.2,.5),(0.,0.),(1e5,1e5),jerk_d=8.,
                              known_force_rate_n_s=1e12,origin_m=.01)
        self.assertEqual(total,10.)
        self.assertAlmostEqual(total,ff+fb)
        self.assertAlmostEqual(c.pid.u_prev,fb-c.last['holding_deadzone_compensation_v'])

    def test_saturated_feedforward_remembers_achievable_valve_state(self):
        c=self.controller(pressure_alpha=0.,enable_startup_regularization=False)
        c.update((0.,.2,.5),(0.,0.),(1e5,1e5),jerk_d=8.,known_force_rate_n_s=1e12,origin_m=.01)
        self.assertGreater(c.last['feedforward_requested_v'],10.)
        a=math.exp(-.001/.0025)
        self.assertAlmostEqual(c.previous_valve_target_v,(1-a)*10.,places=12)
        self.assertLess(c.previous_valve_target_v,c.last['valve_target_v'])

    def test_holding_deadzone_bias_does_not_accumulate_in_pid(self):
        parameters=dict(self.pid,kp=1.,ki=0.,kd=0.)
        c=ModelAssistedController(parameters,self.p)
        for _ in range(5):
            total,ff,fb=c.update((.001,0.,0.),(0.,0.),(1e5,1e5))
            self.assertAlmostEqual(c.pid.u_prev,.1)
            self.assertAlmostEqual(fb,.1)
            self.assertEqual(c.last['holding_deadzone_compensation_v'],0.)
            self.assertAlmostEqual(total,ff+fb)

    def test_unknown_load_cannot_change_command_before_measurements_change(self):
        twins=[ClosedLoopTwin(config=self.p,load_force=f) for f in (0.,163000.)]
        controllers=[self.controller() for _ in twins]
        for twin in twins:twin.reset([.01,0.,1e5,1e5])
        # 比较初始边界误差衰减期间的整段指令，而非只检查第一个采样。
        samples=math.ceil(5*controllers[0].valve_tau/self.pid['ts'])+3
        traces=[]
        for c,twin in zip(controllers,twins):
            x,v,p1,p2=twin.state
            traces.append([c.update((0.,.2,0.),(x-.01,v),(p1,p2),jerk_d=0.,origin_m=.01)
                           for _ in range(samples)])
        self.assertEqual(traces[0],traces[1])
        self.assertTrue(any(abs(command[0])>1e-8 for command in traces[0]))

    def test_continuous_holding_inverse_compensates_real_leakage(self):
        c=self.controller(pressure_alpha=0.,enable_valve_inverse=False,
                          force_error_gain_s=0.,enable_startup_regularization=False)
        _,ff,_=c.update((0.,0.,0.),(0.,0.),(4.55e6,9e6),jerk_d=0.,origin_m=.14)
        self.assertLess(ff,-c.deadzone_v)
        twin=ClosedLoopTwin(config=self.p,load_force=0.,valve_tau=0.)
        derivatives=twin._dynamics(0.,np.array([.14,0.,4.55e6,9e6]),ff/10)
        self.assertAlmostEqual(self.p['A1']*derivatives[2]-self.p['A2']*derivatives[3],0.,delta=1e-5)

    def test_small_flow_reversal_crosses_deadzone_without_hard_jump(self):
        c=self.controller()
        self.assertGreater(c._deadzone_inverse(.001,True),.4)
        before=c.deadzone_offset_v
        for _ in range(20):
            c._deadzone_inverse(-.001,False)
            self.assertLessEqual(abs(c.deadzone_offset_v-before),40*.001+1e-12)
            before=c.deadzone_offset_v
        self.assertAlmostEqual(before,-.4,places=12)

    def test_force_error_feedback_uses_net_absolute_pressure_force(self):
        c=self.controller(pressure_alpha=0.,force_error_gain_s=80.,enable_startup_regularization=False)
        c.update((0.,.2,0.),(0.,.2),(2e6,1e6),jerk_d=0.,origin_m=.05)
        actual=2e6*self.p['A1']-1e6*self.p['A2']-1e5*(self.p['A1']-self.p['A2'])
        expected=80*(c.last['desired_force_n']-actual)
        self.assertAlmostEqual(c.last['actual_force_n'],actual)
        self.assertAlmostEqual(c.last['force_error_correction_n_s'],expected)
        self.assertAlmostEqual(c.last['desired_force_rate_n_s'],c.last['planned_force_rate_n_s']+expected)

    def test_motion_intent_uses_static_friction_and_known_force_in_both_directions(self):
        # Direction follows velocity, then acceleration, then analytic jerk.
        # A future motion intent can precharge a genuinely stationary piston.
        cases=((0.,0.,10.,1.),(0.,0.,-10.,-1.),
               (0.,.2,-10.,1.),(0.,-.2,10.,-1.),
               (.01,-.2,-10.,1.),(-.01,.2,10.,-1.))
        for velocity,acceleration,jerk,direction in cases:
            with self.subTest(velocity=velocity,acceleration=acceleration,jerk=jerk):
                c=self.controller(pressure_alpha=0.)
                c.update((0.,velocity,acceleration),(0.,velocity),(1e5,1e5),
                         jerk_d=jerk,known_force_n=123.,origin_m=.01)
                friction=sliding_friction(velocity,self.p,direction)
                expected=self.p['mass']*acceleration+friction+123.
                self.assertEqual(c.last['reference_motion_direction'],direction)
                self.assertEqual(c.last['reference_friction_n'],friction)
                self.assertAlmostEqual(c.last['planned_inverse_force_n'],expected)
                self.assertEqual(c.last['velocity_damping_force_n'],0.)
                self.assertEqual(c.last['static_hold_force_adjustment_n'],0.)
                self.assertAlmostEqual(c.last['desired_force_n'],expected)

    def test_velocity_damping_is_shared_current_measurement_feedback(self):
        logs=[]
        for scheduled in (False,True):
            c=self.controller(pressure_alpha=0.,enable_gain_sched=scheduled)
            c.update((0.,.2,.3),(0.,.15),(2e6,1e5),jerk_d=0.,known_force_n=5000.,origin_m=.01)
            planned=self.p['mass']*.3+sliding_friction(.2,self.p)+5000.
            damping=self.p['mass']*c.velocity_error_gain_s*(.2-.15)
            self.assertAlmostEqual(c.last['planned_inverse_force_n'],planned)
            self.assertAlmostEqual(c.last['velocity_damping_force_n'],damping)
            self.assertAlmostEqual(c.last['desired_force_n'],planned+damping)
            # No derivative of this sensor-dependent term is invented as a
            # known reference rate; only analytic trajectory/load derivatives.
            expected_rate=sliding_friction_slope(.2,self.p)*.3
            self.assertAlmostEqual(c.last['planned_force_rate_n_s'],expected_rate)
            logs.append(tuple(c.last[key] for key in ('planned_inverse_force_n',
                                                      'velocity_damping_force_n',
                                                      'desired_force_n','planned_force_rate_n_s')))
        self.assertEqual(logs[0],logs[1])

    def test_stationary_hold_preserves_only_the_valid_static_force_cone(self):
        known=1000.
        for offset in (-1200.,-400.,0.,400.,1200.):
            with self.subTest(offset=offset):
                c=self.controller(pressure_alpha=0.)
                actual=known+offset
                p2=self.p['Pr']
                p1=(actual+p2*self.p['A2']+1e5*(self.p['A1']-self.p['A2']))/self.p['A1']
                c.update((0.,0.,0.),(0.,0.),(p1,p2),jerk_d=0.,
                         known_force_n=known,origin_m=.01)
                expected=known+np.clip(offset,-self.p['Fs'],self.p['Fs'])
                self.assertEqual(c.last['reference_motion_direction'],0.)
                self.assertEqual(c.last['reference_friction_n'],0.)
                self.assertEqual(c.last['planned_inverse_force_n'],known)
                self.assertEqual(c.last['velocity_damping_force_n'],0.)
                self.assertAlmostEqual(c.last['desired_force_n'],expected)
                self.assertAlmostEqual(c.last['static_hold_force_adjustment_n'],expected-known)
                self.assertAlmostEqual(c.last['force_error_correction_n_s'],
                                       c.force_error_gain_s*(expected-actual))

    def test_feedforward_really_prebuilds_pressure_before_breakaway(self):
        parameters=dict(self.pid,kp=0.,ki=0.,kd=0.)
        c=ModelAssistedController(parameters,self.p,pressure_alpha=0.)
        twin=ClosedLoopTwin(config=self.p,load_force=0.,valve_tau=c.valve_tau,rtol=1e-7)
        twin.reset([.01,0.,self.p['Pr'],self.p['Pr']])
        built_while_stuck=False
        first_command=None
        for _ in range(80):
            x,v,p1,p2=twin.state
            command,ff,fb=c.update((0.,0.,0.),(x-.01,v),(p1,p2),jerk_d=10.,origin_m=.01)
            self.assertEqual(fb,0.)
            self.assertEqual(command,ff)
            if first_command is None:
                first_command=command
                self.assertEqual(twin.valve_state,0.)
                self.assertGreater(command,0.)
                self.assertAlmostEqual(command,c.last['valve_target_v'],places=12)
                self.assertEqual(c.last['planned_inverse_force_n'],self.p['Fs'])
            twin.step_voltage(command,c.pid.ts)
            if twin.sticking:
                self.assertEqual(twin.state[0],.01)
                self.assertEqual(twin.state[1],0.)
                built_while_stuck |= twin.state[2]>self.p['Pr']+1.
            if twin.release_count:
                break
        self.assertTrue(built_while_stuck)
        self.assertGreater(twin.release_count,0)
        release=twin.friction_events[0]
        self.assertEqual(release['kind'],'release')
        self.assertEqual(release['direction'],1)
        self.assertGreater(release['time_s'],0.)
        self.assertAlmostEqual(release['net_drive_n'],self.p['Fs']+twin.friction_force_tolerance,delta=1e-3)

    def test_idle_wait_does_not_consume_the_startup_boundary_condition(self):
        waiting=self.controller()
        fresh=self.controller()
        self.assertTrue(waiting.enable_startup_regularization)
        for _ in range(25):
            self.assertEqual(waiting.update((0.,0.,0.),(0.,0.),(1e5,1e5)),(0.,0.,0.))
        samples=math.ceil(5*waiting.valve_tau/waiting.pid.ts)+3
        for index in range(samples):
            expected=fresh.update((0.,0.,0.),(0.,0.),(1e5,1e5),jerk_d=8.,origin_m=.01)
            actual=waiting.update((0.,0.,0.),(0.,0.),(1e5,1e5),jerk_d=8.,origin_m=.01)
            self.assertEqual(actual,expected)
            if index==0:
                self.assertGreater(actual[1],0.)
                self.assertAlmostEqual(actual[1],waiting.last['valve_target_v'],places=12)
                a=math.exp(-waiting.pid.ts/waiting.valve_tau)
                self.assertAlmostEqual(waiting.previous_valve_target_v,(1-a)*actual[1],places=12)
                self.assertLess(waiting.previous_valve_target_v,actual[1])

    def test_constant_nonzero_target_has_constant_command_in_both_directions(self):
        for target in (.45,-.45):
            with self.subTest(target=target):
                c=self.controller()
                twin=ClosedLoopTwin(config=self.p,load_force=0.,valve_tau=c.valve_tau,rtol=1e-7)
                twin.reset([.01,0.,1e5,1e5])
                self.assertEqual(twin.valve_state,0.)
                a=math.exp(-c.pid.ts/c.valve_tau)
                samples=math.ceil(5*c.valve_tau/c.pid.ts)+3
                for index in range(samples):
                    command=c._valve_inverse(target)
                    # 对恒定参考应持续发出同一模型目标，不能先boost再骤降。
                    self.assertAlmostEqual(command,target,delta=1e-7)
                    twin.step_voltage(command,c.pid.ts)
                    expected=target*(1-a**(index+1))
                    self.assertAlmostEqual(10*twin.valve_state,expected,delta=1e-7)
                    c.previous_valve_target_v=10*twin.valve_state

    def test_changing_reference_preserves_the_original_decaying_boundary_error(self):
        c=self.controller()
        twin=ClosedLoopTwin(config=self.p,load_force=0.,valve_tau=c.valve_tau,rtol=1e-7)
        twin.reset([.01,0.,1e5,1e5])
        a=math.exp(-c.pid.ts/c.valve_tau)
        targets=(.45,.45,.5,.55,0.,0.,-.35,-.35,.6,.6,0.,-.2)
        initial_error=targets[0]
        for index,target in enumerate(targets):
            command=c._valve_inverse(target)
            self.assertLess(abs(command),10.)
            twin.step_voltage(command,c.pid.ts)
            # 独立正向积分应只保留原初态误差的自然衰减；过零不能重启补偿。
            expected=target-initial_error*a**(index+1)
            self.assertAlmostEqual(10*twin.valve_state,expected,delta=1e-7)
            c.previous_valve_target_v=10*twin.valve_state

    def test_reset_repeats_the_same_nonzero_startup_commands(self):
        c=self.controller()
        samples=math.ceil(5*c.valve_tau/c.pid.ts)+3
        def trace():
            return [c.update((0.,0.,0.),(0.,0.),(1e5,1e5),jerk_d=8.,origin_m=.01)
                    for _ in range(samples)]
        first=trace()
        self.assertGreater(first[0][1],0.)
        c.reset()
        self.assertEqual(c.previous_valve_target_v,0.)
        self.assertEqual(trace(),first)

    def test_startup_valve_memory_matches_actual_commands_after_saturation(self):
        parameters=dict(self.pid,kp=0.,ki=0.,kd=0.)
        c=ModelAssistedController(parameters,self.p,pressure_alpha=0.)
        twin=ClosedLoopTwin(config=self.p,load_force=0.,valve_tau=c.valve_tau,rtol=1e-7)
        twin.reset([.01,0.,1e5,1e5])
        saturated=False
        samples=math.ceil(5*c.valve_tau/c.pid.ts)+3
        for index in range(samples):
            x,v,p1,p2=twin.state
            rate=0. if index==0 else 1e12
            total,ff,fb=c.update((0.,0.,0.),(x-.01,v),(p1,p2),jerk_d=8.,
                                 known_force_rate_n_s=rate,origin_m=.01)
            self.assertEqual(fb,0.)
            self.assertEqual(total,ff)
            self.assertLessEqual(abs(total),10.)
            if index==0:
                self.assertEqual(twin.valve_state,0.)
                self.assertGreater(ff,0.)
                self.assertAlmostEqual(ff,c.last['valve_target_v'],places=12)
                self.assertLess(c.previous_valve_target_v,ff)
            twin.step_voltage(total,c.pid.ts)
            # 未限幅目标不能写入FF记忆；独立Twin接受的是实际有限电压。
            self.assertAlmostEqual(c.previous_valve_target_v,10*twin.valve_state,delta=1e-6)
            saturated=saturated or abs(c.last['feedforward_requested_v'])>10.
        self.assertTrue(saturated)

    def test_startup_regularization_leaves_pure_pid_commands_unchanged(self):
        c=self.controller(enable_ff=False,enable_gain_sched=False)
        baseline=DigitalPID(**self.pid)
        for target,measurement in ((0.,0.),(.0001,0.),(.0002,.00005),(.0001,.00015),
                                   (0.,.0001),(-.0001,0.),(0.,0.)):
            total,ff,fb=c.update((target,0.,0.),(measurement,0.),(1e5,1e5),jerk_d=8.)
            expected=baseline.update(target,measurement)
            self.assertEqual(ff,0.)
            self.assertEqual((total,fb),(expected,expected))

    def test_protection_and_parameter_validation(self):
        for kwargs in ({'gain_prediction_s':-1.},{'gain_prediction_s':math.nan},
                       {'valve_tau':math.inf},{'pressure_alpha':1.},
                       {'force_error_gain_s':-1.},{'zero_flow_band_v':-1.},{'deadzone_slew_v_s':0.},
                       {'velocity_error_gain_s':-1.},{'velocity_error_gain_s':math.inf}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.controller(**kwargs)
        c=self.controller(pressure_alpha=0.)
        with self.assertRaises(ValueError):c.update((0.,0.,0.),(0.,0.),(-1.,1e5))
        c.update((0.,.2,0.),(0.,.2),(self.p['Ps']-1.,1e5),jerk_d=0.,origin_m=.08)
        self.assertGreaterEqual(c.last['supply_delta_pa'],5e5)
        self.assertTrue(np.isfinite(c.last['voltage_v']))


class HybridFrictionTests(unittest.TestCase):
    def setUp(self):
        self.p=classroom_config()

    def constant_load_twin(self,force,velocity=0.):
        # A signed deterministic test fixture, not an extra controller input.
        p=dict(self.p,C_ip=0.,C_ep=0.)
        twin=ClosedLoopTwin(config=p,load_force=0.,valve_tau=.0025,rtol=1e-7)
        twin.load_at=lambda time:force
        twin.reset([.01,velocity,p['Pr'],p['Pr']])
        return twin

    def test_slip_limits_retain_static_friction_and_dissipate_energy(self):
        for direction in (-1,1):
            self.assertEqual(breakaway_friction(direction,self.p),direction*self.p['Fs'])
            self.assertEqual(sliding_friction(0.,self.p,direction),direction*self.p['Fs'])
            for speed in (1e-9,.001,.02,.05,.2):
                velocity=direction*speed
                self.assertGreater(sliding_friction(velocity,self.p)*velocity,0.)
                epsilon=1e-8
                finite=(sliding_friction(velocity+epsilon,self.p,direction)
                        -sliding_friction(velocity-epsilon,self.p,direction))/(2*epsilon)
                self.assertAlmostEqual(sliding_friction_slope(velocity,self.p,direction),finite,
                                       delta=max(1e-4,abs(finite)*1e-6))
        self.assertEqual(sliding_friction(0.,self.p),0.)

    def test_subthreshold_external_forces_are_balanced_without_drift(self):
        for force in (-self.p['Fs'],-400.,0.,400.,self.p['Fs']):
            with self.subTest(force=force):
                twin=self.constant_load_twin(force)
                before=twin.state.copy()
                twin.step_voltage(0.,.01)
                np.testing.assert_array_equal(twin.state,before)
                diagnostics=twin.friction_diagnostics()
                self.assertTrue(diagnostics['sticking'])
                self.assertFalse(diagnostics['release_pending'])
                self.assertEqual(diagnostics['net_drive_force_n'],-force)
                self.assertEqual(diagnostics['friction_force_n'],-force)
                self.assertEqual(twin.friction_events,[])

    def test_superthreshold_external_forces_cause_physical_release(self):
        for force in (-1000.,1000.):
            with self.subTest(force=force):
                twin=self.constant_load_twin(force)
                twin.step_voltage(0.,.001)
                direction=-int(np.sign(force))
                self.assertGreater(direction*twin.state[1],0.)
                self.assertGreater(direction*(twin.state[0]-.01),0.)
                event=twin.friction_events[0]
                self.assertEqual(event['kind'],'release')
                self.assertEqual(event['time_s'],0.)
                self.assertEqual(event['direction'],direction)
                self.assertGreater(abs(event['net_drive_n']),self.p['Fs'])

    def test_pressure_and_valve_evolve_while_piston_is_stuck(self):
        for direction in (-1,1):
            with self.subTest(direction=direction):
                twin=self.constant_load_twin(0.)
                pressure_changed=False
                for _ in range(60):
                    twin.step_voltage(direction*2.,.00025)
                    if twin.sticking:
                        self.assertEqual(twin.state[0],.01)
                        self.assertEqual(twin.state[1],0.)
                        self.assertGreater(direction*twin.valve_state,0.)
                        pressure_changed |= bool(np.max(abs(twin.state[2:]-self.p['Pr']))>1.)
                    if twin.release_count:
                        break
                self.assertTrue(pressure_changed)
                self.assertGreater(twin.release_count,0)
                event=twin.friction_events[0]
                self.assertEqual(event['direction'],direction)
                self.assertGreater(event['time_s'],0.)
                self.assertAlmostEqual(abs(event['net_drive_n']),
                                       self.p['Fs']+twin.friction_force_tolerance,delta=1e-3)
                self.assertGreater(direction*twin.state[1],0.)

    def test_velocity_zero_event_resticks_and_then_holds_exactly(self):
        for direction in (-1,1):
            with self.subTest(direction=direction):
                twin=self.constant_load_twin(0.,direction*.001)
                self.assertFalse(twin.sticking)
                twin.step_voltage(0.,.005)
                self.assertTrue(twin.sticking)
                self.assertEqual(twin.state[1],0.)
                event=twin.friction_events[0]
                self.assertEqual(event['kind'],'restick')
                self.assertGreater(event['time_s'],0.)
                self.assertLess(event['time_s'],.005)
                self.assertLessEqual(abs(event['net_drive_n']),self.p['Fs'])
                position=twin.state[0]
                twin.step_voltage(0.,.002)
                self.assertEqual(twin.state[0],position)
                self.assertEqual(twin.state[1],0.)
                self.assertEqual(twin.restick_count,1)

    def test_zero_velocity_outside_static_cone_reverses_without_false_hold(self):
        for direction in (-1,1):
            with self.subTest(direction=direction):
                twin=self.constant_load_twin(direction*2000.,direction*.001)
                twin.step_voltage(0.,.001)
                event=twin.friction_events[0]
                self.assertEqual(event['kind'],'reverse_release')
                self.assertEqual(event['direction'],-direction)
                self.assertGreater(abs(event['net_drive_n']),self.p['Fs'])
                self.assertEqual(twin.restick_count,0)
                self.assertGreater(-direction*twin.state[1],0.)

    def test_sampled_load_boundary_distinguishes_pending_from_released(self):
        twin=ClosedLoopTwin(config=self.p,load_force=1000.,load_window=(.001,.003),
                            valve_tau=.0025,rtol=1e-7)
        twin.reset([.01,0.,self.p['Pr'],self.p['Pr']])
        twin.step_voltage(0.,.001)
        diagnostics=twin.friction_diagnostics()
        self.assertTrue(diagnostics['sticking'])
        self.assertTrue(diagnostics['release_pending'])
        self.assertEqual(twin.release_count,0)
        self.assertEqual(twin.state[1],0.)
        self.assertEqual(diagnostics['net_drive_force_n'],-1000.)
        self.assertEqual(diagnostics['friction_force_n'],-self.p['Fs'])
        twin.step_voltage(0.,.001)
        self.assertEqual(twin.friction_events[0]['time_s'],.001)
        self.assertEqual(twin.friction_events[0]['kind'],'release')

    def test_friction_parameters_are_validated_before_integration(self):
        for changes in ({'Fs':400.},{'Fc':-1.},{'vs':0.},{'sigma':-1.},{'Fs':math.nan}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                ClosedLoopTwin(config=dict(self.p,**changes),load_force=0.)


class ReferenceTests(unittest.TestCase):
    def test_reference_distance_velocity_and_clock_agree(self):
        t,x,v,a=generate_scurve(dist=.13,T_total=1.2,dt=.001)
        self.assertEqual(len(t),1201)
        np.testing.assert_array_equal(t,np.arange(1201)*.001)
        self.assertAlmostEqual(v.max(),.2,places=12)
        np.testing.assert_array_equal(x[1000:],np.full(201,.13))
        np.testing.assert_array_equal(v[1000:],np.zeros(201))
        self.assertAlmostEqual(float(trapezoid(v,t)),.13,places=8)

    def test_process_reference_is_analytic_and_matches_approach_cruise_hold(self):
        t,x,v,a,j=generate_process_scurve()
        self.assertAlmostEqual(v[160],.032,places=12)
        self.assertAlmostEqual(v[350],.2,places=12)
        self.assertAlmostEqual(v[500],.2,places=12)
        self.assertLessEqual(abs(j).max(),40.)
        np.testing.assert_array_equal(x[1000:],np.full(201,.13))
        np.testing.assert_array_equal(v[1000:],np.zeros(201))
        self.assertAlmostEqual(float(trapezoid(v,t)),.13,delta=1e-7)
        for index in (20,75,120,200,300,500,850,980):
            self.assertAlmostEqual((x[index+1]-x[index-1])/.002,v[index],delta=1e-5)
            self.assertAlmostEqual((v[index+1]-v[index-1])/.002,a[index],delta=1e-8)

    def test_media_zero_air_restores_original_and_air_has_physical_compliance(self):
        p=classroom_config()
        zero=AeratedHydraulicModel(dict(p,air_fraction_at_pr=0.))
        original=HydraulicTwin(p)
        mixed=AeratedHydraulicModel(p)
        for pressure in (1e5,5e5,2.5e6,17e6):
            self.assertEqual(zero._get_beta(pressure),original._get_beta(pressure))
            self.assertGreater(mixed._get_beta(pressure),0.)
            self.assertLess(mixed._get_beta(pressure),original._get_beta(pressure))
        self.assertLess(mixed._get_beta(1e5),2e7)
        self.assertGreater(mixed._get_beta(17e6),1.7e9)
        with self.assertRaises(ValueError):AeratedHydraulicModel(dict(p,air_fraction_at_pr=1.))

    def test_known_process_reference_matches_applied_load_and_stays_continuous(self):
        cfg=Experiment('reference')
        twin=ClosedLoopTwin(load_force=cfg.load_force,load_window=(cfg.load_start,cfg.load_end),
                            load_ramp=cfg.load_ramp,unload_ramp=cfg.unload_ramp)
        for t in np.arange(1201)*.001:
            force,rate=planned_load(t,cfg)
            self.assertAlmostEqual(force,twin.load_at(t),delta=1e-7)
        for t in (cfg.load_start,cfg.load_end):
            left=planned_load(t-1e-8,cfg)[0]
            right=planned_load(t+1e-8,cfg)[0]
            self.assertAlmostEqual(left,right,delta=1e-5)
        force,rate=planned_load(cfg.load_start+cfg.load_ramp/2,cfg)
        self.assertAlmostEqual(force,.5*cfg.load_force,delta=1e-7)
        self.assertAlmostEqual(rate,.5*cfg.load_force*np.pi/cfg.load_ramp,delta=1e-5)


if __name__=='__main__':
    unittest.main(verbosity=2)
