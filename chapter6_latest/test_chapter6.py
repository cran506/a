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
from trajectory import generate_scurve
from simulation import Experiment,planned_load


class CompositeControllerTests(unittest.TestCase):
    def setUp(self):
        self.p=classroom_config();self.p.update(ambient_pressure=1e5,makeup_area=.001)
        self.pid=dict(kp=3.8,ki=100.,kd=.02,ts=.001,alpha=.85,integral_separation=False)

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
        self.assertAlmostEqual(force,c.model._friction(.2),places=7)

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

    def test_analytic_friction_rate_matches_original_model_finite_difference(self):
        c=self.controller()
        for v in (0.,.001,.03,.2,-.04):
            epsilon=1e-7
            slope=(c.model._friction(v+epsilon)-c.model._friction(v-epsilon))/(2*epsilon)
            self.assertAlmostEqual(c._friction_rate(v,.7),slope*.7,delta=max(1e-4,abs(slope)*1e-6))

    def test_zero_reference_without_motion_intent_has_no_hidden_pulse(self):
        c=self.controller()
        self.assertEqual(c.update((0.,0.,0.),(0.,0.),(1e5,1e5)),(0.,0.,0.))

    def test_planned_jerk_produces_calculated_valve_precharge(self):
        c=self.controller(pressure_alpha=0.)
        _,ff,_=c.update((0.,0.,0.),(0.,0.),(1e5,1e5),jerk_d=8.,origin_m=.01)
        a=math.exp(-.001/.0025)
        self.assertAlmostEqual(ff,c.last['valve_target_v']/(1-a),places=12)
        self.assertGreater(ff,1.2)
        self.assertLess(ff,1.3)
        self.assertAlmostEqual(c.previous_valve_target_v,c.last['valve_target_v'],places=12)

    def test_zero_leakage_still_allows_mass_force_precharge(self):
        self.p['C_ip']=0.
        c=self.controller(pressure_alpha=0.,enable_valve_inverse=False)
        c.update((0.,0.,.5),(0.,0.),(1e5,1e5),origin_m=.01)
        self.assertGreater(c.last['valve_target_v'],c.deadzone_v)
        self.assertGreater(c.last['linear_voltage_v'],0.)

    def test_retraction_uses_negative_inverse_and_real_second_chamber_pressure(self):
        c=self.controller(pressure_alpha=0.,gain_prediction_s=0.,enable_valve_inverse=False)
        _,ff,_=c.update((0.,-.1,-.2),(0.,-.1),(2e6,17e6),jerk_d=-5.,origin_m=.08)
        self.assertLess(ff,-c.deadzone_v)
        self.assertAlmostEqual(c.last['supply_delta_pa'],self.p['Ps']-17e6)

    def test_total_saturation_tracks_feedback_headroom(self):
        c=self.controller(pressure_alpha=0.)
        total,ff,fb=c.update((.1,.2,.5),(0.,0.),(1e5,1e5),jerk_d=8.,
                              known_force_rate_n_s=1e12,origin_m=.01)
        self.assertEqual(total,10.)
        self.assertAlmostEqual(total,ff+fb)
        self.assertAlmostEqual(c.pid.u_prev,fb-c.last['holding_deadzone_compensation_v'])

    def test_saturated_feedforward_remembers_achievable_valve_state(self):
        c=self.controller(pressure_alpha=0.)
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
            self.assertAlmostEqual(fb,.5)
            self.assertAlmostEqual(total,ff+fb)

    def test_unknown_load_cannot_change_command_before_measurements_change(self):
        twins=[ClosedLoopTwin(config=self.p,load_force=f) for f in (0.,163000.)]
        controllers=[self.controller() for _ in twins]
        for twin in twins:twin.reset([.01,0.,1e5,1e5])
        commands=[]
        for c,twin in zip(controllers,twins):
            x,v,p1,p2=twin.state
            commands.append(c.update((0.,.2,0.),(x-.01,v),(p1,p2),jerk_d=0.,origin_m=.01))
        self.assertEqual(commands[0],commands[1])

    def test_protection_and_parameter_validation(self):
        for kwargs in ({'gain_prediction_s':-1.},{'gain_prediction_s':math.nan},
                       {'valve_tau':math.inf},{'pressure_alpha':1.}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.controller(**kwargs)
        c=self.controller(pressure_alpha=0.)
        with self.assertRaises(ValueError):c.update((0.,0.,0.),(0.,0.),(-1.,1e5))
        c.update((0.,.2,0.),(0.,.2),(self.p['Ps']-1.,1e5),jerk_d=0.,origin_m=.08)
        self.assertGreaterEqual(c.last['supply_delta_pa'],5e5)
        self.assertTrue(np.isfinite(c.last['voltage_v']))


class ReferenceTests(unittest.TestCase):
    def test_reference_distance_velocity_and_clock_agree(self):
        t,x,v,a=generate_scurve(dist=.13,T_total=1.2,dt=.001)
        self.assertEqual(len(t),1201)
        np.testing.assert_array_equal(t,np.arange(1201)*.001)
        self.assertAlmostEqual(v.max(),.2,places=12)
        np.testing.assert_array_equal(x[1000:],np.full(201,.13))
        np.testing.assert_array_equal(v[1000:],np.zeros(201))
        self.assertAlmostEqual(float(trapezoid(v,t)),.13,places=8)

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
