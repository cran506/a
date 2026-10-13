"""习题5-1回归：控制律机制与已保存数据；单测不重跑完整液压ODE。"""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'homework5_1_test_mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'homework5_1_test_cache'))
import numpy as np
from pid_algorithms import IncrementalPID,PositionalPID


class PIDMechanismTests(unittest.TestCase):
    def test_measurement_derivative_step_uses_cm_without_setpoint_kick(self):
        for cls in (PositionalPID,IncrementalPID):
            with self.subTest(controller=cls.__name__):
                c=cls()
                self.assertEqual(c.update(0.,0.),0.)
                self.assertEqual(c.update(.1,0.),10.)
                self.assertAlmostEqual(c.last_raw,42.22,places=12)
                self.assertEqual(c.last_d,0.)
                self.assertAlmostEqual(c.last_di,.22,places=12)
                for _ in range(20):
                    self.assertEqual(c.update(.1,0.),10.)
                self.assertEqual(c.u_prev,10.)
                if cls is PositionalPID:
                    self.assertAlmostEqual(c.i_total,21*.22,places=12)
                    self.assertAlmostEqual(c.last_raw,42.+21*.22,places=12)
                else:
                    self.assertAlmostEqual(c.last_raw,10.22,places=12)
                    self.assertEqual(c.i_total,0.)

    def test_original_error_derivative_retains_its_kick_and_decay(self):
        a,b=PositionalPID(derivative_mode='error'),IncrementalPID(derivative_mode='error')
        expected_a=(132.22,118.94,107.685)
        expected_b=(132.22,-3.28,-14.535)
        expected_d=(90.,76.5,65.025)
        for raw_a,raw_b,derivative in zip(expected_a,expected_b,expected_d):
            actual_a=a.update(.1,0.)
            actual_b=b.update(.1,0.)
            self.assertAlmostEqual(a.last_raw,raw_a,places=11)
            self.assertAlmostEqual(b.last_raw,raw_b,places=11)
            self.assertAlmostEqual(a.last_d,derivative,places=11)
            self.assertAlmostEqual(b.last_d,derivative,places=11)
            self.assertEqual(actual_a,10.)
            self.assertAlmostEqual(actual_b,np.clip(raw_b,-10.,10.),places=11)

    def test_unsaturated_forms_are_equivalent_for_both_derivative_modes(self):
        for mode in ('measurement','error'):
            with self.subTest(mode=mode):
                parameters=dict(out_min=-1000.,out_max=1000.,derivative_mode=mode)
                a,b=PositionalPID(**parameters),IncrementalPID(**parameters)
                for k in range(500):
                    target=.002*((k%17)-8)
                    measured=.0009*((k%23)-11)
                    ua,ub=a.update(target,measured),b.update(target,measured)
                    self.assertLess(abs(a.last_raw),1000.)
                    self.assertLess(abs(b.last_raw),1000.)
                    self.assertAlmostEqual(ua,ub,delta=1e-10)
                    self.assertAlmostEqual(a.last_d,b.last_d,delta=1e-12)

    def test_quantized_measurement_jump_has_shared_signed_filtered_derivative(self):
        lsb=.3/2**12
        for cls in (PositionalPID,IncrementalPID):
            with self.subTest(controller=cls.__name__):
                c=cls(kp=0.,ki=0.)
                c.update(0.,0.)
                c.update(.1,0.)
                self.assertEqual(c.last_d,0.)
                c.update(.1,lsb)
                expected=-(1-c.alpha)*c.kd/c.ts*100*lsb
                self.assertAlmostEqual(c.last_d,expected,places=12)
                for _ in range(8):
                    c.update(.1,lsb)
                    expected*=c.alpha
                    self.assertAlmostEqual(c.last_d,expected,places=12)

    def test_integral_only_reversal_exposes_windup_without_derivative_effects(self):
        for sign in (-1,1):
            with self.subTest(sign=sign):
                a,b=PositionalPID(kp=0.,kd=0.),IncrementalPID(kp=0.,kd=0.)
                for _ in range(100):
                    a.update(sign*.1,0.)
                    b.update(sign*.1,0.)
                self.assertAlmostEqual(a.i_total,sign*22.,places=11)
                self.assertEqual(a.u_prev,sign*10.)
                self.assertEqual(b.u_prev,sign*10.)
                # Only Ki acts here: no claim that every complete PID must
                # desaturate whenever its error becomes nonpositive.
                self.assertEqual(a.update(-sign*.01,0.),sign*10.)
                self.assertAlmostEqual(b.update(-sign*.01,0.),sign*(10.-.022),places=12)
                self.assertGreater(abs(a.last_raw),10.)

    def test_clipped_output_memory_applies_the_exact_integral_tracking_correction(self):
        c=IncrementalPID()
        previous_effective=0.
        previous_actual=0.
        overflow_seen=False
        inputs=[(.1,0.)]*30+[(-.1,0.)]*30+[(.1,.08),(0.,.1),(-.1,-.08),(0.,0.)]
        for target,measured in inputs:
            actual=c.update(target,measured)
            expected=(previous_effective+c.last_di+actual-c.last_raw)
            self.assertAlmostEqual(c.last_effective_integral,expected,places=10)
            self.assertEqual(c.state_before,previous_actual)
            self.assertEqual(c.u_prev,actual)
            self.assertLessEqual(abs(actual),10.)
            self.assertEqual(c.i_total,0.)
            overflow_seen |= abs(c.last_raw)>10.
            previous_effective=c.last_effective_integral
            previous_actual=actual
        self.assertTrue(overflow_seen)
        # The bounded state is output voltage, not the effective integral.
        first=IncrementalPID()
        first.update(.1,0.)
        self.assertAlmostEqual(first.last_effective_integral,-32.,places=12)

    def test_reset_repeats_a_cold_start_in_the_same_relative_coordinate(self):
        for cls in (PositionalPID,IncrementalPID):
            for mode in ('measurement','error'):
                with self.subTest(controller=cls.__name__,mode=mode):
                    c=cls(derivative_mode=mode)
                    trace=((0.,0.),(.1,0.),(.1,.03),(.1,.11),(-.1,.02),(0.,0.))
                    def run(controller):
                        return [(controller.update(r,y),controller.last_raw,controller.last_d,
                                 controller.last_di,controller.i_total,
                                 controller.last_effective_integral) for r,y in trace]
                    expected=run(c)
                    c.reset()
                    for field in ('e_prev','d_input_prev','d_prev','u_prev','i_total','last_raw',
                                  'last_d','last_di','last_effective_integral','last_delta','state_before'):
                        self.assertEqual(getattr(c,field),0.)
                    self.assertEqual(run(c),expected)
                    self.assertEqual(run(cls(derivative_mode=mode)),expected)


class HybridEventTests(unittest.TestCase):
    def test_real_zero_velocity_reversal_is_followed_by_a_later_restick(self):
        from twin_adapter import ClosedLoopTwin,physical_config
        # Recorded failing state from /tmp/ch5_hw51_guard_snapshot.json.
        # Keep the fixture here so the regression remains portable.
        snapshot=dict(
            state=[.15024694554971024,-.000844494384895573,
                   9393506.601500293,18510739.021173734],
            valve_state=-.06824271493584522,
            voltage=-.665970403579959,time=.8340000000000006,direction=-1)
        records=[]
        default_guard=ClosedLoopTwin.friction_event_sign_value
        for guard in (default_guard,1e-12,1e-8):
            with self.subTest(event_sign_value=guard):
                p=physical_config();p['xv_max']=.0025
                plant=ClosedLoopTwin(config=p,load_force=0.,valve_tau=.0053,rtol=1e-7)
                plant.reset(snapshot['state'])
                plant.t=snapshot['time']
                plant.valve_state=snapshot['valve_state']
                plant.sticking=False
                plant.slip_direction=snapshot['direction']
                plant.friction_event_sign_value=guard
                plant.step_voltage(snapshot['voltage'],.001)
                self.assertTrue(np.all(np.isfinite(plant.state)))
                self.assertGreater(min(plant.state[2:]),0.)
                self.assertAlmostEqual(plant.t,snapshot['time']+.001,places=12)
                self.assertTrue(plant.sticking)
                self.assertEqual(plant.state[1],0.)
                self.assertEqual(plant.slip_direction,0)
                self.assertLessEqual(abs(plant.net_drive_force()),p['Fs'])
                events=plant.friction_events
                self.assertGreaterEqual(len(events),2)
                self.assertLess(len(events),plant.max_friction_events_per_step)
                times=np.array([event['time_s'] for event in events])
                self.assertTrue(np.all(np.diff(times)>0.),'Repeated initial zero roots must not create same-time events')
                self.assertGreater(times[0],snapshot['time'])
                self.assertLess(times[-1],plant.t)
                reverse=events[0]
                self.assertEqual(reverse['kind'],'reverse_release')
                self.assertEqual(reverse['direction'],1)
                self.assertGreater(reverse['net_drive_n'],p['Fs'])
                later=[event for event in events[1:] if event['kind']=='restick']
                self.assertTrue(later)
                restick=later[0]
                self.assertGreater(restick['time_s'],reverse['time_s'])
                self.assertEqual(restick['direction'],0)
                self.assertLessEqual(abs(restick['net_drive_n']),p['Fs'])
                self.assertAlmostEqual(restick['friction_n'],restick['net_drive_n'],places=10)
                records.append(([event['kind'] for event in events],times))
        if len(records)==3:
            # The event's right-limit sign is a root guard, not a velocity seed:
            # changing its magnitude must not alter physical switching times.
            for kinds,times in records[1:]:
                self.assertEqual(kinds,records[0][0])
                self.assertLess(np.max(abs(times-records[0][1])),1e-8)


class SavedResultTests(unittest.TestCase):
    def test_saved_csv_is_causal_and_summary_describes_the_actual_arrays(self):
        folder=Path(__file__).resolve().parent/'outputs'
        if not (folder/'summary.json').is_file():
            self.skipTest('运行 work_5_1.py 后可检查保存的数据；控制律测试不依赖生成产物')
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        pair={}
        for kind in ('positional','incremental'):
            with self.subTest(controller=kind):
                array=np.genfromtxt(folder/f'{kind}.csv',delimiter=',',names=True)
                d={name:array[name] for name in array.dtype.names}
                cfg=summary[kind]['config'];m=summary[kind]['metrics']
                t=d['time_s'];ts=cfg['ts'];limit=cfg['voltage_limit_v']
                self.assertEqual(len(t),round(cfg['duration']/ts)+1)
                np.testing.assert_allclose(t,np.arange(len(t))*ts,rtol=0.,atol=1e-12)
                expected_r=np.where(np.arange(len(t))>=round(cfg['step_time']/ts),cfg['step_amplitude_m'],0.)
                np.testing.assert_array_equal(d['reference_m'],expected_r)
                for name,values in d.items():
                    if name=='explicit_integral_v' and kind=='incremental':
                        self.assertTrue(np.all(np.isnan(values)))
                    else:
                        self.assertTrue(np.all(np.isfinite(values)),name)
                self.assertGreater(min(d['p1_pa'].min(),d['p2_pa'].min()),0.)
                np.testing.assert_allclose(d['true_error_m'],expected_r-d['position_m'],rtol=0.,atol=1e-14)
                np.testing.assert_allclose(d['sampled_error_m'],expected_r-d['measured_m'],rtol=0.,atol=1e-14)
                lsb=cfg['sensor_span_m']/2**cfg['adc_bits']
                codes=np.clip(np.rint((d['position_m']+cfg['origin_m'])/lsb),0,2**cfg['adc_bits']-1)
                np.testing.assert_allclose(d['measured_m'],codes*lsb-cfg['origin_m'],rtol=0.,atol=1e-13)
                np.testing.assert_array_equal(d['state_before_v'][1:],d['state_after_v'][:-1])
                np.testing.assert_array_equal(d['state_after_v'],d['voltage_v'])
                self.assertLessEqual(np.max(abs(d['state_after_v'])),limit)
                np.testing.assert_allclose(d['voltage_v'],np.clip(d['raw_voltage_v'],-limit,limit),rtol=0.,atol=1e-12)
                error=100*d['sampled_error_m']
                derivative_input=error if cfg['derivative_mode']=='error' else -100*d['measured_m']
                previous_d=np.r_[0.,d['derivative_v'][:-1]]
                expected_d=cfg['alpha']*previous_d+(1-cfg['alpha'])*cfg['kd']/ts*np.diff(derivative_input,prepend=0.)
                np.testing.assert_allclose(d['derivative_v'],expected_d,rtol=0.,atol=1e-10)
                di=cfg['ki']*error*ts
                np.testing.assert_allclose(d['integral_increment_v'],di,rtol=0.,atol=1e-12)
                if kind=='positional':
                    integral=np.cumsum(di)
                    np.testing.assert_allclose(d['explicit_integral_v'],integral,rtol=0.,atol=1e-10)
                    expected_raw=cfg['kp']*error+integral+d['derivative_v']
                else:
                    expected_raw=d['state_before_v']+cfg['kp']*np.diff(error,prepend=0.)+di+d['derivative_v']-previous_d
                np.testing.assert_allclose(d['raw_voltage_v'],expected_raw,rtol=0.,atol=1e-10)
                tau=cfg['valve_tau_s']
                a=math.exp(-ts/tau) if tau else 0.
                expected_valve=a*d['valve_normalized'][:-1]+(1-a)*d['voltage_v'][:-1]/10.
                np.testing.assert_allclose(d['valve_normalized'][1:],expected_valve,rtol=0.,atol=2e-7)
                peak=np.max(d['position_m'])
                overshoot=100*max(0.,peak-cfg['step_amplitude_m'])/cfg['step_amplitude_m']
                self.assertAlmostEqual(m['peak_position_mm'],peak*1000,places=10)
                self.assertAlmostEqual(m['overshoot_percent'],overshoot,places=10)
                self.assertAlmostEqual(m['max_internal_voltage_v'],np.max(d['raw_voltage_v']),places=10)
                self.assertAlmostEqual(m['final_error_mm'],d['true_error_m'][-1]*1000,places=10)
                step=t>=cfg['step_time']-1e-12
                crossings=np.flatnonzero(step & (d['true_error_m']<=0.))
                expected_cross=float(t[crossings[0]]) if crossings.size else None
                self.assertEqual(m['first_nonpositive_true_error_sample_s'],expected_cross)
                # Future metadata may add the sampled controller-error crossing;
                # it remains distinct from interpolated true-position crossing.
                if 'first_nonpositive_sampled_error_s' in m:
                    measured_crossings=np.flatnonzero(step & (d['sampled_error_m']<=0.))
                    measured_cross=float(t[measured_crossings[0]]) if measured_crossings.size else None
                    self.assertEqual(m['first_nonpositive_sampled_error_s'],measured_cross)
                positive=np.flatnonzero(step & (d['voltage_v']>=limit-1e-9))
                entry=int(positive[0]) if positive.size else None
                self.assertEqual(m['first_positive_saturation_s'],float(t[entry]) if entry is not None else None)
                exits=np.flatnonzero((np.arange(len(t))>entry)&(d['voltage_v']<limit-1e-9)) if entry is not None else np.array([])
                self.assertEqual(m['initial_positive_saturation_exit_s'],float(t[exits[0]]) if exits.size else None)
                comparison=summary['reference_comparison'][kind]
                for name,item in comparison.items():
                    self.assertEqual(item['actual'],m[name])
                pair[kind]=(cfg,d)
        ca,da=pair['positional'];cb,db=pair['incremental']
        for name in ca:
            if name not in ('controller','radau_nfev','release_count','restick_count'):
                self.assertEqual(ca[name],cb[name],name)
        np.testing.assert_array_equal(da['time_s'],db['time_s'])
        np.testing.assert_array_equal(da['reference_m'],db['reference_m'])
        before=da['time_s']<ca['step_time']-1e-12
        for name in ('position_m','velocity_m_s','p1_pa','p2_pa','valve_normalized','voltage_v'):
            np.testing.assert_array_equal(da[name][before],db[name][before])


if __name__=='__main__':
    unittest.main(verbosity=2)
