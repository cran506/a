"""给定教材参数的代数验算；测试的期望值由量纲和公式独立计算。"""
import json
import math
from pathlib import Path
import unittest

from textbook_controller import ModelAssistedController
from textbook_trajectory import generate_textbook_trajectory


PARAMS = json.loads(Path(__file__).with_name('textbook_config.json').read_text(encoding='utf-8'))
PID = dict(kp=0., ki=0., kd=0., ts=.001, alpha=.85, integral_separation=False)


class TextbookReferenceTests(unittest.TestCase):
    def test_reference_analytic_start_and_cruise(self):
        t, x, v, a = generate_textbook_trajectory()
        self.assertEqual(float(a[0]), .2)
        self.assertAlmostEqual(float(v[160]), .032, places=12)
        self.assertAlmostEqual(float(v[350]), .2, places=12)
        self.assertAlmostEqual(float(v[500]), .2, places=12)
        self.assertAlmostEqual(float(v[1000]), 0., places=12)
        self.assertTrue(all(x[k+1] >= x[k] for k in range(len(x)-1)))

    def test_first_forward_sample_follows_supplied_force_and_leakage(self):
        c = ModelAssistedController(PID, PARAMS)
        voltage, feed, feedback = c.update((0., 0., .2), (0., 0.), (5e6, 10e6))
        self.assertEqual(feedback, 0.)
        expected_force = (PARAMS['m']*.2 + PARAMS['Fc'] +
                          (PARAMS['Fs']-PARAMS['Fc'])*math.exp(-(1e-5/PARAMS['vs'])**2))
        self.assertAlmostEqual(c.last['F_drive_n'], expected_force, places=9)
        self.assertAlmostEqual(c.last['P_req_est_pa'], expected_force/PARAMS['A1'], places=6)
        self.assertAlmostEqual(voltage, 1.2, places=6)
        self.assertEqual(voltage, feed)

    def test_heavy_voltage_is_computed_from_given_pressure(self):
        p = PARAMS
        light = ModelAssistedController(PID, p)
        heavy = ModelAssistedController(PID, p)
        nominal = ModelAssistedController(PID, p, enable_gain_sched=False)
        reference = (.055, .2, 0.)
        yl = (.055, .2)
        ul, *_ = light.update(reference, yl, (5e6, 10e6))
        uh, *_ = heavy.update(reference, yl, (16.95e6, 10e6))
        un, *_ = nominal.update(reference, yl, (16.95e6, 10e6))
        force = p['sigma']*.2 + p['Fc'] + (p['Fs']-p['Fc'])*math.exp(-(.2/p['vs'])**2)
        leak_voltage = p['Ct']*(force/p['A1'])/(p['K_flow']*math.sqrt(p['Ps']-p['P1_0']))
        expected_light = p['deadzone'] + p['A1']*.2/(p['K_flow']*math.sqrt(16e6)) + leak_voltage
        expected_heavy = p['deadzone'] + p['A1']*.2/(p['K_flow']*math.sqrt(4.05e6)) + leak_voltage
        self.assertAlmostEqual(ul, expected_light, places=10)
        self.assertAlmostEqual(uh, expected_heavy, places=10)
        self.assertAlmostEqual(un, expected_light, places=10)
        self.assertGreater(uh, 4.3)
        self.assertLess(uh, 4.4)

    def test_negative_motion_uses_rod_side_pressure(self):
        p = PARAMS
        low = ModelAssistedController(PID, p)
        high = ModelAssistedController(PID, p)
        r = (-.01, -.1, -.2)
        y = (-.01, -.1)
        u1, *_ = low.update(r, y, (5e6, 10e6))
        u2, *_ = high.update(r, y, (5e6, 17e6))
        self.assertLess(u1, 0.)
        self.assertLess(u2, u1)


if __name__ == '__main__':
    unittest.main()
