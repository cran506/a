"""防止评估重写训练来源，核对真实保持电压与故障边界。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluation import evaluate,run_policy,trajectory_metrics
from hydraulic_twin import HydraulicTwinCore


class EvaluationTests(unittest.TestCase):
    def test_changed_physics_rejected_without_rewriting_training_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'models').mkdir()
            training={'physical_config_sha256':hashlib.sha256(b'original').hexdigest()}
            metadata=root/'models/training_metadata.json'
            metadata.write_text(json.dumps(training));before=metadata.read_bytes()
            (root/'physical_config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'训练后已改变'):evaluate(root)
            self.assertEqual(before,metadata.read_bytes())

    def test_logged_voltage_replays_native_flow_and_fault_clock(self):
        rows,obs,info=run_policy('nominal_ff')
        self.assertEqual((len(rows),obs.shape),(601,(601,5)))
        self.assertEqual(rows[0]['previous_control_v'],0.)
        self.assertEqual(rows[99]['load_n'],0.)
        self.assertEqual(rows[100]['load_n'],5000.)
        self.assertEqual(rows[100]['beta_mpa'],500.)
        twin=HydraulicTwinCore()
        twin.reset(np.array([rows[0]['position_mm']/1000,rows[0]['velocity_mms']/1000,
                             rows[0]['p1_mpa']*1e6,rows[0]['p2_mpa']*1e6]))
        for before,after in zip(rows[:-1],rows[1:]):
            state,_=twin.step(before['control_v'],load_n=before['load_n'],
                             beta_multiplier=before['beta_mpa']/1000.)
            expected=np.array([after['position_mm']/1000,after['velocity_mms']/1000,
                               after['p1_mpa']*1e6,after['p2_mpa']*1e6])
            np.testing.assert_allclose(state,expected,rtol=1e-12,atol=1e-8)

    def test_recovery_requires_final_persistent_band(self):
        rows=[]
        for index in range(151):
            error=-5. if index<10 else (2. if index==60 else 0.)
            rows.append(dict(time_s=.5+index*.001,velocity_error_mms=error,
                             control_v=3.,position_error_mm=0.))
        self.assertAlmostEqual(trajectory_metrics(rows)['recovery_1mms_s'],.061)
        rows[-1]['velocity_error_mms']=2.
        self.assertIsNone(trajectory_metrics(rows)['recovery_1mms_s'])


if __name__=='__main__':unittest.main()
