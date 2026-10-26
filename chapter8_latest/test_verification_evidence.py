"""Failed/stale cached numerical files must not certify the current model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_experiment import (PHYSICS_SOURCE_FILES, _file_hashes,
                               independent_evidence_checks)


class IndependentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        (self.root/'outputs').mkdir()
        for name in PHYSICS_SOURCE_FILES:
            (self.root/name).write_text('fixture '+name,encoding='utf-8')
        self.safety=dict(passed=True,checks={'barrier':True},
            independent_random_validation=dict(passed=True,checks={'dense_flow':True}),
            minimum_safe_voltage_v=-.3)
        self.metrics={'safety':copy.deepcopy(self.safety)}
        self.normalization=dict(passed=True,checks={'coordinate_equivalence':True},
                                max_held_out_normalized_action_difference=0.)
        (self.root/'outputs/normalization_validation.json').write_text(
            json.dumps(self.normalization),encoding='utf-8')
        self._write_safety()
        self.physics=dict(passed=True,checks={'independent_radau':True},
            source_sha256=_file_hashes(self.root,PHYSICS_SOURCE_FILES),
            safety_evidence_sha256=_file_hashes(self.root,
                ('outputs/safety.csv','outputs/safety_summary.json')))
        self._write_physics()

    def _write_safety(self):
        (self.root/'outputs/safety.csv').write_text('time_s,x_safe_mm\n0,20\n',encoding='utf-8')
        (self.root/'outputs/safety_summary.json').write_text(json.dumps(self.safety),encoding='utf-8')

    def _write_physics(self):
        (self.root/'outputs/physics_validation.json').write_text(json.dumps(self.physics),encoding='utf-8')

    def checks(self):
        return independent_evidence_checks(self.root,self.metrics)[0]

    def test_valid_bound_evidence_passes(self):
        self.assertTrue(all(self.checks().values()))

    def test_existing_failed_physics_file_is_rejected(self):
        self.physics['checks']['independent_radau']=False
        self._write_physics()
        self.assertFalse(self.checks()['independent_physics_validation_passed'])

    def test_truthy_strings_cannot_certify_physics(self):
        self.physics['passed']='true'
        self._write_physics()
        self.assertFalse(self.checks()['independent_physics_validation_passed'])

    def test_changed_source_requires_new_independent_validation(self):
        (self.root/'hydraulic_twin.py').write_text('changed dynamics',encoding='utf-8')
        self.assertFalse(self.checks()['independent_validation_uses_current_sources'])

    def test_added_safety_configuration_requires_revalidation(self):
        (self.root/'safety_config.json').write_text('{"k_gain":2640}',encoding='utf-8')
        self.assertFalse(self.checks()['independent_validation_uses_current_sources'])

    def test_changed_safety_csv_is_rejected(self):
        (self.root/'outputs/safety.csv').write_text('replaced trajectory',encoding='utf-8')
        self.assertFalse(self.checks()['independent_validation_binds_current_safety_records'])

    def test_nested_safety_failure_and_metric_mismatch_are_rejected(self):
        self.safety['independent_random_validation']['checks']['dense_flow']=False
        self._write_safety()
        checks=self.checks()
        self.assertFalse(checks['independent_random_safety_checks_passed'])
        self.assertFalse(checks['safety_summary_matches_evaluation_metrics'])

    def test_normalization_false_checks_or_excessive_error_are_rejected(self):
        path=self.root/'outputs/normalization_validation.json'
        self.normalization['max_held_out_normalized_action_difference']=1e-4
        path.write_text(json.dumps(self.normalization),encoding='utf-8')
        self.assertFalse(self.checks()['empirical_normalization_coordinate_error_below_1e_minus8'])
        self.normalization['max_held_out_normalized_action_difference']=0.
        self.normalization['checks']['coordinate_equivalence']=False
        path.write_text(json.dumps(self.normalization),encoding='utf-8')
        self.assertFalse(self.checks()['empirical_normalization_validation_passed'])

    def test_missing_or_malformed_evidence_is_rejected(self):
        (self.root/'outputs/physics_validation.json').write_text('{broken',encoding='utf-8')
        self.assertFalse(self.checks()['independent_physics_validation_passed'])
        (self.root/'outputs/safety_summary.json').unlink()
        checks=self.checks()
        self.assertFalse(checks['independent_sampled_safety_passed'])
        self.assertFalse(checks['independent_validation_binds_current_safety_records'])


if __name__=='__main__':unittest.main()
