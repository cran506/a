"""核对课件图(a)对应的训练初始化、统计和验证来源。"""
import csv
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from stable_baselines3 import SAC

ROOT=Path(__file__).resolve().parent


class TrainingProtocolTests(unittest.TestCase):
    def test_untrained_policy_has_no_teacher_action_mean(self):
        model=SAC.load(ROOT/'models/sac_untrained.zip',device='cpu')
        np.testing.assert_array_equal(model.actor.mu.weight.detach().numpy(),0.)
        np.testing.assert_array_equal(model.actor.mu.bias.detach().numpy(),0.)
        np.testing.assert_array_equal(model.actor.log_std.weight.detach().numpy(),0.)
        self.assertAlmostEqual(float(model.actor.log_std.bias.detach()[0]),-.1,places=6)

    def test_exact_final_checkpoint_without_offline_mean_replacement(self):
        meta=json.loads((ROOT/'models/training_metadata.json').read_text())
        self.assertFalse(meta['teacher_initialization'])
        self.assertFalse(meta['teacher_prior'])
        self.assertFalse(meta['post_training_teacher_consolidation'])
        self.assertEqual(meta['actual_timesteps'],1000000)
        self.assertEqual(meta['deployed_model_sha256'],hashlib.sha256((ROOT/'models/sac_model.zip').read_bytes()).hexdigest())

    def test_constant_window_statistic_and_entropy_coordinate(self):
        with (ROOT/'outputs/training.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len({row['reward_source'] for row in rows}),1)
        self.assertTrue(all(int(row['reward_window_count'])==10000 for row in rows[1:]))
        self.assertTrue(all(int(row['updates'])==0 for row in rows if int(row['timesteps'])<=50000))
        for row in rows:
            self.assertAlmostEqual(float(row['raw_reward_mean']),float(row['reward_mean'])+float(row['terminal_penalty_mean']),places=7)
            self.assertAlmostEqual(float(row['voltage_entropy'])-float(row['normalized_action_entropy']),np.log(10.),places=7)

    def test_validation_is_fixed_physical_task_not_training_returns(self):
        meta=json.loads((ROOT/'outputs/training_validation_metadata.json').read_text())
        self.assertFalse(meta['training_reward_replacement'])
        self.assertTrue(meta['validation_not_added_to_replay'])
        self.assertEqual(meta['actual_extra_validation_interactions'],11*16*50)
        for name,digest in meta['checkpoint_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
