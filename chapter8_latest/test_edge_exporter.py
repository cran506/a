import sys
import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
from edge_exporter import (EdgeLightweightExporter, magnitude_prune, IntegerLayer,
                           PerChannelIntegerLayer,equalize_relu_channels)


class EdgeTests(unittest.TestCase):
    def test_equal_magnitude_rank_prunes_exact_count(self):
        w, mask = magnitude_prune(np.ones((10, 10)), .7)
        self.assertEqual(np.count_nonzero(w == 0), 70)
        self.assertEqual(np.count_nonzero(~mask), 70)

    def test_zero_representable_for_one_sided_and_zero_weights(self):
        for value in (0., 3., -3.):
            w = np.full((10, 10), value)
            dequant, q, scale, zero, sparsity = EdgeLightweightExporter.prune_and_quantize_weights(w, .7)
            pruned, mask = magnitude_prune(w, .7)
            np.testing.assert_array_equal(dequant[pruned == 0.], 0.)
            np.testing.assert_allclose(dequant[mask], pruned[mask], atol=scale)

    def test_integer_centering_no_int8_wrap(self):
        layer = IntegerLayer(np.array([[127]], dtype=np.int8), np.array([0], dtype=np.int32),
                             1., -128, 1., -128, 1000., 0, 1, 0, False)
        result = layer.forward(np.array([127], dtype=np.int8))
        self.assertEqual(int(result[0]), 127)  # 255*255 is large positive, not 1.

    def make_exporter(self, ratio=.7, balance_inputs=True, **kwargs):
        rng = np.random.default_rng(12)
        weights = [rng.normal(0., .2, (64, 5)).astype('f'),
                   rng.normal(0., .2, (64, 64)).astype('f'),
                   rng.normal(0., .2, (1, 64)).astype('f')]
        biases = [rng.normal(0., .02, size=w.shape[0]).astype('f') for w in weights]
        obs = rng.uniform(-1., 1., (1024, 5)).astype('f')
        return EdgeLightweightExporter(weights, biases, obs, ratio,
                                       balance_inputs=balance_inputs, **kwargs), obs

    def test_balanced_sparse_float_network_is_equivalent(self):
        # Different channel ranges and a nonzero constant channel make missing
        # centering, the wrong bias sign or re-pruning the folded weights fail.
        rng = np.random.default_rng(91)
        obs = rng.normal(size=(500, 4)).astype(np.float32)
        obs = obs * np.array([8., .03, 2., 0.], np.float32) + np.array([12., -.3, 4., 2.5], np.float32)
        weights = [rng.normal(0., .08, (12, 4)).astype(np.float32),
                   rng.normal(0., .1, (1, 12)).astype(np.float32)]
        biases = [rng.normal(0., .01, w.shape[0]).astype(np.float32) for w in weights]
        edge = EdgeLightweightExporter(weights, biases, obs, .7)
        folded_w, folded_b = edge._integer_float_parameters()
        balanced = edge._balanced_input(obs)
        equivalent = edge._float_forward(balanced, folded_w, folded_b)
        np.testing.assert_allclose(equivalent, edge.action(obs, 'pruned_fp32'), atol=5e-7, rtol=2e-5)
        np.testing.assert_array_equal(folded_w[0] == 0, edge.pruned_weights[0] == 0)
        self.assertEqual(float(edge.input_channel_scale[-1]), 1.)
        # Neither original FP32 path is changed by the integer-only prefix.
        plain = EdgeLightweightExporter(weights, biases, obs, .7, balance_inputs=False)
        np.testing.assert_array_equal(edge.action(obs, 'fp32'), plain.action(obs, 'fp32'))
        np.testing.assert_array_equal(edge.action(obs, 'pruned_fp32'), plain.action(obs, 'pruned_fp32'))

    def test_smoothquant_zero_weight_column_is_finite_and_constant_channel_kept(self):
        rng = np.random.default_rng(239)
        obs = rng.normal(size=(300, 3)).astype(np.float32)
        obs[:, 2] = 4.
        weight = np.array([[0., .4, .3], [0., -.5, .1]], np.float32)
        edge = EdgeLightweightExporter([weight], [np.array([.2, -.1], np.float32)],
                                       obs, 0., smoothquant_alpha=.25)
        self.assertTrue(np.isfinite(edge.input_channel_scale).all())
        self.assertTrue(np.all(edge.input_channel_scale > 0))
        self.assertEqual(float(edge.input_channel_scale[-1]), 1.)
        folded_w, folded_b = edge._integer_float_parameters()
        np.testing.assert_array_equal(folded_w[0][:, 0], 0.)
        np.testing.assert_allclose(edge._float_forward(edge._balanced_input(obs), folded_w, folded_b),
                                   edge.action(obs, 'pruned_fp32'), atol=5e-7)
        report = edge.validation_report(obs)
        self.assertEqual(report['input_balancing_mode'], 'smoothquant')
        self.assertEqual(report['smoothquant_alpha'], .25)

    def test_mean_std_mode_retains_version_two_behavior(self):
        edge, obs = self.make_exporter(balance_mode='mean_std')
        np.testing.assert_array_equal(edge.input_center, obs.mean(axis=0, dtype=np.float64).astype(np.float32))
        np.testing.assert_array_equal(edge.input_channel_scale, obs.std(axis=0, dtype=np.float64).astype(np.float32))
        with tempfile.TemporaryDirectory() as path:
            edge.save(path)
            predict = edge.load_int8(Path(path)/'actor_int8.npz')
            np.testing.assert_array_equal(predict(obs), edge.action(obs))

    def test_integer_quantization_error_is_separate_from_pruning(self):
        e, obs = self.make_exporter()
        report = e.validation_report(obs)
        self.assertLess(report['int8_vs_pruned_action_mae'], .015)
        self.assertGreater(report['actual_weight_sparsity'], .699)
        self.assertLess(report['actual_weight_sparsity'], .701)

    def test_standalone_int8_artifact_equal_output(self):
        e, obs = self.make_exporter()
        with tempfile.TemporaryDirectory() as path:
            sizes = e.save(path)
            predict = e.load_int8(Path(path) / 'actor_int8.npz')
            np.testing.assert_array_equal(predict(obs), e.action(obs))
            with np.load(Path(path) / 'actor_int8.npz', allow_pickle=False) as archive:
                self.assertEqual(archive['weight_0'].dtype, np.int8)
                self.assertEqual(archive['bias_0'].dtype, np.int32)
                np.testing.assert_array_equal(archive['input_center'], e.input_center)
                np.testing.assert_array_equal(archive['input_channel_scale'], e.input_channel_scale)
                self.assertEqual(json.loads(str(archive['metadata_json']))['format_version'], 2)
                self.assertEqual(json.loads(str(archive['metadata_json']))['input_balancing_mode'], 'smoothquant')
            self.assertLess(sizes['actor_int8.npz'], sizes['actor_fp32.npz'])

    def test_legacy_artifact_without_affine_prefix_uses_identity(self):
        edge, obs = self.make_exporter(balance_inputs=False)
        with tempfile.TemporaryDirectory() as path:
            edge.save(path)
            filename = Path(path) / 'actor_int8.npz'
            with np.load(filename, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files
                          if key not in {'input_center', 'input_channel_scale'}}
            metadata = json.loads(str(arrays['metadata_json']))
            metadata['format_version'] = 1
            metadata.pop('input_balancing')
            arrays['metadata_json'] = np.array(json.dumps(metadata))
            np.savez_compressed(filename, **arrays)
            predict = edge.load_int8(filename)
            np.testing.assert_array_equal(predict(obs), edge.action(obs))

    def test_invalid_serialized_input_scale_is_rejected(self):
        edge, _ = self.make_exporter()
        with tempfile.TemporaryDirectory() as path:
            edge.save(path)
            filename = Path(path) / 'actor_int8.npz'
            with np.load(filename, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays['input_channel_scale'][0] = 0.
            np.savez_compressed(filename, **arrays)
            with self.assertRaises(ValueError):
                edge.load_int8(filename)

    def test_nonfinite_calibration_rejected(self):
        with self.assertRaises(ValueError):
            EdgeLightweightExporter([np.ones((1, 1))], [np.zeros(1)], [[np.nan]])

    def test_relu_channel_equalization_preserves_function_and_pruning_mask(self):
        edge,obs=self.make_exporter()
        weights,biases,scales=equalize_relu_channels(edge.pruned_weights,edge.pruned_biases,obs)
        self.assertEqual(len(scales),len(weights)-1)
        for original,changed in zip(edge.pruned_weights,weights):
            np.testing.assert_array_equal(original==0.,changed==0.)
        np.testing.assert_allclose(edge._float_forward(obs,weights,biases),
                                   edge.action(obs,'pruned_fp32'),atol=5e-7,rtol=2e-5)

    def test_per_channel_int32_head_and_version_three_roundtrip(self):
        edge,obs=self.make_exporter(quantization_mode='per_channel',
                                    equalize_hidden=True,final_int32=True)
        first=edge.integer_layers[0]
        self.assertIsInstance(first,PerChannelIntegerLayer)
        self.assertEqual(first.weight_scale.shape,(64,))
        q=np.array([[127,-128]],dtype=np.int8)
        layer=PerChannelIntegerLayer(np.array([[127,-127],[64,0]],dtype=np.int8),
              np.array([5,-10],dtype=np.int32),.1,-128,np.array([.01,.02]),
              .1,0,np.array([1,2],dtype=np.int64),0,False,True)
        np.testing.assert_array_equal(layer.forward(q),[[32390,16310]])
        self.assertEqual(layer.forward(q).dtype,np.int32)
        with tempfile.TemporaryDirectory() as path:
            edge.save(path)
            filename=Path(path)/'actor_int8.npz'
            restored=edge.load_int8(filename)
            np.testing.assert_array_equal(restored(obs),edge.action(obs))
            with np.load(filename,allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata_json']))
                self.assertEqual(metadata['format_version'],3)
                self.assertTrue(metadata['final_int32_logits'])
                self.assertEqual(archive['weight_0'].dtype,np.int8)
                self.assertEqual(archive['bias_0'].dtype,np.int32)
                self.assertTrue(np.all(archive['weight_scale_0']>0))
            report=edge.validation_report(obs)
            self.assertEqual(report['weight_quantization'],'per_channel')
            self.assertGreater(report['actual_weight_sparsity'],.699)

    def test_numpy_polishing_uses_fixed_mask_and_reduces_calibration_loss(self):
        rng=np.random.default_rng(23)
        obs=rng.normal(size=(256,4)).astype(np.float32)
        obs[:,2]=obs[:,0];obs[:,3]=0.
        weights=[np.array([[.9,.7,.3,.1]],np.float32)]
        edge=EdgeLightweightExporter(weights,[np.zeros(1,np.float32)],obs,.5)
        teacher=edge.weights[0].copy()
        result=edge.polish_pruned(obs,steps=500,batch_size=64,
                                  learning_rate=.003,min_learning_rate=.0003,seed=7)
        self.assertLess(result['final_action_mse'],result['initial_action_mse']*.5)
        np.testing.assert_array_equal(edge.weights[0],teacher)
        self.assertTrue(np.all(edge.pruned_weights[0][~edge.masks[0]]==0.))
        self.assertEqual(result['selection_metric'],'full calibration action MSE only')

    def test_requantization_rejects_int32_overflow(self):
        with self.assertRaises(ValueError):
            EdgeLightweightExporter([np.ones((1, 33026))], [np.zeros(1)], np.zeros((1, 33026)), 0.)

if __name__ == '__main__':
    unittest.main()
