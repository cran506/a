"""Integer representation and backward-loading checks for the W8/A16 export."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from edge_exporter import EdgeLightweightExporter,_quantize


class MixedPrecisionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(8186)
        self.cal=rng.normal(0.,.4,(100,5)).astype(np.float32)
        self.weights=[rng.normal(0.,.2,(8,5)).astype(np.float32),
                      rng.normal(0.,.2,(1,8)).astype(np.float32)]
        self.biases=[np.full(8,.1,np.float32),np.array([.02],np.float32)]

    def build(self,bits):
        return EdgeLightweightExporter(self.weights,self.biases,self.cal,prune_ratio=.70,
            quantization_mode='per_channel',equalize_hidden=True,final_int32=True,
            activation_bits=bits)

    def test_w8a16_uses_integer_activations_and_reloads_exactly(self):
        edge=self.build(16);first=edge.integer_layers[0]
        q=_quantize(edge._balanced_input(self.cal),first.input_scale,first.input_zero,16)
        self.assertEqual(q.dtype,np.int16)
        for index,layer in enumerate(edge.integer_layers):
            self.assertEqual(layer.weights.dtype,np.int8)
            self.assertEqual(layer.bias.dtype,np.int32)
            q=layer.forward(q)
            self.assertEqual(q.dtype,np.int16 if index==0 else np.int32)
        with tempfile.TemporaryDirectory() as directory:
            edge.save(directory)
            path=Path(directory)/'actor_int8.npz'
            with np.load(path) as saved:
                metadata=json.loads(str(saved['metadata_json']))
            self.assertEqual(metadata['format_version'],4)
            self.assertEqual(metadata['activation_bits'],16)
            np.testing.assert_array_equal(edge.action(self.cal),
                EdgeLightweightExporter.load_int8(path)(self.cal))

    def test_old_w8a8_artifact_without_new_metadata_still_loads(self):
        edge=self.build(8)
        with tempfile.TemporaryDirectory() as directory:
            edge.save(directory);path=Path(directory)/'actor_int8.npz'
            with np.load(path) as saved:arrays={name:saved[name].copy() for name in saved.files}
            metadata=json.loads(str(arrays['metadata_json']))
            metadata.pop('activation_bits')
            for layer in metadata['layers']:layer.pop('activation_bits')
            arrays['metadata_json']=np.array(json.dumps(metadata))
            np.savez_compressed(path,**arrays)
            np.testing.assert_array_equal(edge.action(self.cal),
                EdgeLightweightExporter.load_int8(path)(self.cal))

    def test_invalid_width_and_unsafe_dot_product_are_rejected(self):
        with self.assertRaises(ValueError):self.build(12)
        with self.assertRaises(ValueError):
            EdgeLightweightExporter(self.weights,self.biases,self.cal,activation_bits=16)
        # At full activation range a very wide dense dot product cannot safely
        # fit INT32, even when these calibration samples are small.
        width=300
        weights=[np.ones((1,width),np.float32)]
        with self.assertRaisesRegex(ValueError,'overflow'):
            EdgeLightweightExporter(weights,[np.zeros(1,np.float32)],
                np.vstack([np.zeros(width),np.ones(width)]).astype(np.float32),
                prune_ratio=0.,quantization_mode='per_channel',activation_bits=16)


if __name__=='__main__':unittest.main()
