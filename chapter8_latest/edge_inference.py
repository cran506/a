"""Standalone integer actor inference: no PyTorch, Gymnasium or SB3 import."""
import argparse
import json
from pathlib import Path
import numpy as np
from edge_exporter import EdgeLightweightExporter


class EdgeController:
    def __init__(self, models_dir=None):
        directory=Path(models_dir or Path(__file__).resolve().parent/'models')
        metadata=json.loads((directory/'training_metadata.json').read_text())
        self.names=metadata['observation_names']
        self.mean=np.array(metadata['obs_mean'],dtype=np.float32)
        self.std=np.array(metadata['obs_std'],dtype=np.float32)
        self.actor=EdgeLightweightExporter.load_int8(directory/'actor_int8.npz')

    def action_volts(self,raw_observation):
        obs=np.asarray(raw_observation,dtype=np.float32)
        if obs.shape!=(10,) or not np.isfinite(obs).all():
            raise ValueError('Expected ten finite raw observations in metadata units')
        normalized=(obs-self.mean)/(self.std+1e-8)
        return float(self.actor(normalized).reshape(-1)[0])*10.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obs',required=True,help='JSON list of ten raw sensor/reference values')
    parser.add_argument('--models-dir',type=Path)
    args=parser.parse_args()
    controller=EdgeController(args.models_dir)
    print(json.dumps({'observation_names':controller.names,
                      'control_v':controller.action_volts(json.loads(args.obs))},ensure_ascii=False))
