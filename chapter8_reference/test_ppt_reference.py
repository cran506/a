"""教学示意必须明确标记，并保持真实训练日志原样。"""
import json
from pathlib import Path
import tempfile
import unittest
import matplotlib.pyplot as plt
from ppt_training_reference import draw_ppt_panel,generate_ppt_reference


class TeachingReferenceTests(unittest.TestCase):
    def test_schematic_preserves_measured_csv_and_marks_its_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs').mkdir()
            log=root/'outputs/training.csv';before=b'timesteps,reward\n1000,-123.45\n';log.write_bytes(before)
            generate_ppt_reference(root)
            self.assertEqual(log.read_bytes(),before)
            meta=json.loads((root/'outputs/ppt_training_target.json').read_text())
            self.assertFalse(meta['contains_measured_training_data'])
            self.assertFalse(meta['random_jitter_added'])
            self.assertEqual(meta['source']['page'],12)

    def test_embedded_main_panel_is_labelled_as_illustration(self):
        fig,ax=plt.subplots();right=draw_ppt_panel(ax)
        self.assertIn('非训练日志',ax.get_title())
        self.assertIn('教学示意',ax.get_ylabel())
        self.assertIn('教学示意',right.get_ylabel());plt.close(fig)


if __name__=='__main__':unittest.main()
