import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

from scripts.analysis.trajectory_repair.audit_feedback import audit
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint


class AuditIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'TRAIN').mkdir()
        (self.root / 'DEV_SELECT').mkdir()
        self.path = self.root / 'TRAIN/train.pt'
        self.index = dict(TRAIN=[str(self.path)], DEV_SELECT=[str(self.root / 'DEV_SELECT/dev.pt')],
                          base_model_sha256='base', protocol_sha256='protocol')
        f = dict(status='OK', requested_trajectory=np.zeros((8, 3)), simulated_states=np.zeros((41, 11)),
                 non_drivable_body_by_time=np.zeros(41), protocol_sha256='scorer',
                 metrics=dict(no_at_fault_collisions=1., drivable_area_compliance=1.,
                              time_to_collision_within_bound=1., ego_progress=1., comfort=1.,
                              driving_direction_compliance=1., score=1.))
        row = dict(split='TRAIN', scene_token='train', log_id='log', status='OK',
                   checkpoint_id='base', raw_reward=1., feedback=f, scorer_config_hash='scorer',
                   planned_trajectory=np.zeros((8, 3)))
        torch.save(dict(split='TRAIN', scene_token='train', protocol_sha256='protocol',
                        checkpoint_id='base', log_id='log', reference_records=[row], candidate_records=[row]), self.path)
        self.path.with_suffix('.json').write_text(json.dumps(dict(status='COMPLETE', split='TRAIN',
            protocol_sha256='protocol', checkpoint_id='base', payload_sha256=file_sha(self.path))))
        self.write_index()

    def write_index(self):
        self.index.pop('sha256', None)
        self.index['sha256'] = fingerprint(self.index)
        (self.root / 'index_TRAIN.json').write_text(json.dumps(self.index))

    def test_does_not_load_development_records(self):
        # DEV file deliberately does not exist. Audit can read index membership
        # for leakage checks, but never opens DEV labels or actor features.
        result = audit(self.root)
        self.assertEqual(result['counts']['scenes'], 1)
        self.assertEqual(result['counts']['candidates'], 2)
        self.assertEqual(result['new_scorer_calls'], 0)

    def test_cross_split_token_rejected(self):
        self.index['DEV_SELECT'] = [str(self.root / 'DEV_SELECT/train.pt')]
        self.write_index()
        with self.assertRaises(ValueError):
            audit(self.root)

    def test_index_cannot_escape_declared_train_directory(self):
        self.index['TRAIN'] = [str(self.root / 'DEV_SELECT/train.pt')]
        self.write_index()
        with self.assertRaises(ValueError):
            audit(self.root)

    def test_changed_payload_rejected_before_deserialization(self):
        with self.path.open('ab') as stream:
            stream.write(b'changed')
        with mock.patch('torch.load') as load:
            with self.assertRaises(ValueError):
                audit(self.root)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
