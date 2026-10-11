"""CPU controls for the standalone proposal tool; never initializes CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from tools.structured_world import profile_proposal_execution as profile


class ProfileTests(unittest.TestCase):
    def test_actual_default_workload_is_balanced_and_alternates(self):
        warm, timed, configuration = [], [], []
        def measure(mode):
            timed.append(mode)
            return {'wall_ms': 50. if mode == 'IO' else 29., 'CUDA_ms': 45. if mode == 'IO' else 24.}
        rows = profile.windows(configuration.append, warm.append, measure)
        self.assertEqual([row['first_mode'] for row in rows], ['IO', 'KV', 'IO'])
        self.assertEqual([warm.count(mode) for mode in ('IO', 'KV')], [60, 60])
        self.assertEqual([timed.count(mode) for mode in ('IO', 'KV')], [150, 150])
        self.assertEqual(timed[:4], ['IO', 'KV', 'IO', 'KV'])
        self.assertEqual(timed[100:104], ['KV', 'IO', 'KV', 'IO'])
        self.assertEqual(profile.summary(rows)['per_window_wall_saving_ms'], [21., 21., 21.])

    def test_four_medians_and_all_window_20ms_gate(self):
        rows = [{'IO': [{'wall_ms': 100., 'CUDA_ms': 90.}],
                 'KV': [{'wall_ms': value, 'CUDA_ms': value-10.}]} for value in (70., 75., 85.)]
        summary = profile.summary(rows)
        self.assertEqual([summary[key] for key in ('IO_wall_median_ms', 'KV_wall_median_ms',
                                                  'IO_CUDA_median_ms', 'KV_CUDA_median_ms')], [100., 75., 90., 65.])
        self.assertFalse(summary['every_window_saves_at_least_20ms'])
        with self.assertRaises(ValueError): profile.summary([])

    def test_fixed_head_digest_binds_dtype_shape_names_and_exact_bytes(self):
        value = torch.tensor([0., -0., 1.])
        digest = profile.tensor_digest({'weights': value, 'scalar': torch.tensor(1.)})
        self.assertEqual(digest, profile.tensor_digest({'scalar': torch.tensor(1.), 'weights': value.clone()}))
        self.assertNotEqual(digest, profile.tensor_digest({'weights': value.reshape(1, 3), 'scalar': torch.tensor(1.)}))
        self.assertNotEqual(digest, profile.tensor_digest({'weights': value.double(), 'scalar': torch.tensor(1.)}))
        self.assertNotEqual(digest, profile.tensor_digest({'renamed': value, 'scalar': torch.tensor(1.)}))
        value[1] = 0.
        self.assertNotEqual(digest, profile.tensor_digest({'weights': value, 'scalar': torch.tensor(1.)}))
        self.assertFalse(profile.exact(torch.tensor([-0.]), torch.tensor([0.])))
        self.assertTrue(profile.exact(torch.tensor([1.], dtype=torch.bfloat16), torch.tensor([1.], dtype=torch.bfloat16)))

    def test_native_tiny_head_hash_and_inventory_after_opt_in_scope(self):
        from tools.structured_world.test_proposal_execution import make_head, inputs
        from starVLA.model.modules.structured_world import proposal_execution as adapter
        head = make_head()
        memory, noise = inputs()
        before = profile.inventory(head)
        digest = profile.tensor_digest(head.state_dict())
        with torch.no_grad(): expected = head.predict_action(memory, initial_noise=noise)
        adapter.install_proposal_kv_cache(head, enabled=True)
        with torch.no_grad(): actual = head.predict_action(memory, initial_noise=noise)
        adapter.uninstall_proposal_kv_cache(head)
        self.assertTrue(profile.exact(expected, actual))
        self.assertEqual(before, profile.inventory(head))
        self.assertEqual(digest, profile.tensor_digest(head.state_dict()))
        next(head.qwen_proj.parameters()).detach().add_(.1)
        self.assertNotEqual(digest, profile.tensor_digest(head.state_dict()))

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
