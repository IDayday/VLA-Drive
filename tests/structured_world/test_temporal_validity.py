import unittest
import numpy as np
import torch
from starVLA.dataloader.structured_world.temporal_validity import mask_auxiliary_time_mismatch


class TemporalValidity(unittest.TestCase):
    def test_native_navsim_ego_remains_complete_when_scene_timestamp_is_missing(self):
        from starVLA.dataloader.foresight_dataset import encode_ego
        from starVLA.dataloader.structured_world.dataset import validate_native_ego_contract
        physical = np.column_stack((np.arange(1, 9), np.zeros(8), np.zeros(8))).astype(np.float32)
        ego = torch.from_numpy(encode_ego(physical)); before = ego.clone()
        auxiliary = physical.copy(); auxiliary[1:] = 0.
        valid = np.array([True, False, False, False, False, False, False, False])
        self.assertTrue(validate_native_ego_contract(ego, auxiliary, valid).all())
        self.assertTrue(torch.equal(before, ego))
        auxiliary[0, 0] += 1.
        with self.assertRaises(ValueError): validate_native_ego_contract(ego, auxiliary, valid)

    def test_auxiliary_timestamp_mismatch_never_becomes_free_or_removes_ego(self):
        arrays = {'occupancy': np.ones((7, 2, 3), np.uint8), 'occupancy_valid': np.ones((7, 2, 3), bool),
            'instances': np.ones((7, 2, 3), np.int64), 'future_valid': np.ones(6, bool),
            'ego_physical': np.arange(18, dtype=np.float32).reshape(6, 3)}
        original_ego = arrays['ego_physical'].copy()
        valid = mask_auxiliary_time_mismatch(arrays, [.501, 1.1, 1.45, 2.3, 2.5, 3.04])
        self.assertEqual(valid.tolist(), [True, False, True, False, True, True])
        for frame in (2, 4):
            self.assertTrue((arrays['occupancy'][frame] == 255).all())
            self.assertFalse(arrays['occupancy_valid'][frame].any())
        self.assertTrue(arrays['future_valid'].all())
        np.testing.assert_array_equal(arrays['ego_physical'], original_ego)
        self.assertTrue((arrays['occupancy'][0] == 1).all())


if __name__ == '__main__': unittest.main()
