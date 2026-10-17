"""CPU-only behavior checks on real runtime functions and explicit toy tensors."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_pair as pair


class PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = pair.load_metadata()
        cls.configs = pair.expected_pair(cls.metadata)

    def test_actual_configs_and_target_contracts_parse_without_dataset_or_model(self):
        configs = {c: pair.small_json(pair.CONFIGS / f'PCA{c}.json') for c in (256, 384)}
        self.assertEqual(pair.verify_pair(configs, self.metadata)['status'], 'CONFIGURATION_PASS')
        for c in configs:
            self.assertEqual(pair.small_json(pair.CONFIGS / f'TARGET_CONTRACT_{c}.json'), configs[c]['space_time']['teacher'])

    def test_384_is_byte_identical_existing_config_and_not_a_new_run(self):
        import hashlib
        pin = pair.small_json(pair.HERE / 'INPUT_PINS.json')['reference_config']
        self.assertEqual(hashlib.sha256((pair.CONFIGS / 'PCA384.json').read_bytes()).hexdigest(), pin['sha256'])
        self.assertEqual(self.configs[384], self.metadata['reference_config'])

    def test_only_width_derived_transform_and_isolated_output_differ(self):
        left, right = deepcopy(self.configs[256]), deepcopy(self.configs[384])
        for cfg in (left, right):
            cfg['space_time']['teacher'].pop('channels')
            cfg['space_time']['teacher'].pop('transform_identity')
            cfg['runtime'].pop('output_root')
        self.assertEqual(left, right)
        self.assertNotEqual(self.configs[256]['space_time']['teacher']['transform_identity'], self.configs[384]['space_time']['teacher']['transform_identity'])

    def test_reject_changed_scientific_factors_and_disguised_transform(self):
        changes = [('space_time.world_queries_per_view', 96), ('space_time.sampled_positions', 48),
                   ('space_time.teacher.channels', 128), ('space_time.teacher.output_grid_hw', [6, 8]),
                   ('space_time.teacher.image_hw', [192, 256]), ('space_time.current_weight', 2),
                   ('optimizer.lr', 2e-5), ('execution.stop_after', 1000),
                   ('execution.max_updates', 5000), ('execution.seed', 41),
                   ('data.pca_path', '/wrong-basis'), ('data.root', '/wrong-population'),
                   ('space_time.teacher.transform_identity', self.configs[384]['space_time']['teacher']['transform_identity'])]
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                configs = deepcopy(self.configs)
                obj = configs[256]
                keys = field.split('.')
                for key in keys[:-1]:
                    obj = obj[key]
                obj[keys[-1]] = value
                pair.verify_pair(configs, self.metadata)

    def test_reject_basis_refit_and_post_pca_whitening(self):
        for key, value in [('identity', '0' * 64), ('fit_role', 'dev'), ('whiten', True), ('standardize_channels', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                metadata = deepcopy(self.metadata)
                metadata['pca_receipt'][key] = value
                pair.expected_pair(metadata)

    def test_shared_readout_initialization_and_training_rng_are_identical(self):
        import torch
        from starVLA.model.modules.vehicle_joint.initialization import initialization_seed
        from starVLA.model.modules.st_world.sparse_readout import SparseFeatureReadout
        torch.set_num_threads(2)
        torch.manual_seed(123)
        outer = torch.get_rng_state().clone()
        heads = []
        for c in (256, 384):
            with initialization_seed(42 + 3600):
                heads.append(SparseFeatureReadout(2048, c))
            self.assertTrue(torch.equal(torch.get_rng_state(), outer))
        left, right = [m.state_dict() for m in heads]
        for name in left:
            if not name.startswith('output.'):
                self.assertTrue(torch.equal(left[name], right[name]), name)
        self.assertEqual(sum(p.numel() for p in heads[0].parameters()), 8017664)
        self.assertEqual(sum(p.numel() for p in heads[1].parameters()), 8083328)
        # The last-layer bias follows a differently sized weight draw. This is
        # explicitly not a claim of prefix-equivalent final-layer initialization.
        self.assertFalse(torch.equal(left['output.bias'], right['output.bias'][:256]))

    def test_pca_is_pre_standardized_nonwhitened_and_uses_common_prefix(self):
        import torch
        from tools.st_world_policy.teacher_pca import TeacherPCA
        tensors = {'channel_mean': torch.tensor([1., 2., 3.]),
                   'channel_std': torch.tensor([2., 4., 8.]),
                   'pca_mean': torch.tensor([.5, 0., -.5]),
                   'components': torch.eye(3), 'eigenvalues': torch.tensor([9., 4., 1.])}
        x = torch.tensor([[5., 10., 19.]])
        small = TeacherPCA(tensors, dimension=2, identity='fixture', teacher_contract_identity='fixture')
        large = TeacherPCA(tensors, dimension=3, identity='fixture', teacher_contract_identity='fixture')
        expected = (x - tensors['channel_mean']) / tensors['channel_std'] - tensors['pca_mean']
        self.assertTrue(torch.equal(large.transform(x), expected))
        self.assertTrue(torch.equal(small.transform(x), expected[:, :2]))
        self.assertTrue(torch.equal(large.inverse(large.transform(x)), x))

    def test_current_loss_denominator_is_original_valid_elements(self):
        import torch
        from starVLA.model.modules.st_world.losses import masked_mse
        mask = torch.tensor([True, False])
        target = torch.zeros(2, 384)
        prediction = torch.cat((torch.ones(2, 256), torch.full((2, 128), 2.)), dim=-1).requires_grad_()
        loss = masked_mse(prediction, target, mask, 384, world_size=1).loss
        self.assertAlmostEqual(loss.item(), 2.)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[1], torch.zeros(384)))
        self.assertAlmostEqual(prediction.grad[0, 0].item(), 2 / 384)

    def test_error_blocks_do_not_mix_channel_dimensions_or_call_missing_tail_zero(self):
        import numpy as np
        from channel_errors import paired_channel_errors
        target = np.ones((2, 384), dtype=np.float32)
        p384 = target.copy(); p384[:, :256] += 1.; p384[:, 256:] += 2.
        p256 = target[:, :256] + 3.
        ids = np.array(['scene-a/view0/patch4', 'scene-b/view2/patch71'])
        report = paired_channel_errors(p256, p384, target, np.array([True, False]),
            request_ids256=ids, request_ids384=ids, target_request_ids=ids)
        self.assertEqual(report['common_first256']['arm256']['mse'], 9.)
        self.assertEqual(report['common_first256']['arm384']['mse'], 1.)
        self.assertEqual(report['extra_128']['arm384']['mse'], 4.)
        self.assertEqual(report['arm384_full_mse'], report['arm384_weighted_block_mse'])
        self.assertEqual(report['extra_128']['arm256'], 'NOT_PREDICTED_NOT_ZERO_ERROR')
        with self.assertRaises(ValueError):
            paired_channel_errors(p256, p384, target, np.array([True, False]),
                request_ids256=ids, request_ids384=ids[::-1], target_request_ids=ids)
        report = paired_channel_errors(p256, p384, target, np.array([False, False]),
            request_ids256=ids, request_ids384=ids, target_request_ids=ids)
        self.assertEqual(report['status'], 'NO_VALID_REQUESTS')


if __name__ == '__main__':
    unittest.main()
