import unittest
from scripts.analysis.qwen_sourceflow.finite_navtest_scan import Owner, evaluation_arguments, seal, validate_plan


class FiniteNavtestScanTest(unittest.TestCase):
    def plan(self):
        p = dict(seed=2101, scenes_each=12146, logs_each=136, metric='canonical_NAVSIM_v1.1',
                 inference='single_candidate_source_shift5_ODE_K10', temporary_eval_host='training-vla-zt3',
                 models={str(i): dict(kind='rl', parent_selection='selected') for i in range(12)},
                 total_new_single_candidate_inferences=12*12146)
        p['sha256'] = seal(p)
        return p

    def test_contract_rejects_new_seed_or_extra_models(self):
        p = self.plan(); validate_plan(p)
        p['seed'] = 2102; p.pop('sha256'); p['sha256'] = seal(p)
        with self.assertRaises(ValueError): validate_plan(p)

    def test_parent_not_silently_shared_between53_and77(self):
        def model(parent):
            return dict(kind='rl', parent_checkpoint=parent, checkpoint='adapter', parent_selection=parent+'_selection',
                        bundle=parent+'_bundle', output='out', data='data')
        _, a = evaluation_arguments(model('parent53'))
        _, b = evaluation_arguments(model('parent77'))
        self.assertEqual(a[a.index('--checkpoint')+1], 'parent53')
        self.assertEqual(b[b.index('--checkpoint')+1], 'parent77')
        self.assertNotEqual(a[a.index('--parent-selection')+1], b[b.index('--parent-selection')+1])

    def test_temporal_uses_its_registration_and_real_three_frame_entry(self):
        module, args = evaluation_arguments(dict(kind='temporal', checkpoint='checkpoint',
            output='out', data='three_frames', registration='registered'))
        self.assertTrue(module.endswith('evaluate_temporal_policy'))
        self.assertIn('--registration', args)
        self.assertNotIn('--parent-selection', args)

    def test_dispatch_cannot_start_training(self):
        for module in ['scripts.analysis.qwen_sourceflow.train_il', 'scripts.analysis.qwen_sourceflow.train_rl']:
            with self.assertRaises(PermissionError):
                Owner.submit(object.__new__(Owner), 'model', 'train', module, [], {})


if __name__ == '__main__': unittest.main()
