import copy
import unittest

from scripts.analysis.qwen_sourceflow.compare_rl_checkpoints import require_same_contract


class CompareRLCheckpointsTest(unittest.TestCase):
    def identity(self):
        return dict(schema='native', cohort='NAVTEST', data_sha256='data', token_sha256='tokens',
                    count=12146, metric='v1.1', protocol='canonical', seed=2101, inference='ODE',
                    precision='fixed', source_files={'actor.py': 'actor'}, qualification=False,
                    full_population=True, parent_model_sha256='IL53',
                    rl_export=dict(parent_model_sha256='IL53', model_sha256='RL'))

    def test_matched_full_same_parent_rl(self):
        a = self.identity(); b = copy.deepcopy(a)
        b['rl_export']['model_sha256'] = 'another_RL'
        require_same_contract(a, b, ['scene'], ['scene'])

    def test_other_parent_or_source_rejected(self):
        for mutate in (lambda x: x['rl_export'].update(parent_model_sha256='IL77'),
                       lambda x: x['source_files'].update({'actor.py': 'different'})):
            a = self.identity(); b = copy.deepcopy(a); mutate(b)
            with self.assertRaises(ValueError): require_same_contract(a, b, ['scene'], ['scene'])

    def test_il_and_partial_population_are_not_rl_comparators(self):
        for key, value in [('rl_export', None), ('full_population', False), ('qualification', True)]:
            a = self.identity(); b = copy.deepcopy(a); b[key] = value
            with self.assertRaises(ValueError): require_same_contract(a, b, ['scene'], ['scene'])


if __name__ == '__main__': unittest.main()
