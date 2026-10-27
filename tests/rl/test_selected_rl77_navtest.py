import copy
import unittest
from scripts.analysis.qwen_sourceflow.selected_rl77_navtest_request import LABELS, HOSTS, validate_request
from scripts.analysis.qwen_sourceflow.finite_navtest_scan import seal


class SelectedRL77Test(unittest.TestCase):
    def plan(self):
        return dict(schema='native_selected_RL77_three_NAVTEST_v1', seed=2101,
            scenes_each=12146, logs_each=136, metric='canonical_NAVSIM_v1.1',
            inference='single_candidate_source_shift5_ODE_K10', temporary_eval_host=None,
            total_new_single_candidate_inferences=36438, fixed_hosts=HOSTS,
            max_evaluation_workers_per_model=8,
            models={k:dict(kind='rl', updates=u, parent_checkpoint='IL77', parent_selection='selection') for k,u in LABELS.items()})

    def test_exact_fixed_scope(self):
        p=self.plan();p['sha256']=seal(p);validate_request(p)
        for field, value in [('seed',2102),('temporary_eval_host','training-vla-zt3'),('scenes_each',32)]:
            bad=self.plan();bad[field]=value;bad['sha256']=seal(bad)
            with self.assertRaises(ValueError):validate_request(bad)

    def test_parent_and_checkpoint_cannot_drift(self):
        for field,value in [('updates',50000),('parent_checkpoint','IL53')]:
            bad=self.plan();bad['models']['RL77_040000'][field]=value;bad['sha256']=seal(bad)
            with self.assertRaises(ValueError):validate_request(bad)


if __name__=='__main__':unittest.main()
