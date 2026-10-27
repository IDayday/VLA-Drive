import unittest
from scripts.analysis.qwen_sourceflow.finite_navtest_scan import seal
from scripts.analysis.qwen_sourceflow.latest_navtest_request import validate_request, scoped_report_paths, LABELS


class LatestNavtestTest(unittest.TestCase):
    def plan(self):
        p=dict(schema='native_latest_three_NAVTEST_v1',seed=2101,scenes_each=12146,logs_each=136,
               metric='canonical_NAVSIM_v1.1',inference='single_candidate_source_shift5_ODE_K10',
               temporary_eval_host=None,total_new_single_candidate_inferences=36438,
               models={k:dict(kind=v[0],updates=v[1],parent_selection='IL77') for k,v in LABELS.items()})
        p['sha256']=seal(p);return p

    def test_exact_three_protocol(self):
        p=self.plan();validate_request(p)
        for key,value in [('seed',2102),('temporary_eval_host','training-vla-zt3'),('total_new_single_candidate_inferences',4*12146)]:
            q=dict(p);q.pop('sha256');q[key]=value;q['sha256']=seal(q)
            with self.assertRaises(ValueError):validate_request(q)

    def test_report_guard_preserves_leading_porcelain_space(self):
        prefix='reports/test'
        self.assertEqual(scoped_report_paths(' M reports/test/README.md\0?? reports/test/result.json\0',prefix),
                         ['reports/test/README.md','reports/test/result.json'])
        with self.assertRaises(ValueError):scoped_report_paths(' M actor.py\0',prefix)


if __name__=='__main__':unittest.main()
