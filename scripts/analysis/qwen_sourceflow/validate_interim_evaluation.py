"""CPU checks for the explicit intermediate Navtest registration contract."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint
from starVLA.rl.qwen_sourceflow.interim_evaluation import validate_interim_registration


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'request.json'
        self.tokens=[str(i) for i in range(12146)]
        self.data=SimpleNamespace(sha='data',tokens=self.tokens,manifest=dict(cohort='NAVTEST'),
            records={t:dict(log=str(i%136)) for i,t in enumerate(self.tokens)})
        self.parent=dict(path='/sealed/IL20k',files={'model.pt':'model'},seal=dict(updates=20000))
        self.body=dict(schema='native_user_interim_NAVTEST_v1',checkpoint='/sealed/IL20k',model_sha256='model',
            updates=20000,data_sha256='data',token_sha256=fingerprint(self.tokens),count=12146,logs=136,
            seed=2101,metric='canonical_NAVSIM_v1.1',inference='single_candidate_source_shift5_ODE_K10',
            output='/evaluation',used_for_parent_selection=False,used_for_training_or_tuning=False)

    def tearDown(self):self.temp.cleanup()

    def run_check(self,change=None):
        body=copy.deepcopy(self.body)
        if change:body.update(change)
        body['sha256']=fingerprint(body);self.path.write_text(json.dumps(body))
        return validate_interim_registration(self.path,self.parent,self.data,self.tokens,'/evaluation')

    def test_exact_registration(self):self.assertEqual(self.run_check()['count'],12146)
    def test_wrong_checkpoint(self):
        with self.assertRaises(ValueError):self.run_check(dict(model_sha256='wrong'))
    def test_wrong_data(self):
        with self.assertRaises(ValueError):self.run_check(dict(data_sha256='wrong'))
    def test_wrong_output(self):
        with self.assertRaises(ValueError):self.run_check(dict(output='/other'))
    def test_no_parent_selection(self):
        with self.assertRaises(ValueError):self.run_check(dict(used_for_parent_selection=True))
    def test_no_training_tuning(self):
        with self.assertRaises(ValueError):self.run_check(dict(used_for_training_or_tuning=True))
    def test_seed_fixed(self):
        with self.assertRaises(ValueError):self.run_check(dict(seed=42))

    def test_navtest_cache_routing(self):
        from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches,NAVTEST_CACHE,TRAIN_DEV_CACHES
        self.assertEqual(evaluation_caches('NAVTEST'),[NAVTEST_CACHE])
        self.assertEqual(evaluation_caches('DEV'),TRAIN_DEV_CACHES)

    def test_missing_cache_fails_preflight(self):
        from scripts.analysis.qwen_sourceflow.evaluate_policy import preflight_caches
        with self.assertRaises(FileNotFoundError):preflight_caches(self.data,self.tokens[:1],[self.temp.name])


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RegistrationTests))
    if args.output:Path(args.output).write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun),indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
