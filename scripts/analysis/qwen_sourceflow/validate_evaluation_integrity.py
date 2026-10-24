"""CPU regressions for identity seals and scene-to-score association."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts.analysis.qwen_sourceflow.evaluation_integrity import verify_identity,verify_scene,CORE_METRICS
from scripts.analysis.qwen_sourceflow import analyze_paired,summarize_results
from starVLA.rl.qwen_sourceflow.contracts import fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def seal(body):
    result=copy.deepcopy(body);result.pop('sha256',None);result['sha256']=fingerprint(result);return result


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'scenes').mkdir();(self.root/'invocations').mkdir()
        self.population=self.root/'population.json'
        atomic_json(self.population,dict(navtest_tokens=['sample'],token_logs={'sample':'log'}))
        protocol=seal(dict(name='navsim_v1.1_independent_reference_candidate'))
        self.identity=seal(dict(schema='native_qwen_flow_evaluation_v1',checkpoint='/fixture/IL',
            parent_model_sha256='parent',rl_export=None,cohort='NAVTEST',data_sha256='fixture',
            qualification=False,full_population=True,count=1,token_sha256=fingerprint(['sample']),
            protocol=protocol,seed=2101,metric='canonical_NAVSIM_v1.1',
            inference='single_candidate_source_shift5_ODE_K10',precision='fixture',source_files={'fixture':'sha'}))
        self.row=seal(dict(identity=self.identity['sha256'],token='sample',log='log',status='OK',invocation='fixture',
            normalized_action=[[0.,0.,0.]]*8,requested_trajectory=[[0.,0.,0.]]*8,
            feedback=dict(token='sample',log='log',status='OK',protocol=protocol['name'],protocol_sha256=protocol['sha256'],
                metrics={k:1. for k in CORE_METRICS},requested_trajectory=[[0.,0.,0.]]*8)))
        atomic_json(self.root/'invocations/fixture.json',dict(status='COMPLETE',end_unix=10.,wall_seconds=5.,
            physical_inference=1,new=1,reused=0,model_loading_seconds=1.,forward_seconds=2.,scoring_wait_seconds=1.))
        self.write()

    def tearDown(self):self.tmp.cleanup()

    def write(self):
        atomic_json(self.root/'identity.json',self.identity);atomic_json(self.root/'scenes/sample.json',self.row)

    def verify(self):
        verify_identity(self.identity)
        return verify_scene(self.row,self.identity,token='sample',log='log')

    def test_valid_sealed_inline_record(self):self.assertEqual(self.verify()['token'],'sample')

    def test_unsealed_identity_edit_rejected(self):
        self.identity['seed']=9999
        with self.assertRaisesRegex(ValueError,'identity seal'):self.verify()

    def test_changed_protocol_seal_rejected(self):
        self.identity['protocol']['name']='different';self.identity=seal(self.identity)
        with self.assertRaisesRegex(ValueError,'protocol seal'):self.verify()

    def test_unsealed_scene_edit_rejected(self):
        self.row['requested_trajectory'][0][0]=1.
        with self.assertRaisesRegex(ValueError,'inline'):self.verify()

    def test_feedback_token_mismatch_rejected(self):
        self.row['feedback']['token']='other';self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'feedback'):self.verify()

    def test_feedback_log_mismatch_rejected(self):
        self.row['feedback']['log']='other';self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'feedback'):self.verify()

    def test_feedback_protocol_mismatch_rejected(self):
        self.row['feedback']['protocol_sha256']='other';self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'feedback'):self.verify()

    def test_scored_request_mismatch_rejected(self):
        self.row['feedback']['requested_trajectory'][0][0]=1.;self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'Scored request'):self.verify()

    def test_legacy_four_dimensional_action_rejected(self):
        self.row['normalized_action']=[[0.]*4]*8;self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'native action'):self.verify()

    def test_missing_metric_rejected(self):
        self.row['feedback']['metrics'].pop('score');self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'metric'):self.verify()

    def test_nonfinite_metric_rejected(self):
        self.row['feedback']['metrics']['score']=float('nan');self.row=seal(self.row)
        with self.assertRaisesRegex(ValueError,'metric'):self.verify()

    def test_valid_zero_score_remains_valid(self):
        self.row['feedback']['metrics']['score']=0.;self.row=seal(self.row)
        self.assertEqual(self.verify()['feedback']['metrics']['score'],0.)

    def test_out_of_prior_range_finite_action_is_not_clipped(self):
        self.row['normalized_action']=[[10.,-12.,3.]]*8;self.row=seal(self.row)
        self.assertEqual(self.verify()['normalized_action'][0],[10.,-12.,3.])

    def test_paired_reader_uses_identity_check(self):
        self.identity['seed']=9999;self.write()
        with patch.object(analyze_paired,'POPULATION',self.population):
            with self.assertRaisesRegex(ValueError,'identity seal'):analyze_paired.read(self.root)

    def test_complete_summary_uses_linkage_check(self):
        self.row['feedback']['token']='other';self.row=seal(self.row);self.write()
        with patch('starVLA.rl.qwen_sourceflow.contracts.POPULATION',self.population),\
             patch.object(sys,'argv',['summary','--evaluation',str(self.root),'--output',str(self.root/'summary.json')]):
            with self.assertRaisesRegex(ValueError,'feedback'):summarize_results.main()

    def test_valid_summary_score_and_cost_are_preserved(self):
        out=self.root/'summary.json'
        with patch('starVLA.rl.qwen_sourceflow.contracts.POPULATION',self.population),\
             patch.object(sys,'argv',['summary','--evaluation',str(self.root),'--output',str(out)]),\
             contextlib.redirect_stdout(io.StringIO()):summarize_results.main()
        result=json.loads(out.read_text());self.assertEqual(result['PDMS'],100.)
        self.assertEqual(result['physical_inference_including_retries'],1)
        self.assertEqual(result['worker_wall_seconds'],5.)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrityTests))
    atomic_json(a.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scope='Synthetic CPU analysis-integrity regression; no driving score evidence'))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
