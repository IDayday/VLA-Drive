"""CPU checks for explicit parent authorization; no actor/optimizer changes."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.qwen_sourceflow.loading import inspect_il,inspect_il_checkpoint
from starVLA.rl.qwen_sourceflow.selection import validate_selection
from starVLA.rl.flow_grpo.transactions import atomic_json


def sealed(body):
    body=copy.deepcopy(body);body.pop('sha256',None);body['sha256']=fingerprint(body);return body


class ParentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.checkpoint=self.root/'update053000';self.checkpoint.mkdir()
        self.identity=sealed(dict(schema='native_qwen_sourceflow_IL_v1',qualification=False,
            config=dict(milestones=[20000,40000,60000,80000,100000])))
        atomic_json(self.checkpoint/'identity.json',self.identity)
        (self.checkpoint/'model.pt').write_bytes(b'CPU metadata fixture, never a model')
        files={n:file_sha(self.checkpoint/n) for n in ['identity.json','model.pt']}
        atomic_json(self.checkpoint/'files.json',files)
        atomic_json(self.checkpoint/'COMPLETE',dict(schema='native_qwen_flow_IL_READY_v1',
            phase='READY',updates=53000,files_sha256=file_sha(self.checkpoint/'files.json')))
        self.parent=inspect_il_checkpoint(self.checkpoint)
        self.authorization=self.root/'authorization.json';self.summary=self.root/'observed.json';self.selection=self.root/'selection.json'
        self.auth=dict(schema='native_IL_user_parent_authorization_v1',checkpoint=str(self.checkpoint),
            model_sha256=files['model.pt'],updates=53000,training_identity=self.identity['sha256'],
            Navtest_observed=True,IL_training_continues=True,RL_variants=['canonical','source'],
            RL_updates_per_variant=50000,training_seed=42,evaluation_seed=2101,
            user_instruction='Use53k for RL',supersedes_selection_rule='wait for full five-DEV window')
        self.observed=dict(status='COMPLETE',policy_type='IL',model_sha256=files['model.pt'],cohort='NAVTEST',
            scenes=12146,logs=136,metric='canonical_NAVSIM_v1.1',seed=2101,
            inference='single_candidate_source_shift5_ODE_K10',qualification=False,full_population=True,rl_export=None)
        self.record=dict(schema='native_IL_user_selection_v1',
            selected=dict(checkpoint=str(self.checkpoint),updates=53000,model_sha256=files['model.pt'],training_identity=self.identity['sha256']),
            selected_model_sha256=files['model.pt'],Navtest_used=True,
            selection_rule='explicit user-selected IL parent; observed Navtest disclosed')
        self.publish()

    def tearDown(self):self.temp.cleanup()

    def publish(self):
        atomic_json(self.authorization,sealed(self.auth));atomic_json(self.summary,sealed(self.observed))
        self.record.update(authorization_path=str(self.authorization),authorization_sha256=file_sha(self.authorization),
            observed_summary_path=str(self.summary),observed_summary_sha256=file_sha(self.summary))
        atomic_json(self.selection,sealed(self.record))

    def test_default_nonmilestone_still_rejected(self):
        with self.assertRaises(ValueError):inspect_il(self.checkpoint)

    def test_explicit_user_parent_accepted(self):
        self.assertEqual(inspect_il(self.checkpoint,parent_selection=self.selection)['seal']['updates'],53000)

    def test_navtest_disclosure_required(self):
        self.record['Navtest_used']=False;self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_authorization_exact_model(self):
        self.auth['model_sha256']='other';self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_authorization_exact_checkpoint(self):
        self.auth['checkpoint']='/other';self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_authorization_does_not_expand_budget(self):
        self.auth['RL_updates_per_variant']=170000;self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_authorization_tamper_rejected(self):
        self.authorization.write_text(self.authorization.read_text().replace('Use53k','Use50k'))
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_qualification_cannot_become_formal(self):
        parent=copy.deepcopy(self.parent);parent['identity']['qualification']=True
        with self.assertRaises(ValueError):validate_selection(self.selection,parent)

    def test_no_qualification_flag_workaround(self):
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,qualification=True,parent_selection=self.selection)

    def test_no_interim_registration_workaround(self):
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,interim_navtest_registration='request',parent_selection=self.selection)

    def test_observed_evidence_parent_required(self):
        self.observed['model_sha256']='other';self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_observed_complete_population_required(self):
        self.observed['scenes']=12;self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_observed_il_not_rl(self):
        self.observed['policy_type']='RL';self.publish()
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)

    def test_selection_no_rng_consumption(self):
        import torch
        before=torch.get_rng_state().clone();inspect_il(self.checkpoint,parent_selection=self.selection)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_existing_five_DEV_contract_still_accepted(self):
        candidates=[]
        for update in [20000,40000,60000,80000,100000]:
            path=self.root/f'DEV{update}.json';summary=copy.deepcopy(self.observed)
            summary.update(cohort='DEV',scenes=1696,logs=16);atomic_json(path,summary)
            candidates.append(dict(updates=update,summary_path=str(path),summary_sha256=file_sha(path)))
        record=sealed(dict(schema='native_IL_DEV_selection_v1',selected_model_sha256=self.parent['files']['model.pt'],
            candidates=candidates,Navtest_used=False));atomic_json(self.selection,record)
        self.assertEqual(validate_selection(self.selection,self.parent)['schema'],'native_IL_DEV_selection_v1')
        with self.assertRaises(ValueError):inspect_il(self.checkpoint,parent_selection=self.selection)
        seal=json.loads((self.checkpoint/'COMPLETE').read_text());seal['updates']=40000
        atomic_json(self.checkpoint/'COMPLETE',seal)
        self.assertEqual(inspect_il(self.checkpoint,parent_selection=self.selection)['seal']['updates'],40000)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ParentTests))
    atomic_json(a.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scope='Metadata authorization and unchanged default eligibility only'))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
