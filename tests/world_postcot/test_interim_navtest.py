import json
from pathlib import Path
import tempfile
import unittest
from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from starVLA.rl.qwen_world_postcot.interim_navtest import validate_interim
from tools.qwen_world_postcot.user_interim_owner import validate_complete_summary


class InterimWorldTest(unittest.TestCase):
    def case(self, root):
        tokens=[str(i) for i in range(12146)]
        parent=dict(path='checkpoint',model_sha256='native',seal=dict(updates=37000,files_sha256='files'),identity=dict(qualification=False))
        kwargs=dict(parent=parent,model_sha256='export',data_sha256='data',tokens=tokens,
                    population_sha256='population',protocol_sha256='protocol',output=str(root/'out'),seed=2101)
        row=dict(schema='world_postcot_user_interim_NAVTEST_v1',checkpoint='checkpoint',native_model_sha256='native',
                 model_sha256='export',updates=37000,checkpoint_files_sha256='files',data_sha256='data',
                 token_sha256=fingerprint(tokens),population_sha256='population',protocol_sha256='protocol',
                 output=str((root/'out').resolve()),seed=2101,count=12146,prediction_type='x0',
                 inference='single_candidate_cleanx_K10_shift5',used_for_parent_selection=False,used_for_training_or_tuning=False)
        return row,kwargs

    def test_full_exact_request_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,kwargs=self.case(root);row['sha256']=fingerprint(row);p=root/'reg.json';p.write_text(json.dumps(row))
            validate_interim(p,**kwargs)
            for key,value in [('seed',2102),('used_for_parent_selection',True),('model_sha256','other')]:
                bad=dict(row);bad.pop('sha256');bad[key]=value;bad['sha256']=fingerprint(bad);p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):validate_interim(p,**kwargs)

    def test_wrong_parent_or_partial_population_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,kwargs=self.case(root);row['sha256']=fingerprint(row);p=root/'reg.json';p.write_text(json.dumps(row))
            with self.assertRaises(ValueError):validate_interim(p,**dict(kwargs,tokens=kwargs['tokens'][:16]))
            kwargs['parent']['seal']['updates']=35000
            with self.assertRaises(ValueError):validate_interim(p,**kwargs)

    def test_exact_user_selected_40k_requires_matching_parent_and_registration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,kwargs=self.case(root)
            row['updates']=40000;kwargs['parent']['seal']['updates']=40000
            row['sha256']=fingerprint(row);p=root/'reg.json';p.write_text(json.dumps(row))
            validate_interim(p,**kwargs)
            kwargs['parent']['seal']['updates']=41000
            row.pop('sha256');row['updates']=41000;row['sha256']=fingerprint(row);p.write_text(json.dumps(row))
            with self.assertRaises(ValueError):validate_interim(p,**kwargs)

    def test_full_population_has_distinct_fragment_and_recording_counts(self):
        tokens=[str(i) for i in range(12146)]
        pop=dict(navtest_tokens=tokens,token_logs={t:f'recording{int(t)%136%44}_{int(t)%136*100}_{int(t)%136*100+99}' for t in tokens})
        identity=dict(tokens=tokens,token_sha256=fingerprint(tokens),cohort='NAVTEST',qualification=False)
        identity['sha256']=fingerprint(identity)
        summary=dict(status='COMPLETE',scenes=12146,scene_denominator=12146,invalid_scenes=0,
                     physical_logs=44,evaluation_identity_sha256=identity['sha256'])
        summary['sha256']=fingerprint(summary)
        self.assertEqual(validate_complete_summary(summary,identity,pop),dict(manifest_logs=136,physical_logs=44))
        for key,value in [('scenes',12145),('physical_logs',136),('invalid_scenes',1)]:
            bad=dict(summary);bad.pop('sha256');bad[key]=value;bad['sha256']=fingerprint(bad)
            with self.assertRaises(ValueError):validate_complete_summary(bad,identity,pop)


if __name__=='__main__':unittest.main()
