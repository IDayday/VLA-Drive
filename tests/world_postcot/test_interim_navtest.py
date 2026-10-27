import json
from pathlib import Path
import tempfile
import unittest
from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from starVLA.rl.qwen_world_postcot.interim_navtest import validate_interim


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


if __name__=='__main__':unittest.main()
