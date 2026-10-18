import copy, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('binding',HERE/'binding_factory.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
class Factory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=b.actual_inputs();cls.cfg,_=b.document(cls.inputs['pins']['config']['path'])
        cls.result,_=b.document(b.PREP/'controls/RESULT.json')
    def test_actual_complete_canonical_inputs(self):
        saved=json.loads((HERE/'INPUTS.json').read_text());saved.pop('native_argv_for_existing_supervisor')
        self.assertEqual(self.inputs,saved)
        self.assertEqual(self.inputs['execution']['stop_after'],4)
        self.assertEqual(self.inputs['pins']['config']['sha256'],'5fd3da85fe1a9561124b652d78126aaab0be8e013ff8f9400124863ae8babfac')
        self.assertIn('/.configured.staging-',self.inputs['pins']['config']['path'])
        self.assertFalse(self.inputs['training_started']);self.assertFalse(b.OUT.exists())
    def test_failed_or_incomplete_factory_rejected(self):
        for fault in ('failed','returncode','live','missing'):
            bad=copy.deepcopy(self.result)
            if fault=='failed':bad['status']='FAILED_NO_AUTO_RETRY'
            if fault=='returncode':bad['resources']['returncode']=2
            if fault=='live':bad['resources']['cleanup']['live_owned_remaining']=1
            if fault=='missing':del bad['cli_result']['outputs']['configured/PROVIDER.json']
            with self.subTest(fault=fault),self.assertRaises(ValueError):b.validate_factory(bad)
    def test_schedule_budget_cell_changes_rejected(self):
        for keys,v in [(('execution','stop_after'),5),(('execution','global_batch'),4),(('execution','max_updates'),4),(('optimizer','warmup'),1),(('runtime','loader_workers'),1),(('space_time','cell'),'T4')]:
            bad=copy.deepcopy(self.cfg);bad[keys[0]][keys[1]]=v
            with self.subTest(keys=keys),self.assertRaises(ValueError):b.validate_recipe(bad)
    def test_missing_dependency_is_error_not_mock(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b,'PREP',Path(d)):
            with self.assertRaises(FileNotFoundError):b.actual_inputs()
    def test_wrong_hash_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{}');link=Path(d)/'alias.json';link.symlink_to(p)
            with self.assertRaises(ValueError):b.document(p,{'path':str(p),'sha256':'0'*64})
            with self.assertRaises(ValueError):b.document(link)
    def test_original_registrar_construct_and_verify_fixtures(self):
        # Fixture artifacts are only metadata; no actual registration/lease is issued.
        with tempfile.TemporaryDirectory() as d:
            def write(name,value):
                p=Path(d)/name;p.write_text(json.dumps(value));return b.document(p)[1]
            env=write('env.json',{'schema':'st_training_environment_v1'})
            target=write('target.json',self.cfg['space_time']['teacher'])
            lease=dict(schema='st_training_resource_lease_v1',hostnames=[b.HOST],world_size=1,
                cuda_visible_devices_by_host={b.HOST:'0'},gpu_uuids_by_host={b.HOST:[b.GPU_UUID]},
                max_invocation_seconds=1800,cpu_threads=2,tensor_cache_write_bytes=0,
                checkpoint_reservation_bytes=80<<30,valid_from_unix=0,valid_until_unix=2200)
            lp=write('lease.json',lease)
            reg=b.registration_value(self.inputs,env,lp,target,1)
            rp=write('reg.json',reg);b.registrar().verify_registration(rp['path'],repo=b.RUNTIME)
            self.assertEqual(len(reg['artifacts']),7);self.assertFalse(reg['weights_usable_for_formal_init'])
            for field,value in [('valid_until_unix',100),('world_size',2),('gpu_uuids_by_host',{b.HOST:['wrong']})]:
                wrong=copy.deepcopy(lease);wrong[field]=value;pin=write('wrong.json',wrong)
                with self.subTest(field=field),self.assertRaises(ValueError):b.registration_value(self.inputs,env,pin,target,1)
if __name__=='__main__':unittest.main(verbosity=2)
