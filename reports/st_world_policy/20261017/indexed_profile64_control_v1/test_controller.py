"""CPU fixtures only: no real prepare/run/GPU query or live resource changes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('profile_control',HERE/'controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ControlTests(unittest.TestCase):
    def test_exact_command(self):
        cmd=c.command()
        self.assertEqual(cmd[:4],['timeout','--signal=TERM','--kill-after=180s','1620s'])
        self.assertIn('tools.st_world_policy.train_world',cmd)
        self.assertNotIn('--resume',cmd)
        self.assertEqual(str(c.GPU_LOCK),'/tmp/foresight_gpu_0.lock')
        self.assertEqual(c.GPU,'0')

    def test_unchanged_actual_recipe(self):
        pins=json.loads((HERE/'INPUTS.json').read_text())['pins']
        cfg=json.loads(Path(pins['config']['path']).read_text())
        c.validate_recipe(cfg)
        mutations=[lambda x:x['execution'].update(scope='SMOKE'),
            lambda x:x['execution'].update(stop_after=2),lambda x:x['execution'].update(sample_limit=64),
            lambda x:x['optimizer'].update(warmup=32),lambda x:x['runtime'].update(save_every=1),
            lambda x:x['runtime'].update(max_seconds=900),lambda x:x['runtime'].update(profile_mode='timing')]
        for mutation in mutations:
            bad=deepcopy(cfg);mutation(bad)
            with self.assertRaises(RuntimeError):c.validate_recipe(bad)

    def test_storage_accounting(self):
        self.assertEqual(c.storage_requirement(0),(1024+2103+80)*(1<<30))
        self.assertEqual(c.storage_requirement(100),(1024+2103+80)*(1<<30)-100)
        for bad in (-1,True,c.RESERVE+1):
            with self.assertRaises(RuntimeError):c.storage_requirement(bad)

    def admission_fixture(self):
        return {'schema':'st_indexed_profile64_resource_admission_v1','status':'ADMITTED','host':c.HOST,
            'gpu':'0','source_sha':c.SOURCE,'run_id':c.RUN_ID,'output_root':str(c.OUT),'controller_sha256':'pin',
            'inputs_sha256':'pin','checkpoint_reservation_bytes':c.RESERVE,'other_commitments_bytes':c.OTHER_COMMITMENTS,
            'free_floor_bytes':c.FREE_FLOOR,'max_invocation_seconds':1500,'outer_timeout_seconds':1620,
            'cleanup_seconds':180,'valid_from_unix':0,'valid_until_unix':4000}

    def test_missing_or_changed_resource_admission(self):
        a=self.admission_fixture()
        with mock.patch.object(c,'read',return_value=a),mock.patch.object(c,'digest',return_value='pin'),mock.patch.object(c.time,'time',return_value=1000):
            c.admission('/fixture-only','pin')
            for name,value in [('gpu','1'),('checkpoint_reservation_bytes',120<<30),
                ('other_commitments_bytes',1983<<30),('valid_until_unix',2799),('valid_until_unix',float('inf')),
                ('valid_until_unix',True),('status','PROPOSED')]:
                with self.subTest(name=name,value=value):
                    bad=deepcopy(a);bad[name]=value
                    with mock.patch.object(c,'read',return_value=bad),self.assertRaises(RuntimeError):
                        c.admission('/fixture-only','pin')
            with self.assertRaises(RuntimeError):c.admission(None,None)

    def test_busy_gpu_guard(self):
        with mock.patch.object(c.subprocess,'check_output',return_value='GPU-fixture, 65, 90\n'):
            with self.assertRaises(RuntimeError):c.gpu_record(True)

    def test_preflight_does_not_prepare_or_run(self):
        with (mock.patch.object(c,'common',return_value=({},{})),mock.patch.object(c,'prepare') as p,
             mock.patch.object(c,'execute') as e,mock.patch.object(c,'gpu_record') as g,
             mock.patch.object(c.sys,'argv',['controller.py'])):
            c.main()
        p.assert_not_called();e.assert_not_called();g.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
