"""Small source/JSON-only lease tests; never imports v5 or dispatches scoring."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('score_lease_test',HERE/'score_with_execution_lease.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def pin(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='score-lease-fixture-')
        self.root=Path(self.temp.name)
        self.real_root=m.ROOT;m.ROOT=self.root
        self.job=m.JOBS[0];self.host='training-vlawm-tsinghua-zt-worker-0'
        self.route={'job_id':self.job,'hostname':self.host,'source_sha':m.SOURCE,
                    'environment':{'path':'/fixture/environment.json','sha256':'e'*64},
                    'inference':{'score_workers':2,'score_max_calls':1728},
                    'valid_from_unix':0.,'valid_until_unix':1.}
        path=self.root/'formal/evaluation/registrations'/self.job/'ROUTE.json'
        path.parent.mkdir(parents=True);path.write_text(json.dumps(self.route));self.rpin=pin(path)
        self.lease={'schema':'st_fixed_dev_score_execution_lease_v1','job_id':self.job,
            'original_export_route':self.rpin,'hostname':self.host,'source_sha':m.SOURCE,
            'environment':self.route['environment'],'score_helper':m.HELPER,
            'valid_from_unix':100.,'valid_until_unix':5000.,'limits':m.LIMITS.copy(),
            'scope':'score_existing_complete_export_only','automatic_retry':False,'export_authorized':False}
        self.lpath=self.root/'LEASE.json';self.save_lease()
    def save_lease(self):
        self.lpath.write_text(json.dumps(self.lease));self.lpin=pin(self.lpath)
    def tearDown(self):m.ROOT=self.real_root;self.temp.cleanup()
    def validate(self,lease=None,route=None,now=200,remaining_seconds=4260):
        return m.validate_lease(lease or self.lease,route or self.route,job=self.job,
                route_pin=self.rpin,hostname=self.host,now=now,remaining_seconds=remaining_seconds)
    def install(self):
        self.writer=Mock();self.old_time=Mock()
        module=SimpleNamespace(check_route_time_v3=self.old_time,atomic_json=self.writer,main=Mock())
        m.install(module,lease_pin=self.lpin,job=self.job,route_pin=self.rpin,hostname=self.host)
        return module

    def test_expired_export_route_can_use_explicit_fresh_score_window(self):
        original=copy.deepcopy(self.route);self.validate()
        self.assertEqual(self.route,original)

    def test_times_finite_fresh_bounded_and_full_cleanup_coverage(self):
        for update in ({'valid_from_unix':201},{'valid_until_unix':4459},
                       {'valid_from_unix':float('nan')},{'valid_until_unix':float('inf')},
                       {'valid_from_unix':True},{'valid_until_unix':30000}):
            value={**self.lease,**update}
            with self.subTest(update=update),self.assertRaises(ValueError):self.validate(value)
        with self.assertRaises(ValueError):self.validate(remaining_seconds=7200)

    def test_wrong_identity_or_resource_fields_rejected(self):
        for field,value in [('job_id',m.JOBS[1]),('hostname','foreign-host'),('source_sha','bad'),
                            ('environment',{}),('score_helper',{}),('original_export_route',{}),
                            ('scope','export'),('automatic_retry',True),('export_authorized',True),
                            ('limits',{**m.LIMITS,'score_workers':4}),
                            ('limits',{**m.LIMITS,'max_calls_total_across_resume':9999}),
                            ('limits',{**m.LIMITS,'maximum_score_invocations':3})]:
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate({**self.lease,field:value})
        with self.assertRaises(ValueError):self.validate({**self.lease,'unexpected':True})

    def test_both_original_time_sites_use_current_time_and_same_route(self):
        module=self.install()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            module.check_route_time_v3(self.route,200,4260)
            with self.assertRaises(ValueError):module.check_route_time_v3(self.route,1000,4260)
            with self.assertRaises(ValueError):module.check_route_time_v3({**self.route,'gpu':7},200,4260)
        self.old_time.assert_not_called()

    def test_pin_change_between_checks_rejected(self):
        module=self.install()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            module.check_route_time_v3(self.route,200,4260)
            self.lpath.write_text(self.lpath.read_text()+' ')
            with self.assertRaises(ValueError):module.check_route_time_v3(self.route,201,4260)

    def test_gpu_visibility_rejected(self):
        module=self.install()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaises(ValueError):module.check_route_time_v3(self.route,200,4260)

    def test_additive_receipts_preserve_original_fields_and_immutable_writer(self):
        module=self.install();before={'route':self.rpin,'ordinal':2,'maximum_invocations':2,'argv':['original']}
        root=self.root/'formal/evaluation/registrations'/self.job/'score_invocations'/'fixture'
        for name in ('ADMISSION.json','LAUNCH.json'):
            module.atomic_json(root/name,before,immutable=True)
            _,actual=self.writer.call_args.args
            self.assertEqual({k:actual[k] for k in before},before)
            self.assertEqual(actual['score_execution_lease'],self.lpin)
            self.assertNotIn('score_execution_lease',before)
            self.assertEqual(self.writer.call_args.kwargs,{'immutable':True})
        with self.assertRaises(ValueError):module.atomic_json(root/'ADMISSION.json',before,immutable=False)
        with self.assertRaises(ValueError):module.atomic_json(root/'ADMISSION.json',{'route':{}},immutable=True)

    def test_no_lease_delegates_original_main_without_install(self):
        module=SimpleNamespace(main=Mock(return_value=17))
        with patch.object(m,'load_helper',return_value=module),patch.object(m,'install') as installed, \
                patch.object(sys,'argv',['wrapper','--job-id',self.job,'--route-sha256',self.rpin['sha256']]):
            self.assertEqual(m.main(),17);installed.assert_not_called();module.main.assert_called_once()

    def test_original_budget_and_time_calls_remain_in_pinned_v5_source(self):
        source=m.pinned_small(m.HELPER,decode=False).decode();tree=ast.parse(source)
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
        names=[n.func.id for n in calls if isinstance(n.func,ast.Name)]
        self.assertIn('check_score_budget',names);self.assertIn('validate_route_v3',names)
        self.assertEqual(names.count('check_route_time_v3'),1)
        validate=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='validate_route_v3')
        self.assertIn('check_route_time_v3',ast.unparse(validate))
        budget=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='check_score_budget')
        namespace={'Path':Path};exec(compile(ast.Module(body=[budget],type_ignores=[]),'pinned-v5-budget','exec'),namespace)
        reg=self.root/'budget';out=self.root/'output';(reg/'score_invocations/1').mkdir(parents=True)
        (reg/'score_invocations/2').mkdir()
        with self.assertRaisesRegex(AssertionError,'Two scoring'):
            namespace['check_score_budget'](reg,out,False)


if __name__=='__main__':unittest.main(verbosity=2)
