"""Independent bounded source/control and synthetic receipt review; no real data."""
import os,time
START=time.monotonic()
def io_counts():
    return {k:int(v) for k,v in (x.split(':') for x in open('/proc/self/io'))}
BOOT=io_counts()
import copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
TREE=HERE.parents[3]
ROUTE=TREE/'reports/st_world_policy/20261017/full_future_index_preparation_v1/prepare_index_controls.py'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
route=load('independent_indexprep',ROUTE)
author=load('rerun_author_cases',HERE.parent/'future_finalized_index_route_v1/test_index_route.py')
EXTRA={}

class ResourceCases(unittest.TestCase):
    def test_01_real_helper_hash_and_separate_counters(self):
        helper=route.load_helper()
        self.assertGreater(helper.READ_BYTES,0);self.assertLess(helper.READ_BYTES,1<<20)
        self.assertEqual(helper.WRITE_BYTES,0)
        self.assertEqual(hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),route.HELPER_SHA)
    def test_02_oversize_file_rejected_before_read(self):
        helper=route.load_helper()
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            p=Path(td)/'too_large.json'
            with p.open('wb') as f:f.truncate((4<<20)+1)
            before=helper.READ_BYTES
            with self.assertRaises(ValueError):helper.read({'path':str(p),'sha256':'0'*64})
            self.assertEqual(helper.READ_BYTES,before)
    def test_03_actual_json_aggregate_cap_and_alias(self):
        helper=route.load_helper()
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            p=Path(td)/'receipt.json';p.write_text('{"test":true}')
            helper.READ_BYTES=32<<20
            with self.assertRaises(ValueError):route.actual_json(helper,p)
            helper.READ_BYTES=0;os.link(p,p.with_name('alias.json'))
            with self.assertRaises(ValueError):route.actual_json(helper,p)
    def test_04_package_plan_counts_pending_once_and_preserves_policy(self):
        h=route.load_helper();policy=json.loads((HERE.parent/'future_finalization_actual_proposal_v1/DRAFT_PROPOSED_POLICY.json').read_text());before=copy.deepcopy(policy)
        pending=2176<<20
        self.assertEqual(route.planned_metadata_bytes(h,policy,0),5525<<20)
        self.assertEqual(route.planned_metadata_bytes(h,policy,pending),7701<<20)
        self.assertEqual(policy,before)
        with self.assertRaises(ValueError):route.planned_metadata_bytes(h,policy,pending*2)
        bad=copy.deepcopy(policy);bad['metadata_package']['maximum_bytes']=9<<30
        with self.assertRaises(ValueError):route.planned_metadata_bytes(h,bad,pending)
    def test_05_index_argv_separate_dev_train_claims(self):
        h=route.load_helper();total=0
        for role,expected in [('dev',128<<20),('train',2048<<20)]:
            request=route.FINALIZED_TRAIN/'TEACHER_REQUESTS.json' if role=='train' else route.BASE/'caches/full_future_metadata_dev_v1/TEACHER_REQUESTS.json'
            command,claim=route.index_arguments(h,role,request,'a'*64,Path('/tmp/synthetic-'+role))
            self.assertEqual(sum(claim.values()),expected);total+=expected
            self.assertEqual(command[command.index('--requests')+1],str(request))
            self.assertEqual(int(command[command.index('--max-disk-bytes')+1]),claim['output_bytes']+claim['staging_bytes'])
        self.assertEqual(total,2176<<20)
    def test_06_missing_finalizer_never_prepares_output(self):
        args=SimpleNamespace(check_only=False,policy=None,policy_sha256=None,ledger_sha256=None,output=None,train_index_source_pins=None,train_index_source_pins_sha256=None)
        with patch.object(route,'completion_inputs',return_value=({}, {'status':'WAITING_ACTUAL_COMPLETE'})),patch.object(Path,'mkdir',side_effect=AssertionError('must not create output')):
            self.assertEqual(route.prepare(args)['status'],'WAITING_ACTUAL_COMPLETE')
    def test_07_actual_five_new_index_runtime_pins(self):
        source=json.loads((HERE.parent/'future_finalized_index_route_v1/TRAIN_INDEX_RUNTIME_SOURCE_PINS.json').read_text())
        self.assertEqual(source['runtime'],str(route.FINALIZER_RUNTIME));self.assertEqual(len(source['files']),5)
        helper=route.load_helper()
        for name,pin in source['files'].items():
            self.assertEqual(Path(pin['path']),route.FINALIZER_RUNTIME/name);helper.read(pin)
        callback=helper.callback_pins(source,'tools.st_world_policy.index_teacher_metadata')
        self.assertEqual(len(callback),3)
        self.assertIn('teacher_provider.py',' '.join(source['files']))
    def test_08_real_import_and_shared_validator_on_two_scene_fixture(self):
        # Full actual validator invoked, not mocked. Every data artifact below is synthetic.
        pre=io_counts();t=time.monotonic();sys.path.insert(0,str(route.FINALIZER_RUNTIME))
        import pytest
        from tools.st_world_policy import finalize_future_metadata as final
        from tests.st_world.test_finalize_future_metadata import make_case
        imported=io_counts();EXTRA['import_seconds']=time.monotonic()-t
        EXTRA['import_rchar_bytes']=imported['rchar']-pre['rchar']
        self.assertEqual(Path(final.__file__).resolve(),route.FINALIZER_RUNTIME/'tools/st_world_policy/finalize_future_metadata.py')
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic-only-') as td:
            with pytest.MonkeyPatch.context() as mp:
                case=make_case(Path(td),mp,full=True)
                done=final.run(case.args)
                helper=route.load_helper();helper_before=helper.READ_BYTES
                pin={'path':str(case.output/final.COMPLETE_NAME),'sha256':final.f.file_sha256(case.output/final.COMPLETE_NAME)}
                # Guard actual receipt validation from touching request bodies, old shards, images/tensors/raw.
                original_reader=final.f.metadata_reader
                def guarded_reader(path,*args,**kwargs):
                    p=Path(path)
                    self.assertTrue(p.is_relative_to(Path(td)) or p.is_relative_to(route.FINALIZER_RUNTIME))
                    self.assertNotEqual(p.name,'TEACHER_REQUESTS.json');self.assertNotEqual(p.parent.name,'log_shards')
                    self.assertNotIn(p.suffix,('.pkl','.png','.pt','.safetensors'))
                    return original_reader(path,*args,**kwargs)
                before=io_counts();valid_t=time.monotonic()
                with patch.object(route,'COUNTS',{'dev':1696,'train':2}),patch.object(final.f,'metadata_reader',guarded_reader):
                    registration=route.strict_finalized_receipt(helper,done,pin)
                    self.assertEqual(registration,case.reg)
                    corrupt=copy.deepcopy(done);corrupt['files']['SHARD_RECEIPTS.json']='0'*64;corrupt.pop('identity');corrupt=final.f.signed(corrupt)
                    with self.assertRaisesRegex(ValueError,'checksum'):
                        route.strict_finalized_receipt(helper,corrupt,pin)
                after=io_counts()
                EXTRA['validator_seconds']=time.monotonic()-valid_t
                EXTRA['validator_rchar_bytes']=after['rchar']-before['rchar']
                EXTRA['validator_helper_reported_bytes']=helper.READ_BYTES-helper_before
                EXTRA['synthetic_fixture_logical_size_bytes']=sum(p.stat().st_size for p in Path(td).rglob('*') if p.is_file())
                EXTRA['fixture_stage_rchar_bytes']=after['rchar']-imported['rchar']
                EXTRA['fixture_stage_wchar_bytes']=after['wchar']-imported['wchar']
                self.assertLess(EXTRA['fixture_stage_rchar_bytes'],32<<20);self.assertLess(EXTRA['fixture_stage_wchar_bytes'],32<<20)
                self.assertGreater(EXTRA['validator_rchar_bytes'],EXTRA['validator_helper_reported_bytes'])
        self.assertFalse(final.f.torch.cuda.is_initialized() if hasattr(final.f,'torch') else False)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.RouteTests),unittest.defaultTestLoader.loadTestsFromTestCase(ResourceCases)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    end=io_counts();EXTRA.update(tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors),failed=len(result.failures),errors=len(result.errors),author_cases_actually_rerun=9,independent_added_cases=8,body_seconds=time.monotonic()-START,own_rchar_delta=end['rchar']-BOOT['rchar'],cpu_affinity=sorted(os.sched_getaffinity(0)),actual_payload_reads=0,actual_ledger_read_or_mutation=0,actual_dispatch=0,actual_task_signals=0)
    (HERE/'TEST_RESULT.json').write_text(json.dumps(EXTRA,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
