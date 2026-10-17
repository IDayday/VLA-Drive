"""Small adapter fixtures only; shared receipt-validator tests were reviewed separately."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('index_route',HERE.parent/'full_future_index_preparation_v1/prepare_index_controls.py')
route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pin(name):return {'path':name,'sha256':digest(name)}


class RouteTests(unittest.TestCase):
    def fixture(self):
        plan=pin('old-plan.json');internal=pin('finalizer-reg.json');storage_pin=pin('storage.json')
        done={'schema':'st_future_metadata_finalization_complete_v1','status':'COMPLETE','scope':'full_population',
            'scenes':101592,'source_plan':plan,'finalization_registration':internal,
            'source_terminal':pin('old-failed.json'),'files':{'TEACHER_REQUESTS.json':'a'*64},'request_identity':'b'*64}
        done_pin=pin(str(route.FINALIZED_TRAIN/'FINALIZATION_COMPLETE.json'));result_pin=pin(str(route.FINALIZATION_RESULT))
        admission={'schema':'st_future_finalization_owned_admission_v1','status':'APPROVED_FUTURE_FINALIZATION_ONLY',
            'reviewer':'independent-reviewer','automatic_retry':False,'registration':storage_pin,
            'finalization_registration':internal,'source_terminal':done['source_terminal'],
            'output_root':str(route.FINALIZED_TRAIN),'control_root':str(route.FINALIZATION_RESULT.parent),
            'runtime':str(route.FINALIZER_RUNTIME),'limits':{'read_bytes':16<<30}}
        storage={'writer_kind':'future_finalization','output_root':str(route.FINALIZED_TRAIN),
            'cwd':str(route.FINALIZER_RUNTIME),'command':['python','-m','tools.st_world_policy.finalize_future_metadata',
            'finalize-existing','--registration',internal['path'],'--registration-sha256',internal['sha256'],
            '--output',str(route.FINALIZED_TRAIN)]}
        result={'schema':'st_future_finalization_invocation_v1','status':'FINALIZATION_COMPLETE',
            'automatic_retry':False,'scientific_result':False,'resources':{'returncode':0,'failure':None,
            'cleanup':{'live_owned_remaining':0},'read_accounting':{'status':'COMPLETE'}},
            'cleanup':{'live_owned_remaining':0},'cli_result':{'status':'GUARDED_METADATA_RETURN',
            'artifact':copy.deepcopy(done),'registration_sha256':storage_pin['sha256']},
            'registration':storage_pin,'admission':pin('admission.json'),'finalization_registration':internal,
            'source_terminal':done['source_terminal'],'charged_rchar_bytes':123,
            'ledger_result':str(route.FINALIZED_TRAIN/'CONTROL_FINALIZATION_RESULT.json')}
        documents={'admission.json':admission,'storage.json':storage}
        helper=SimpleNamespace(PYTHON='python',document=lambda p:documents[p['path']])
        return helper,plan,done,done_pin,result,result_pin,admission,storage

    def bind(self,fixture):return route.bind_finalized_train(*fixture[:6])

    def test_exact_new_branch_delegates_to_shared_validator(self):
        fixture=self.fixture();validated={'source_code_root':'old-frozen-runtime'}
        with patch.object(route,'strict_finalized_receipt',return_value=validated) as strict:
            self.assertIs(self.bind(fixture),validated)
            strict.assert_called_once_with(fixture[0],fixture[2],fixture[3])

    def test_rejects_terminal_lineage_and_budget_changes(self):
        for change in ('old_success','cleanup','artifact','charged','draft','command','source_terminal','ledger_result'):
            with self.subTest(change=change):
                f=list(self.fixture());r,a,s=f[4],f[6],f[7]
                if change=='old_success':r['status']='BUILD_COMPLETE'
                elif change=='cleanup':r['cleanup']['live_owned_remaining']=1
                elif change=='artifact':r['cli_result']['artifact']['request_identity']='c'*64
                elif change=='charged':r['charged_rchar_bytes']=(16<<30)+1
                elif change=='draft':a['status']='DRAFT_NOT_APPROVED'
                elif change=='command':s['command'][3]='build'
                elif change=='source_terminal':r['source_terminal']=pin('different-failure.json')
                elif change=='ledger_result':r['ledger_result']='old/CONTROL_BUILD_RESULT.json'
                with patch.object(route,'strict_finalized_receipt',side_effect=AssertionError('must reject before validator')):
                    with self.assertRaises(ValueError):self.bind(f)

    def test_rejects_old_receipt_and_wrong_path(self):
        for change in ('schema','path','plan'):
            with self.subTest(change=change):
                f=list(self.fixture())
                if change=='schema':f[2]['schema']='st_future_metadata_complete_v1'
                elif change=='path':f[3]=pin(str(route.FINALIZED_TRAIN/'COMPLETE.json'))
                else:f[2]['source_plan']=pin('changed-plan.json')
                with self.assertRaises(ValueError):self.bind(f)

    def test_shared_validator_failure_propagates(self):
        with patch.object(route,'strict_finalized_receipt',side_effect=ValueError('wrong provenance/order')):
            with self.assertRaisesRegex(ValueError,'provenance/order'):self.bind(self.fixture())

    def test_runtime_pin_mismatch_rejected_before_import(self):
        helper=SimpleNamespace(document=lambda pin:{'implementation':[]})
        with self.assertRaisesRegex(ValueError,'pin set'):
            route.strict_finalized_receipt(helper,{'finalization_registration':pin('fixture')},pin('receipt'))

    def test_legacy_dev_route_still_accepts_exact_attempt2(self):
        helper=SimpleNamespace(digest=digest)
        plan_pin=pin('dev-plan.json')
        plan={'status':'PLAN_ONLY','scope':'full_population','selected_scene_count':1696,
            'selected_logs':['dev-log'],'target_times_s':[0.,.5,1.,2.,4.],'original_selection':pin('selection'),
            'gpu_authorized':False,'teacher_extraction_authorized':False,'training_authorized':False}
        plan['identity']=digest(plan)
        reg={'schema':'st_future_metadata_registration_v1','phase':'data_preparation','approved_scope':'full_population',
            'plan_identity':plan['identity'],'plan_sha256':plan_pin['sha256'],
            'gpu_authorized':False,'teacher_extraction_authorized':False,'training_authorized':False}
        reg['identity']=digest(reg);storage=pin('dev-storage')
        done={'schema':'st_future_metadata_complete_v1','status':'COMPLETE','scope':'full_population',
            'plan_identity':plan['identity'],'registration_identity':reg['identity'],'scenes':1696,'logs':1,
            'target_times_s':plan['target_times_s'],'original_selection':plan['original_selection'],
            'teacher_extractions':0,'tensor_disk_bytes':0,'ego_tensor_decoded':False,'training_admission':False,
            'raw_log_deserialized':True,'request_identity':'d'*64,
            'files':{name:'e'*64 for name in ('TEACHER_REQUESTS.json','RAW_LOG_PROVENANCE.json','SHARD_RECEIPTS.json')}}
        result={'schema':'st_full_future_build_invocation_v1','status':'BUILD_COMPLETE','role':'dev','attempt':2,
            'preparation':route.PREPARATION,'automatic_retry':False,'training_authorized':False,'scientific_result':False,
            'resources':{'returncode':0,'failure':None,'cleanup':{'live_owned_remaining':0}},
            'cli_result':{'status':'GUARDED_METADATA_RETURN','registration_sha256':storage['sha256'],'artifact':done},
            'prior_attempt':route.DEV_FAILED}
        with patch.object(route,'strict_finalized_receipt',side_effect=AssertionError('DEV must stay legacy')):
            route.validate_completion(helper,'dev',plan_pin,plan,reg,storage,done,result)
            result['prior_attempt']=pin('wrong-dev-prior')
            with self.assertRaises(ValueError):route.validate_completion(helper,'dev',plan_pin,plan,reg,storage,done,result)

    def test_budget_retains_history_and_counts_pending_indexes_once(self):
        helper=SimpleNamespace(MiB=1<<20,GiB=1<<30)
        draft=HERE.parent/'future_finalization_actual_proposal_v1/DRAFT_PROPOSED_POLICY.json'
        policy=json.loads(draft.read_text())
        before=copy.deepcopy(policy)
        self.assertEqual(route.planned_metadata_bytes(helper,policy,2176<<20),7701<<20)
        self.assertEqual(policy,before)
        with self.assertRaises(ValueError):route.planned_metadata_bytes(helper,policy,3000<<20)

    def test_index_argv_uses_verified_new_request_path(self):
        helper=SimpleNamespace(MiB=1<<20,GiB=1<<30,PYTHON='python')
        path=route.FINALIZED_TRAIN/'TEACHER_REQUESTS.json'
        command,requirements=route.index_arguments(helper,'train',path,'f'*64,Path('/tmp/synthetic-index'))
        self.assertEqual(command[command.index('--requests')+1],str(path))
        self.assertEqual(sum(requirements.values()),2048<<20)

    def test_completion_inputs_selects_new_train_and_original_dev_roots(self):
        documents={};roles={};resources={};storage_pins={};actual={}
        for role in ('dev','train'):
            old=route.BASE/('caches/full_future_metadata_'+role+'_v1')
            plan=pin(str(old/'PLAN.json'));registration=pin(role+'-registration.json')
            storage_pin=pin(role+'-storage.json');storage_pins[role]=storage_pin
            roles[role]={'plan':plan};resources[role]={'prep_output_root':str(old)}
            documents[plan['path']]={};documents[registration['path']]={}
            documents[storage_pin['path']]={'writer_kind':'future_preparation','output_root':str(old),
                'command':['python','-m','tools.st_world_policy.prepare_future_metadata','build','--plan',plan['path'],
                    '--registration',registration['path'],'--registration-sha256',registration['sha256']]}
            root=route.FINALIZED_TRAIN if role=='train' else old
            complete=root/('FINALIZATION_COMPLETE.json' if role=='train' else 'COMPLETE.json')
            terminal=route.FINALIZATION_RESULT if role=='train' else old/route.RESULT_NAMES['dev']
            done={'request_identity':'e'*64,'files':{'TEACHER_REQUESTS.json':'f'*64},
                'scenes':route.COUNTS[role],'logs':1}
            done_pin=pin(str(complete));terminal_pin=pin(str(terminal))
            if role=='dev':done_pin['sha256']=route.DEV_COMPLETE_SHA;terminal_pin['sha256']=route.DEV_RESULT_SHA
            actual[str(complete)]=(done,done_pin);actual[str(terminal)]=({},terminal_pin)
        resource=pin('resource.json');documents[resource['path']]={'roles':resources}
        documents[route.PREPARATION['path']]={'source_inputs':{'roles':roles,'resource_plan':resource},'registrations':storage_pins}
        helper=SimpleNamespace(document=lambda p:documents[p['path']],READ_BYTES=0)
        with patch.object(route,'actual_json',side_effect=lambda h,p:actual[str(p)]), patch.object(Path,'exists',return_value=True), \
                patch.object(route,'validate_completion') as dev, patch.object(route,'bind_finalized_train',
                    return_value={'source_registration':pin('train-registration.json'),'source_code_root':'old-frozen'}) as train:
            inputs,status=route.completion_inputs(helper)
            self.assertEqual(status['status'],'READY_FOR_INDEX_CONTROL_PREPARATION')
            self.assertEqual(status['roles']['train']['requests'],str(route.FINALIZED_TRAIN/'TEACHER_REQUESTS.json'))
            self.assertEqual(status['roles']['dev']['requests'],str(route.BASE/'caches/full_future_metadata_dev_v1/TEACHER_REQUESTS.json'))
            self.assertEqual(inputs['roles']['train']['original_source_code_root'],'old-frozen')
            self.assertEqual(dev.call_args.args[1],'dev');self.assertEqual(train.call_count,1)


if __name__=='__main__':unittest.main(verbosity=2)
