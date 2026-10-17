"""CPU-only negative fixtures: no actual request body/index/ledger/model is opened."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import pytest

spec=importlib.util.spec_from_file_location('index_control',Path(__file__).with_name('run_index_once.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('remaining,accepted',[(2183*m.GiB,False),(2303*m.GiB-1,False),
    (2303*m.GiB,True),(2383*m.GiB,True),(float(2303*m.GiB),False),(True,False)])
def test_checkpoint_obligations_retain_new_t1_reservation(remaining,accepted):
    cp={'schema':'st_checkpoint_storage_commitments_v1','status':'CURRENT',
        'remaining_bytes':remaining,'free_floor_bytes':1<<40,'valid_until_unix':20000}
    if accepted:m.validate_checkpoint_commitments(cp,19999)
    else:
        with pytest.raises(ValueError):m.validate_checkpoint_commitments(cp,19999)


@pytest.mark.parametrize('change',['floor','expiry','nonfinite','status'])
def test_more_checkpoint_space_does_not_bypass_other_guards(change):
    cp={'schema':'st_checkpoint_storage_commitments_v1','status':'CURRENT',
        'remaining_bytes':2383*m.GiB,'free_floor_bytes':1<<40,'valid_until_unix':20000}
    if change=='floor':cp['free_floor_bytes']=0
    elif change=='expiry':cp['valid_until_unix']=19000
    elif change=='nonfinite':cp['valid_until_unix']=float('inf')
    else:cp['status']='HISTORICAL'
    with pytest.raises(ValueError):m.validate_checkpoint_commitments(cp,19999)


def preparation(role='dev'):
    bindings={r:{'status':'ACTUAL_COMPLETE_BOUND','complete':{'path':'/fixture/'+r+'/COMPLETE.json','sha256':'a'*64},
        'request_identity':'b'*64,'request_file_sha256':'c'*64,'scenes':m.COUNTS[r]} for r in m.COUNTS}
    p={'schema':'st_full_future_metadata_controls_v1','phase':'index','index_phase':'EXACT_REQUEST_BOUND',
        'dispatch_authorized':False,'build_executed':False,'new_teacher_tensor_bytes':0,'actual_completion_bindings':bindings,
        'source_inputs':{'roles':{r:{'future_complete':bindings[r]['complete']} for r in m.COUNTS}},
        'registrations':{r:{'path':'/fixture/'+r+'/REGISTRATION.json','sha256':'d'*64} for r in m.COUNTS}}
    done={'schema':'st_future_metadata_complete_v1','status':'COMPLETE','scope':'full_population','request_identity':'b'*64,
        'scenes':m.COUNTS[role],'training_admission':False,'teacher_extractions':0,'tensor_disk_bytes':0,'files':{'TEACHER_REQUESTS.json':'c'*64}}
    size=(56 if role=='dev' else 1008)*m.MiB
    reg={'schema':'st_metadata_build_registration_v1','writer_kind':'future_index','callback_protocol':'st_metadata_storage_check_v1',
        'command':m.expected_command(role,'b'*64),'storage_requirements':{'output_bytes':size,'staging_bytes':size,'metadata_bytes':(16 if role=='dev' else 32)*m.MiB},
        'limits':{'max_disk_bytes':m.limits(role)['root_bytes'],'max_rss_bytes':4*m.GiB,'max_seconds':m.limits(role)['wall_seconds']}}
    return p,reg,done


def index_result(role='dev'):
    p,_,_=preparation(role);contract={'contract':{'fixture':'original_contract_only'}}
    done={'schema':'st_teacher_metadata_index_v1','status':'COMPLETE','encoding':'canonical_json_zlib_level1_v1',
        'request_identity':'b'*64,'cache_identity':None,'cache_root':None,'scene_count':m.COUNTS[role],'record_count':400,
        'database_bytes':4096,'database_sha256':'f'*64,'builder_source_sha256':m.INDEX_BUILDER_SHA,'builder_peak_rss_bytes':10<<20,'tensor_files_opened':0,'image_files_opened':0,'tensor_disk_bytes':0,
        'coverage':'exact_original_manifest','training_admission':False,'contract':contract['contract'],
        'max_disk_bytes':(112 if role=='dev' else 2016)*m.MiB,'max_builder_rss_bytes':4*m.GiB,
        'dependencies':{'requests':{'path':m.expected_command(role,'b'*64)[5],'sha256':'c'*64}}}
    done['identity']=m.digest(done)
    output={'status':'GUARDED_METADATA_RETURN','registration_sha256':'d'*64,'artifact':{k:done[k] for k in m.FIELDS}}
    return p,contract,done,output


@pytest.mark.parametrize('role',['dev','train'])
def test_real_index_success_schema_has_no_artifact_status(role):
    p,reg,future=preparation(role);m.validate_preparation(p,role,reg,future)
    p,contract,done,output=index_result(role)
    assert 'status' not in output['artifact']
    assert m.validate_result(role,p,output,done,contract)==done['identity']
    assert sum(reg['storage_requirements'].values())==m.limits(role)['root_bytes']
    assert m.limits(role)['read_bytes']==(4 if role=='dev' else 64)*m.GiB


@pytest.mark.parametrize('change',['missing_train','waiting','wrong_scene','wrong_identity','old_phase','scope','teacher','bool_zero','argv','root_budget','runtime_limit'])
def test_invalid_actual_input_or_registration_rejects(change):
    p,reg,done=preparation()
    if change=='missing_train':del p['actual_completion_bindings']['train']
    elif change=='waiting':p['actual_completion_bindings']['train']['status']='WAITING_ACTUAL_COMPLETE'
    elif change=='wrong_scene':done['scenes']=32
    elif change=='wrong_identity':done['request_identity']='e'*64
    elif change=='old_phase':p['phase']='preparation'
    elif change=='scope':done['scope']='smoke'
    elif change=='teacher':done['teacher_extractions']=1
    elif change=='bool_zero':done['tensor_disk_bytes']=False
    elif change=='argv':reg['command']+=['--cache-root','/fixture/tensors']
    elif change=='root_budget':reg['limits']['max_disk_bytes']*=2
    else:reg['limits']['max_seconds']=14400
    with pytest.raises(ValueError):m.validate_preparation(p,'dev',reg,done)


@pytest.mark.parametrize('change',['status','identity','wrong_request','scene','cache','coverage','training','image','tensor','tensor_disk',
    'bool_count','zero_record','db_limit','rss_limit','dep','contract','cli_field','registration','cli_status','builder','database_hash','cache_header'])
def test_result_negative_and_resigned_inconsistency_rejects(change):
    p,contract,done,output=index_result()
    if change=='status':done['status']='PARTIAL'
    elif change=='identity':done['identity']='f'*64
    elif change=='wrong_request':done['request_identity']='e'*64
    elif change=='scene':done['scene_count']=32
    elif change=='cache':done['cache_identity']='e'*64
    elif change=='coverage':done['coverage']='sampled'
    elif change=='training':done['training_admission']=True
    elif change=='image':done['image_files_opened']=1
    elif change=='tensor':done['tensor_files_opened']=1
    elif change=='tensor_disk':done['tensor_disk_bytes']=1
    elif change=='bool_count':done['tensor_files_opened']=False
    elif change=='zero_record':done['record_count']=0
    elif change=='db_limit':done['database_bytes']=128<<20
    elif change=='rss_limit':done['builder_peak_rss_bytes']=5<<30
    elif change=='dep':done['dependencies']['requests']['sha256']='f'*64
    elif change=='contract':done['contract']={}
    elif change=='builder':done['builder_source_sha256']='e'*64
    elif change=='database_hash':done['database_sha256']='not-a-hash'
    elif change=='cache_header':done['cache_header']={'illegal':'external_cache'}
    elif change=='cli_field':output['artifact']['extra']=0
    elif change=='registration':output['registration_sha256']='f'*64
    else:output['status']='COMPLETE'
    if change not in ('identity','cli_field','registration','cli_status'):
        done['identity']=m.digest({k:v for k,v in done.items() if k!='identity'})
        output['artifact']={k:done[k] for k in m.FIELDS}
    with pytest.raises(ValueError):m.validate_result('dev',p,output,done,contract)


def admission():
    return {'schema':'st_full_future_index_resource_admission_v1','status':'PASS_BOUNDED_DEV_INDEX_ONLY','role':'dev',
        'limits':m.limits('dev'),'hostname':m.socket.gethostname(),'reviewer':'independent_fixture',
        'monitor_core':{'path':str(m.CORE_PATH),'sha256':m.CORE_SHA},'issued_at_unix':1,'latest_start_unix':100,'valid_until_unix':20000}


@pytest.mark.parametrize('change',['valid','wrong_schema','wrong_status','gpu','retry','expired','short','core','reviewer','bool_cpu'])
def test_exact_index_admission(change,monkeypatch):
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {0,1});a=admission()
    if change=='wrong_schema':a['schema']='st_full_future_build_resource_admission_v1'
    elif change=='wrong_status':a['status']='PASS_BOUNDED_DEV_BUILD_ONLY'
    elif change=='gpu':a['limits']['gpu']=True
    elif change=='retry':a['limits']['attempts']=2
    elif change=='expired':a['latest_start_unix']=5
    elif change=='short':a['valid_until_unix']=100
    elif change=='core':a['monitor_core']['sha256']='0'*64
    elif change=='reviewer':a['reviewer']='/root/st_indexed_composition'
    elif change=='bool_cpu':a['limits']['cpu_affinity']=[False,1]
    if change=='valid':assert m.validate_admission(a,10)=='dev'
    else:
        with pytest.raises(ValueError):m.validate_admission(a,10)


def test_small_actual_complete_rejects_missing_symlink_and_oversize(tmp_path):
    p=tmp_path/'COMPLETE.json'
    with pytest.raises(FileNotFoundError):m.read_small_actual(p)
    p.write_text('{}');link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):m.read_small_actual(link)
    value,pin=m.read_small_actual(p)
    assert value=={} and pin['sha256']==hashlib.sha256(b'{}').hexdigest()
    p.write_bytes(b'x'*(m.MiB+1))
    with pytest.raises(ValueError):m.read_small_actual(p)


def test_import_keeps_torch_and_native_indexer_unloaded():
    assert 'torch' not in sys.modules
    assert 'tools.st_world_policy.index_teacher_metadata' not in sys.modules


@pytest.mark.parametrize('change',['none','core','drop_old','change_old','extra','registration'])
def test_additive_policy_preserves_all_old_allocations(change):
    p,_,_=preparation();old={'metadata_package':{'schema':'v2','maximum_bytes':8<<30},'frozen_w_package':{'unchanged':True},
        'claims':[{'root':'/fixture/old','exact':123}],'legacy_reservations':[{'unchanged':1}]}
    new=copy.deepcopy(old)
    for role in m.COUNTS:
        pin=p['registrations'][role];root=str(m.BASE/('caches/full_future_request_index_'+role+'_v1'));size=(56 if role=='dev' else 1008)*m.MiB
        new['claims'].append({'root':root,'claim_id':'full-future-'+role+'-index-v1','identity_kind':'st_metadata_build_registration_v1',
            'cache_identity':pin['sha256'],'registration_path':pin['path'],'registration_sha256':pin['sha256'],
            'output_bytes':size,'staging_bytes':size,'metadata_bytes':(16 if role=='dev' else 32)*m.MiB})
    if change=='core':new['metadata_package']['maximum_bytes']=16<<30
    elif change=='drop_old':new['claims'].pop(0)
    elif change=='change_old':new['claims'][0]['exact']=0
    elif change=='extra':new['claims'].append({'root':'/fixture/new'})
    elif change=='registration':new['claims'][1]['registration_sha256']='e'*64
    if change=='none':m.validate_additive_policy(p,old,new)
    else:
        with pytest.raises(ValueError):m.validate_additive_policy(p,old,new)
