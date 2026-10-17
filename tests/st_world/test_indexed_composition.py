"""CPU fixtures only; no real-data/GPU numerical qualification is claimed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.st_world_policy import qualify_indexed_provider as q
from tools.st_world_policy import teacher_provider as tp
from starVLA.dataloader.st_world_metadata_index import build_index
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.dataloader.st_world_ondemand import TeacherBatch
from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader
from starVLA.dataloader.st_world_schema import identity_hash
from test_indexed_provider_factory import artifacts
from test_fixed_provider_training import sign


def setup(tmp_path,scene_count=1):
    a=artifacts(tmp_path)
    cfg_path=Path(a.d['teacher_config']['path']);cfg=json.loads(cfg_path.read_text())
    teacher=tmp_path/'teacher';teacher.mkdir();(teacher/'model.safetensors').write_bytes(b'fixture-not-weights')
    cfg['teacher_root']=str(teacher);atomic_json(cfg_path,cfg)
    p=json.loads((tmp_path/'RESOLVED_PLAN.json').read_text())
    p['inputs']=[tp.pin(r['path']) for r in p['inputs']];atomic_json(tmp_path/'RESOLVED_PLAN.json',sign(p))
    result=json.loads(a.report.read_text());result['plan_identity']=p['identity'];atomic_json(a.report,sign(result))
    # A new explicit input namespace/population cannot reuse old admission.
    m=json.loads(a.requests.read_text());m['scenes'][0]['token']='new-population'
    if scene_count==2:
        second=deepcopy(m['scenes'][0]);second['token']='new-population-2';second['frames'][1]['views'][1]=None
        m['scenes'].append(second)
    sign(m)
    requests=tmp_path/'new_requests.json';atomic_json(requests,m)
    index=tmp_path/'new_index';build_index(index,requests=requests,expected_request_identity=m['identity'],contract=a.fit.contract,max_scenes=32)
    plan=q.make_plan(index,a.source_index,cfg_path,a.report)
    return a,plan


class FakeExecutor:
    def __init__(self,plan):
        self.execution=plan['execution'];self.execution_identity=plan['execution_identity']
        self.contract_identity=plan['provider']['contract']['identity'] if 'identity' in plan['provider']['contract'] else q.TeacherFeatureContract.from_dict(plan['contract']).identity
        self.compute_stats={'calls':0,'requested_images':0,'compute_image_slots':0,'padding_image_slots':0}
        self.rng_checks=0;self.bad=False
    def compute_batch(self,records,*,contract,execution_identity):
        self.compute_stats['calls']+=1;self.compute_stats['requested_images']+=len(records)
        self.compute_stats['compute_image_slots']+=8;self.compute_stats['padding_image_slots']+=8-len(records)
        self.rng_checks+=1
        z=torch.zeros((len(records),*contract.output_grid_hw,contract.channels),dtype=torch.float32)
        if self.bad and self.compute_stats['calls']>2:z+=1
        return TeacherBatch(tuple(r['key'] for r in records),contract.identity,execution_identity,z,torch.ones(z.shape[:-1],dtype=torch.bool))


class FakeBudget:
    mmap_bytes=0
    def check(self):return None


def exercise(plan):
    req,src=q.open_pair(plan);e=FakeExecutor(plan)
    try:
        source=IndexedTeacherCacheReader(src,expected_identity=plan['current_identity'])
        rng=lambda:q.tensor_digest(torch.get_rng_state())
        r=q.qualify_layouts(plan,req,source,e,rng=rng,budget=FakeBudget())
    finally:req.close();src.close()
    r.update(schema=q.RESULT_SCHEMA,plan_identity=plan['identity'],status='INDEXED_COMPOSITION_BITWISE_QUALIFIED',errors=[],
        execution=plan['execution'],execution_identity=plan['execution_identity'],tensor_disk_bytes=0,
        construction_rng_unchanged=True,completed_all_cases=True,fixed_policy_repeatability=True,
        legacy_reference_compatibility=True,frozen_initial_versions_hash='f'*64,frozen_final_versions_hash='f'*64,
        batch_calls=e.compute_stats['calls'],image_forwards=e.compute_stats['compute_image_slots'],
        compute_stats=e.compute_stats,rng_forward_checks=e.rng_checks,
        peak_rss_bytes=100*q.MiB,cuda_peak_allocated_bytes=0,elapsed_seconds=1,process_rchar_bytes=0,tensor_mmap_logical_bytes=0)
    return sign(r)


def descriptor(plan,result,tmp_path):
    out=tmp_path/'composition_run';out.mkdir()
    atomic_json(out/'RESOLVED_PLAN.json',plan);atomic_json(out/'RESULT.json',result)
    path=tmp_path/'COMPOSITION_PROVIDER.json'
    d=tp.prepare_indexed(Path(plan['request_index']['complete']['path']).parent,
        [Path(plan['source_index']['complete']['path']).parent],plan['teacher_config']['path'],out/'RESULT.json',path,limits=plan['limits'])
    return d,path


def test_plan_and_cli_are_metadata_only_and_exact_population(tmp_path,monkeypatch):
    a,p=setup(tmp_path)
    def forbidden(*args,**kw):raise AssertionError('No image/tensor/GPU/teacher access in plan')
    monkeypatch.setattr(IndexedTeacherCacheReader,'read_image',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    q.validate_plan(p)
    assert p['scenes']==1 and p['hit_keys'] and p['miss_keys']
    assert p['request_index']['request_identity']!=json.loads(a.requests.read_text())['identity']
    assert p['budget']['tensor_disk_bytes']==0 and p['budget']['max_image_forwards']==8*p['budget']['max_batch_calls']
    assert p['budget']['planned_input_upper_bytes']<64*q.GiB
    assert q.parser().parse_args(['run','--plan','x','--output','y']).command=='run'
    with pytest.raises(SystemExit):q.parser().parse_args(['run','--plan','x','--output','y','--tolerance','.1'])


def test_actual_adapter_layouts_masks_rng_and_factory_admission(tmp_path):
    a,p=setup(tmp_path);before=torch.get_rng_state().clone();r=exercise(p)
    assert torch.equal(before,torch.get_rng_state())
    assert r['partial_lru_exercised'] and r['mixed_cache_compute_exercised']
    assert r['fixed_policy_summary']['bitwise_equal'] and r['legacy_reference_summary']['bitwise_equal']
    d,path=descriptor(p,r,tmp_path)
    _,_,provider=tp.load_descriptor(tp.pin(path))
    try:
        assert provider.identity==p['provider'] and d['schema']==tp.INDEXED_SCHEMA
        assert provider.max_cached_bytes==p['limits']['max_cached_bytes']
    finally:tp.close_provider_metadata(provider)
    with pytest.raises(ValueError,match='outside completed qualification plan'):
        tp.prepare_indexed(Path(p['request_index']['complete']['path']).parent,[a.source_index],
            p['teacher_config']['path'],a.report,tmp_path/'cannot_borrow.json',limits=p['limits'])


@pytest.mark.parametrize('mutation',['prior_incomplete','prior_changed_execution','source_code','scope','scenes','budget','request_path','keys'])
def test_plan_negative_pre_gpu(tmp_path,monkeypatch,mutation):
    a,p=setup(tmp_path)
    if mutation.startswith('prior_'):
        r=json.loads(a.report.read_text())
        if mutation=='prior_incomplete':r['completed_all_cases']=False
        else:r['execution']['executor_source_sha256']='f'*64
        atomic_json(a.report,sign(r))
    elif mutation=='source_code':p['source_files'][0]['sha256']='0'*64
    elif mutation=='scope':p['scope']='FORMAL'
    elif mutation=='scenes':p['scenes']=33
    elif mutation=='budget':p['budget']['max_logical_input_bytes']=65*q.GiB
    elif mutation=='request_path':p['requests']=tp.pin(a.requests)
    elif mutation=='keys':p['miss_keys']=[]
    sign(p);path=tmp_path/'PLAN.json';atomic_json(path,p)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    with pytest.raises(ValueError):q.run(path,tmp_path/'never_written')
    assert not (tmp_path/'never_written').exists()


@pytest.mark.parametrize('mutation',['partial','mixed','legacy','repeat','mask','truncate','duplicate','rng','padding','source','execution','old_prior','restore','false_partial','sampler'])
def test_incomplete_or_resigned_bad_evidence_cannot_publish(tmp_path,mutation):
    a,p=setup(tmp_path);r=exercise(p)
    if mutation=='partial':r['partial_lru_exercised']=False
    elif mutation=='mixed':r['mixed_cache_compute_exercised']=False
    elif mutation=='legacy':r['legacy_cached_reference'][0]['difference']['max_abs']=1e-7
    elif mutation=='repeat':r['reference_reordered4'][0]['difference']['bitwise_equal']=False
    elif mutation=='mask':r['cases']['cold'][0]['all_masks_equal']=False
    elif mutation=='truncate':r['cases']['cold_restored'][0]['comparisons'].pop()
    elif mutation=='duplicate':r['cases']['reordered'][0]['comparisons'][1]['key']='0'*64
    elif mutation=='rng':r['rng_forward_checks']-=1
    elif mutation=='padding':r['compute_stats']['padding_image_slots']-=1
    elif mutation=='source':p['current_identity']='0'*64;sign(p);r['plan_identity']=p['identity']
    elif mutation=='execution':r['execution']=dict(r['execution'],tf32=True)
    elif mutation=='old_prior':r['schema']='st_fixed_teacher_qualification_v1'
    elif mutation=='restore':r['frozen_final_versions_hash']='0'*64
    elif mutation=='false_partial':r['cases']['partial_lru'][0]['counts']['computed_images']=0
    elif mutation=='sampler':r['cases']['cold_restored'][0]['sampler_ids'][0]='0'*64
    sign(r)
    with pytest.raises(ValueError):descriptor(p,r,tmp_path)
    assert not (tmp_path/'COMPOSITION_PROVIDER.json').exists()


def test_changed_independent_outputs_are_visible_without_relaxation(tmp_path):
    _,p=setup(tmp_path);req,src=q.open_pair(p);e=FakeExecutor(p);e.bad=True
    try:
        r=q.qualify_layouts(p,req,IndexedTeacherCacheReader(src,expected_identity=p['current_identity']),e,
            rng=lambda:None,budget=FakeBudget())
        assert not r['fixed_policy_summary']['bitwise_equal']
        # Source-vs-online comparison remains separate, unaffected by later compute change.
        assert r['legacy_reference_summary']['bitwise_equal']
    finally:req.close();src.close()


def test_legacy_numerical_difference_is_separate_from_fixed_repeatability(tmp_path):
    _,p=setup(tmp_path);req,src=q.open_pair(p);e=FakeExecutor(p)
    try:
        source=IndexedTeacherCacheReader(src,expected_identity=p['current_identity'])
        original=source.read_image
        def changed(key):
            z,m=original(key);return z+1,m
        source.read_image=changed
        r=q.qualify_layouts(p,req,source,e,rng=lambda:None,budget=FakeBudget())
        assert r['fixed_policy_summary']['bitwise_equal']
        assert not r['legacy_reference_summary']['bitwise_equal']
        assert any(not c['online_difference']['bitwise_equal'] for rows in r['cases'].values() for row in rows for c in row['comparisons'])
    finally:req.close();src.close()


def test_existing_shard_first_use_checksum_still_runs_once(tmp_path,monkeypatch):
    _,p=setup(tmp_path)
    from starVLA.dataloader import st_world_targets as targets
    original=targets.file_sha256;seen=[]
    def record(path):
        if str(path).endswith('.safetensors'):seen.append(str(path))
        return original(path)
    monkeypatch.setattr(targets,'file_sha256',record)
    exercise(p)
    assert len(seen)==len(set(seen))==len(p['source_shards'])


def test_budget_exhaustion_is_retryable_incomplete_not_qualification(monkeypatch):
    monkeypatch.setattr(q,'_rchar',lambda:0)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    b=q.Budget({'max_seconds':900,'ram_peak_bytes':100*q.GiB,'max_logical_input_bytes':100,
        'gpu_peak_bytes':16*q.GiB,'max_image_forwards':8,'max_batch_calls':1})
    b.admit(8)
    with pytest.raises(TimeoutError,match='fixed compute'):b.admit(1)
    b.mmap_bytes=101
    with pytest.raises(TimeoutError,match='logical input'):b.check()


def test_multiple_scenes_reorder_shared_keys_and_distinct_missing_masks(tmp_path):
    _,p=setup(tmp_path,scene_count=2);r=exercise(p)
    assert p['scenes']==2 and p['missing_slots']>0
    assert [v['scene'] for v in r['cases']['reordered']]==list(reversed(p['scene_ids']))
    assert len(r['cases']['cold'][0]['comparisons'])!=len(r['cases']['cold'][1]['comparisons'])
    assert r['all_masks_equal'] and r['sampler_equal']
    descriptor(p,r,tmp_path)


def test_receipt_strict_full_tensor_summaries_and_integer_ledgers(tmp_path):
    _,p=setup(tmp_path);valid=exercise(p)
    d={'requests':p['requests'],'request_index':p['request_index'],
       'sources':[{'index':p['source_index'],'cache_identity':p['current_identity']}],
       'teacher_config':p['teacher_config'],'execution':p['execution'],
       'limits':p['limits'],'provider':p['provider']}
    q.validate_receipt(p,valid,d)
    mutations={
        'elements_zero':lambda r:r['legacy_cached_reference'][0]['difference'].update(elements=0),
        'elements_bool':lambda r:r['reference_reordered4'][0]['difference'].update(elements=True),
        'online_one_element':lambda r:r['cases']['cold'][0]['comparisons'][0]['online_difference'].update(elements=1),
        'missing_statistic':lambda r:r['reference_reordered4'][0]['difference'].pop('sum_squared'),
        'contradictory_squared':lambda r:r['reference_reordered4'][0]['difference'].update(sum_squared=1.),
        'negative_reference':lambda r:r['legacy_cached_reference'][0]['difference'].update(reference_squared=-1.),
        'failed_legacy_summary':lambda r:r['legacy_reference_summary'].update(bitwise_equal=False,max_abs=1.),
        'truncated_fixed_summary':lambda r:r['fixed_policy_summary'].update(comparisons=1),
        'summary_bool':lambda r:r['fixed_policy_summary'].update(rms=False),
        'negative_requests':lambda r:r['compute_stats'].update(requested_images=-1,
            padding_image_slots=r['image_forwards']+1),
        'requested_bool':lambda r:r['compute_stats'].update(requested_images=True),
        'fractional_padding':lambda r:r['compute_stats'].update(padding_image_slots=float(r['compute_stats']['padding_image_slots'])),
        'false_read_bytes':lambda r:r.update(process_rchar_bytes=True),
        'negative_mmap_bytes':lambda r:r.update(tensor_mmap_logical_bytes=-1),
        'float_rss_bytes':lambda r:r.update(peak_rss_bytes=float(r['peak_rss_bytes'])),
        'bool_gpu_bytes':lambda r:r.update(cuda_peak_allocated_bytes=False),
        'bool_elapsed':lambda r:r.update(elapsed_seconds=True),
        'bool_layout_count':lambda r:r['cases']['cold'][0]['counts'].update(ram_hits=False),
        'wrong_provider_total':lambda r:r['provider_memory'].update(computed_images=0),
        'bool_provider_peak':lambda r:r['provider_memory'].update(peak_resident_bytes=True),
        'wrong_reference_digest':lambda r:r['reference_digests'][p['keys'][0]].update(features='invalid'),
        'different_sample_schema':lambda r:r['cases']['cold'][0].update(sampler_ids=['0'*64]),
        'reordered_reference':lambda r:r['reference_reordered4'].reverse(),
        'same_total_wrong_calls':lambda r:r.update(batch_calls=r['batch_calls']+1),
    }
    for name,mutate in mutations.items():
        report=deepcopy(valid);mutate(report);sign(report)
        with pytest.raises(ValueError,match='Composition|composition|counter|summary|sample|reference|RNG|Provider|hash|integer|budget|mask|key|Sampling'):
            q.validate_receipt(p,report,d)
