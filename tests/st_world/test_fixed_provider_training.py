"""CPU metadata fixtures, never a real fixed-teacher qualification."""
from copy import deepcopy
import json
from types import SimpleNamespace
from pathlib import Path

from PIL import Image
import pytest

from starVLA.dataloader.st_world_history import image_reference, file_sha256
from starVLA.dataloader.st_world_schema import identity_hash
from tools.st_world_policy import fixed_batch_teacher as fixed
from tools.st_world_policy import ondemand_teacher as legacy
from tools.st_world_policy import qualify_fixed_batch_teacher as qualifier
from tools.st_world_policy.teacher_provider import (
    FIXED_CASES, prepare, pin, load_descriptor, create_provider, runtime_metrics, configure_smoke)
from test_provider_training import provider_artifacts, write


def sign(value):
    value['identity'] = identity_hash({k:v for k,v in value.items() if k != 'identity'})
    return value


def fixed_artifacts(tmp_path, *, legacy_ok=False):
    a = provider_artifacts(tmp_path)
    manifest = json.loads(a.requests.read_text())
    manifest['times_s'] = [0., .5, 1., 2., 4.]
    for ti, t in enumerate(manifest['times_s'][1:]):
        refs = []
        for view in range(3):
            path = tmp_path / f'future{ti}_{view}.png'
            Image.new('RGB',(32,18),(ti*3+view,0,0)).save(path)
            refs.append(image_reference(path))
        manifest['scenes'][0]['frames'].append({'requested_s':t,'timestamp_us':int(t*1e6),'views':refs})
    write(a.requests,sign(manifest))
    execution = deepcopy(a.d['execution'])
    backend = {'tf32':False,'autocast':False,'deterministic_algorithms':True,
        'deterministic_warn_only':False,'cublas_workspace_config':':4096:8',
        'cudnn_benchmark':False,'cudnn_deterministic':False,'flash_sdp_enabled':True,
        'mem_efficient_sdp_enabled':True,'math_sdp_enabled':True}
    execution.update(backend, schema='st_fixed8_dino_execution_v1',batch_policy=deepcopy(fixed.FIXED_BATCH_POLICY),
        executor_source_sha256=file_sha256(fixed.__file__),legacy_executor_source_sha256=file_sha256(legacy.__file__))
    record_bytes = 18*24*(1024*4+1)
    plan = {'schema':'st_fixed_teacher_qualification_plan_v1','requests_path':str(a.requests),
        'teacher_config':a.d['teacher_config']['path'], 'inputs':[pin(a.requests),a.d['teacher_config']],
        'source_files':[pin(fixed.__file__),pin(legacy.__file__),pin(qualifier.__file__)],
        'runner_source_sha256':file_sha256(qualifier.__file__),'contract':a.fit.contract.to_dict(),
        'current_identity':a.current.identity['identity'],'batch_policy':deepcopy(fixed.FIXED_BATCH_POLICY),
        'cases':list(FIXED_CASES),'backend':backend,'scenes':1,'current_unique_images':1,'unique_images':13,
        'budget':{'resident_provider_bytes':8*record_bytes,'max_scene_bytes':64<<20}}
    write(tmp_path/'RESOLVED_PLAN.json',sign(plan))
    metric = {'bitwise_equal':True,'max_abs':0.,'comparisons':1,'elements':1024}
    cases = {}
    for name,count in zip(FIXED_CASES,[13,1,1,1,2,6]):
        rows = [{'fixed8_reference':deepcopy(metric),'cold_reference':deepcopy(metric)} for _ in range(count)]
        cases[name] = {'all_masks_equal':True,'records':rows,'fixed8_reference':deepcopy(metric)}
    cases[FIXED_CASES[-1]].update(partial_lru_exercised=True,sampler_equal=True)
    report = {'schema':'st_fixed_teacher_qualification_v1','plan_identity':plan['identity'],
        'status':'FIXED_POLICY_AND_LEGACY_BITWISE_QUALIFIED' if legacy_ok else 'FIXED_POLICY_REPEATABLE_LEGACY_REVIEW_REQUIRED',
        'errors':[],'tensor_disk_bytes':0,'execution':execution,'execution_identity':identity_hash(execution),
        'batch_policy':deepcopy(fixed.FIXED_BATCH_POLICY),'backend':backend,'construction_rng_unchanged':True,
        'fixed_policy_repeatability':True,'legacy_reference_compatibility':legacy_ok,'completed_all_cases':True,
        'cases':cases,'fixed_policy_summary':deepcopy(metric),
        'legacy_reference_summary':dict(metric,bitwise_equal=legacy_ok,max_abs=0. if legacy_ok else 1e-5),
        'rng_forward_checks':20,'batch_calls':20,'image_forwards':160,
        'compute_stats':{'calls':20,'requested_images':13,'compute_image_slots':160,'padding_image_slots':147},
        'frozen_initial_versions_hash':'a'*64,'frozen_final_versions_hash':'a'*64}
    write(a.report,sign(report))
    a.fixed_limits = dict(a.d['limits'],max_cached_bytes=8*record_bytes)
    a.fixed_path = tmp_path/'FIXED_PROVIDER.json'
    return a


def prepare_fixed(a):
    return prepare(a.requests,[a.current.root],a.d['teacher_config']['path'],a.report,
                   a.fixed_path,limits=a.fixed_limits)


def test_fixed_repeatability_admits_only_qualified_bounded_lru_not_legacy_parity(tmp_path):
    a = fixed_artifacts(tmp_path)
    d = prepare_fixed(a)
    _,_,p = load_descriptor(pin(a.fixed_path))
    assert p.max_cached_bytes == a.fixed_limits['max_cached_bytes']
    assert p.identity['identity'] != a.d['provider']['identity']
    assert json.loads(a.report.read_text())['legacy_reference_compatibility'] is False
    assert d['execution']['schema'] == 'st_fixed8_dino_execution_v1'


def test_default_factory_selects_fixed_executor_and_preserves_compute_accounting(tmp_path,monkeypatch):
    a = fixed_artifacts(tmp_path); d = prepare_fixed(a); calls = []
    def fixed_factory(root,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(execution=d['execution'],execution_identity=identity_hash(d['execution']),
            contract_identity=a.fit.contract.identity,compute_stats={'calls':1,'requested_images':1,
                'compute_image_slots':8,'padding_image_slots':7})
    monkeypatch.setattr(fixed,'FixedBatchDINOExecutor',fixed_factory)
    monkeypatch.setattr(legacy,'FrozenDINOExecutor',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Wrong executor')))
    p = create_provider(pin(a.fixed_path),device='cpu')
    assert len(calls)==1 and calls[0]['expected_execution_identity']==p.execution_identity
    assert runtime_metrics(p,0)['fixed_compute']['compute_image_slots']==8


@pytest.mark.parametrize('mutation',[
    'not_complete','not_repeatable','missing_case','truncated_case','no_partial_lru','sampler_changed',
    'changed_cold_tensor','changed_regroup_tensor','rng_mismatch','frozen_mutation','padding_cost_hidden',
    'legacy_claim_laundered','backend_mismatch','execution_hash_mismatch','unknown_schema','result_hash_mismatch'])
def test_false_or_incomplete_fixed_qualification_never_publishes(tmp_path,mutation):
    a = fixed_artifacts(tmp_path); q = json.loads(a.report.read_text())
    if mutation=='not_complete':q['completed_all_cases']=False
    elif mutation=='not_repeatable':q['fixed_policy_repeatability']=False
    elif mutation=='missing_case':q['cases'].pop(FIXED_CASES[2])
    elif mutation=='truncated_case':q['cases'][FIXED_CASES[-1]]['records'].pop()
    elif mutation=='no_partial_lru':q['cases'][FIXED_CASES[-1]]['partial_lru_exercised']=False
    elif mutation=='sampler_changed':q['cases'][FIXED_CASES[-1]]['sampler_equal']=False
    elif mutation=='changed_cold_tensor':q['cases'][FIXED_CASES[-1]]['records'][0]['cold_reference']['max_abs']=1e-8
    elif mutation=='changed_regroup_tensor':q['cases'][FIXED_CASES[1]]['records'][0]['fixed8_reference']['bitwise_equal']=False
    elif mutation=='rng_mismatch':q['rng_forward_checks']=19
    elif mutation=='frozen_mutation':q['frozen_final_versions_hash']='b'*64
    elif mutation=='padding_cost_hidden':q['compute_stats']['compute_image_slots']=13
    elif mutation=='legacy_claim_laundered':q['legacy_reference_compatibility']=True
    elif mutation=='backend_mismatch':q['backend']['cublas_workspace_config']=':16:8'
    elif mutation=='execution_hash_mismatch':q['execution_identity']='c'*64
    elif mutation=='unknown_schema':q['schema']='st_ondemand_teacher_qualification_v1'
    elif mutation=='result_hash_mismatch':q['identity']='d'*64
    if mutation!='result_hash_mismatch':sign(q)
    write(a.report,q)
    with pytest.raises(ValueError):prepare_fixed(a)
    assert not a.fixed_path.exists()


def test_larger_than_tested_lru_rejected_before_descriptor_write(tmp_path):
    a = fixed_artifacts(tmp_path); a.fixed_limits['max_cached_bytes'] += 1
    with pytest.raises(ValueError,match='actually qualified'):prepare_fixed(a)
    assert not a.fixed_path.exists()


def test_legacy_schema_cannot_admit_new_fixed_lru(tmp_path):
    a=provider_artifacts(tmp_path)
    with pytest.raises(ValueError,match='partial-LRU'):
        prepare(a.requests,[a.current.root],a.d['teacher_config']['path'],a.report,
            tmp_path/'forbidden.json',limits=dict(a.d['limits'],max_cached_bytes=2<<20))


def test_fixed_smoke_preserves_real_official_backend_and_original_basis(tmp_path):
    from starVLA.dataloader.st_world_dataset import projected_teacher_contract
    from tools.st_world_policy.official_runtime import REFERENCE_RECIPE, BACKEND
    from tools.st_world_policy.training_config import official_runtime_policy
    from test_training_config import config
    a = fixed_artifacts(tmp_path, legacy_ok=True); prepare_fixed(a)
    cfg = config()
    cfg['execution'].update(scope='SMOKE', stop_after=2, sample_limit=1,
                            world_size=1, micro_batch=1, global_batch=2)
    cfg['runtime'].update(deterministic=True, gradient_checkpointing=False, profile_mode='none')
    effective = {**{k: cfg['execution'][k] for k in ('world_size','micro_batch','global_batch','max_updates')},
                 **cfg['optimizer'], 'gradient_checkpointing':False}
    cfg['runtime']['official_execution'] = {
        'schema':'st_official_drivedreamer_execution_v1', 'preset':'drivedreamer_f35fe783',
        'scope':'research_stage', 'adaptation_reason':'Explicit bounded CPU metadata test, not a GPU qualification.',
        'declared_recipe_overrides':sorted(k for k,v in REFERENCE_RECIPE.items() if effective[k] != v)}
    cfg['data']['pca_path'] = str(a.pca)
    cfg['space_time']['teacher'] = projected_teacher_contract(a.fit.contract,
        pca_identity=a.receipt['identity'],dimension=2).to_dict()
    before = {name:(a.pca/name).read_bytes() for name in ('PCA.json','COMPLETE.json','statistics.safetensors')}
    source = write(tmp_path/'official_input.json',cfg)
    output = tmp_path/'fixed_diagnostic'
    resolved = json.loads(Path(configure_smoke(source,a.fixed_path,a.fit.root,output)).read_text())
    assert resolved['runtime']['official_execution'] == cfg['runtime']['official_execution']
    assert official_runtime_policy(resolved)['backend'] == BACKEND
    assert resolved['optimizer'] == cfg['optimizer'] and resolved['execution'] == cfg['execution']
    assert resolved['runtime']['loader_workers'] == 0 and resolved['runtime']['gradient_checkpointing'] is False
    lineage = json.loads((output/'PROVIDER_CONFIG_LINEAGE.json').read_text())
    assert lineage['ram_reuse'] == 'qualified_fixed8_bounded_LRU'
    assert lineage['fixed_policy_repeatability'] is True and lineage['legacy_reference_compatibility'] is True
    assert lineage['formal_approved'] is False and lineage['training_started'] is False
    assert all((a.pca/name).read_bytes()==value for name,value in before.items())
