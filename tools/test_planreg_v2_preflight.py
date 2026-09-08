"""Targeted admission/placement tests; no replacement model or relaxed scientific gates."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import planreg_v2_preflight as guard
from planreg_v2_gpu_guard import visible_gpu_processes
from planreg_v2_cluster import training_commands
from navsim.agents.EpisodeDrive.planreg_v2 import CACHE_SCHEMA,LONG_TARGET_VERSION,NORMALIZER_SCHEMA
from navsim.agents.EpisodeDrive.planreg_v2.runtime import validate_formal,load_config


@pytest.fixture
def full_data():
    tokens=['%016x'%i for i in range(103288)]
    token_hash=hashlib.sha256('\n'.join(tokens).encode()).hexdigest()
    manifest=dict(records=[dict(token=t) for t in tokens],token_sha256=token_hash,raw_gt_sha256='raw_gt',
        statistics_contract='original_gt_only',split='trainval_final_fit',smoke=False,schema=CACHE_SCHEMA,long_target_version=LONG_TARGET_VERSION)
    normalizer=dict(mean=torch.zeros(8,3).tolist(),std=torch.ones(8,3).tolist(),metadata=dict(schema=NORMALIZER_SCHEMA,
        mode='stepwise_zscore',count=103288,split='trainval_final_fit',token_sha256=token_hash,raw_gt_sha256='raw_gt',statistics_contract='original_gt_only'))
    return manifest,normalizer


def test_full_statistics_contract_passes(full_data):guard.validate_full_statistics(*full_data)


@pytest.mark.parametrize('field,value',[('count',48),('split','train'),('raw_gt_sha256','different'),
    ('token_sha256','different'),('schema','old'),('statistics_contract','long_targets')])
def test_missing_smoke_or_stale_statistics_rejected(full_data,field,value):
    manifest,normalizer=full_data;normalizer['metadata'][field]=value
    with pytest.raises(ValueError):guard.validate_full_statistics(manifest,normalizer)


def test_duplicate_scene_rejected(full_data):
    manifest,normalizer=full_data;manifest['records'][-1]=manifest['records'][0]
    with pytest.raises(ValueError):guard.validate_full_statistics(manifest,normalizer)


@pytest.mark.parametrize('shape',[(3,),(1,8,3)])
def test_not_stepwise_statistics_rejected(full_data,shape):
    manifest,normalizer=full_data;normalizer['mean']=torch.zeros(shape).tolist()
    with pytest.raises(ValueError):guard.validate_full_statistics(manifest,normalizer)


def test_old_cache_rejected_before_model(full_data):
    manifest,_=full_data;manifest['long_target_version']='uniform_long_v1'
    from navsim.agents.EpisodeDrive.planreg_v2 import ARCHITECTURE_VERSION,RECIPE_VERSION,SCHEDULE_VERSION
    cfg=dict(architecture_version=ARCHITECTURE_VERSION,recipe_version=RECIPE_VERSION,schedule_version=SCHEDULE_VERSION)
    with pytest.raises(ValueError,match='Recompute progressive-long'):validate_formal(cfg,manifest,{})


def test_old_shared_bank_rejected():
    artifact=dict(schema='old_std1e-6',identity={},tensors={},trainable_state={},seed=0,config={})
    with pytest.raises(ValueError,match='Old shared init'):guard.validate_shared(artifact,{})


def summary_and_range():
    fp=guard.source_fingerprint()['sha256']
    return (dict(schema=guard.SCAN_SCHEMA,status='PASS',audited_count=103288,manifest_sha256='manifest',
        preprocessing=guard.preprocessing_identity(),raw_gt_statistics_recomputed_equal=True,errors=[],
        input_ranges=dict(max_tiles=9,max_prefix_tokens=2560)),dict(max_tiles=9),
        dict(source_fingerprint_sha256=fp,max_prefix_tokens=2560))


def test_matching_input_envelope_passes():
    summary,layout,profile=summary_and_range();guard.validate_input_range(summary,'manifest',layout,profile)


@pytest.mark.parametrize('failure',['tiles','prompt','stale','incomplete','data_error'])
def test_unmeasured_or_stale_input_rejected(failure):
    summary,layout,profile=summary_and_range()
    if failure=='tiles':summary['input_ranges']['max_tiles']=13
    elif failure=='prompt':summary['input_ranges']['max_prefix_tokens']=2800
    elif failure=='stale':summary['manifest_sha256']='changed'
    elif failure=='incomplete':summary['audited_count']=48
    else:summary['errors']=['missing image']
    with pytest.raises(ValueError):guard.validate_input_range(summary,'manifest',layout,profile)


def test_hardware_match_passes_and_mismatch_fails():
    actual=dict(hardware=[dict(name='NVIDIA A800-SXM4-80GB',memory_bytes=85094825984,free_bytes=78*2**30)],external_gpu_processes=[])
    expected=copy.deepcopy(actual);guard.validate_hardware(actual,expected)
    expected['hardware'][0]['name']='Different GPU'
    with pytest.raises(ValueError,match='hardware differs'):guard.validate_hardware(actual,expected)
    expected=copy.deepcopy(actual);actual['external_gpu_processes']=[123]
    with pytest.raises(ValueError,match='Other GPU tasks'):guard.validate_hardware(actual,expected)


def test_gpu_guard_never_blocks_or_touches_reserved_devices():
    def query(fields,kind):
        return '0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n' if kind=='gpu' else '101, GPU-a\n202, GPU-c\n'
    assert visible_gpu_processes('2,3',query)==[202]
    assert visible_gpu_processes('GPU-b,GPU-d',query)==[]
    with pytest.raises(ValueError):visible_gpu_processes('GPU-ambiguous',query)


def test_layout_wrapper_requires_real_matching_flat_artifact(tmp_path):
    report=tmp_path/'measured.json';report.write_text('{}')
    flat=tmp_path/'layout.json';value=dict(global_batch=128,report_path=str(report),source_sha256=guard.sha(report));flat.write_text(json.dumps(value))
    wrapper=tmp_path/'wrapper.json';wrapper.write_text(json.dumps(dict(evidence=value,artifact_path=str(flat),artifact_sha256=guard.sha(flat))))
    assert guard.flat_layout(wrapper)==(value,str(flat.resolve()))
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='actual measured'):guard.flat_layout(wrapper)


def test_unequal_node_counts_keep_gb128_without_training(monkeypatch):
    nodes=[dict(host='local' if i==0 else 'node'+str(i),gpus=list(range(n))) for i,n in enumerate([8,8,6,6,4])]
    spec=dict(nodes=nodes,mode='profile',microbatch=4,accumulate=1,workers=2,master_addr='127.0.0.1',
        master_port=29500,config='base.yaml',manifest='input.json',run_output='new_profile')
    commands=training_commands(spec)
    assert len(commands)==5 and all('--profile-only' in ' '.join(c) for c in commands)
    spec['mode']='formal';spec['layout']='measured.json';monkeypatch.delenv('START_FORMAL_AFTER_PREFLIGHT',raising=False)
    with pytest.raises(ValueError,match='START_FORMAL_AFTER_PREFLIGHT'):training_commands(spec)
    monkeypatch.setenv('START_FORMAL_AFTER_PREFLIGHT','1')
    formal=training_commands(spec)
    for c in formal:
        assert '--layout-lock' in ' '.join(c)
        assert not any(x in ' '.join(c) for x in ('--profile-only','--smoke-steps','--run-step-limit','--debug-short-schedule','--warm-start'))
    spec['accumulate']=2
    with pytest.raises(ValueError,match='128'):training_commands(spec)
