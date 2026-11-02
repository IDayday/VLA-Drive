"""CPU orchestration mocks only; these are not real world8 GPU evidence."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from tools.qwen_world_performance import reference_cache_native_qualification as Q
from starVLA.rl.qwen_world_performance import recovery

SOURCE=Path('/mnt/project/qwen-world-performance-rl-20261103/runtime/reference_cache_probe_v1')


def test_plan_three_existing_train_commands_no_dispatch_and_strict_pending(tmp_path):
    plan=Q.make_plan(tmp_path/'qualification',SOURCE)
    a,b,c=plan['stages']
    assert plan['status']=='PREPARED_NOT_DISPATCHED' and (plan['B'],plan['G'],plan['M'])==(128,16,4)
    assert all('tools.qwen_world_performance.train' in s['command'] and '--qualification' in s['command'] for s in plan['stages'])
    assert '--frozen-reference-cache' not in a['command'] and '--frozen-reference-cache' in b['command'] and '--frozen-reference-cache' in c['command']
    assert c['command'][c['command'].index('--resume')+1]==plan['cached2']
    assert b['command'][b['command'].index('--output')+1]==c['command'][c['command'].index('--output')+1]
    assert c['depends_on']=='cached2' and b['depends_on']=='original4'
    assert '--resume' not in a['command'] and '--resume' not in b['command']
    assert not (tmp_path/'qualification/jobs').exists()
    with pytest.raises(FileExistsError):Q.make_plan(tmp_path/'qualification',SOURCE)


def test_reject_other_B_G_M_algorithm(tmp_path):
    cfg=yaml.safe_load((Q.P/'configs/S0_LARGE_SAFETYEP.yaml').read_text())
    cfg['research']['performance']['algorithm']='awfm'
    p=tmp_path/'other.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='B128/G16/M4'):Q.make_plan(tmp_path/'q',SOURCE,config=p)


def mock_checkpoint(path,*,cache):
    path.mkdir();identity=dict(recipe={'research':{'performance':dict(scenes_per_collection=128,group_size=16,inner_updates=4)}},world=8)
    if cache:identity['execution_optimizations']={'frozen_IL_reference_cache':{}}
    identity['sha256']=Q.fingerprint(identity)
    (path/'identity.json').write_text(json.dumps(identity));common=dict(adapters={'p':torch.ones(2)},optimizer={'test':torch.ones(2)},scheduler={'last_epoch':2})
    torch.save(common,path/'optimizer.pt');summaries=[]
    for rank in range(8):
        summary=dict(phase='INNER_COMMITTED',rank=rank);summaries.append(summary)
        leaves=[]
        for n in range(16):
            leaf=dict(chain=torch.zeros(16,11,8,3),timesteps=torch.arange(10),deltas=torch.ones(10),stoch_idx=torch.tensor([7,8,9]),
                cond_bg={},old_logp=torch.zeros(16,3),rewards=torch.zeros(1,16),advantages=torch.zeros(16),adv_mask=torch.zeros(16),normalization_std=torch.tensor(1.),
                global_std=torch.tensor(1.),token=f'{rank}-{n}',log='log',valid_scene=True,stream_position={})
            if cache:leaf['frozen_IL_reference_terms']={'mock':True}
            leaves.append(leaf)
        state=dict(scene_stream=dict(run_identity=identity['sha256'],cursor=128),updates=2,next_inner=2,collections=1,scene_cursor=128,
            raw_scene_exposures=128,valid_scene_exposures=128,padded_scene_exposures=0,new_candidates=2048)
        torch.save(dict(identity_sha256=identity['sha256'],state=state,rollout={'samples':leaves},rng={'torch':torch.get_rng_state()}),path/f'rank{rank}.pt')
    (path/'transaction.json').write_text(json.dumps({'ranks':summaries}))
    files={p.name:Q.file_sha(p) for p in path.iterdir()}
    (path/'files.json').write_text(json.dumps(files))
    (path/'COMPLETE').write_text(json.dumps(dict(schema=2,world=8,updates=2,next_inner=2,phase='INNER_COMMITTED',files_sha256=Q.file_sha(path/'files.json'))))


def test_mock_native_reader_checks_all8_seals_and_cache_flag(tmp_path,monkeypatch):
    path=tmp_path/'cp';mock_checkpoint(path,cache=True);calls=[]
    def validate(state,window,identity,optimizer,scheduler,*,expected_rank):
        calls.append(expected_rank);return dict(phase='INNER_COMMITTED',rank=expected_rank)
    monkeypatch.setattr(recovery,'validate_boundary',validate)
    monkeypatch.setattr(recovery,'validate_ranks',lambda rows:len(rows)==8 or (_ for _ in ()).throw(ValueError()))
    value=Q.inspect_checkpoint(path,2,2,want_cache=True)
    assert calls==list(range(8)) and len(value['ranks'])==8
    assert all(v['cached_leaves']==16 for v in value['ranks'].values())
    with pytest.raises(ValueError,match='flag'):Q.inspect_checkpoint(path,2,2,want_cache=False)
    with (path/'rank7.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='content'):Q.inspect_checkpoint(path,2,2,want_cache=True)


def test_compare_never_relabels_nonzero_transport_as_pass(tmp_path):
    config=tmp_path/'cfg';config.write_text('actual')
    result=tmp_path/'RESULT.json';result.write_text(json.dumps({'returncode':-9}))
    with pytest.raises(ValueError,match='returncode0'):Q.compare(dict(config=str(config),config_sha256=Q.file_sha(config),stages=[dict(transport_result=str(result))]))


def test_migration_requires_formal_identity_and_both_real_qualification_schemas(tmp_path):
    from starVLA.rl.simwam_recipe.lineage import SCIENTIFIC_KEYS
    from starVLA.rl.qwen_world_research_algorithms.frozen_reference import PROTOCOL
    old={k:None for k in SCIENTIFIC_KEYS};old.update(purpose='formal',recipe={},source_files={})
    old['sha256']=Q.fingerprint(old)
    new={k:v for k,v in old.items() if k!='sha256'};new['execution_optimizations']={'frozen_IL_reference_cache':copy.deepcopy(PROTOCOL)};new['sha256']=Q.fingerprint(new)
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(status='PASS',schema='mock_CPU_only')))
    with pytest.raises(ValueError,match='singleGPU plus world8'):Q.migration_draft(old,new,tmp_path,[proof])
    new['purpose']='qualification_not_formal_initialization';new['sha256']=Q.fingerprint({k:v for k,v in new.items() if k!='sha256'})
    with pytest.raises(ValueError,match='Scientific'):Q.migration_draft(old,new,tmp_path,[proof])
