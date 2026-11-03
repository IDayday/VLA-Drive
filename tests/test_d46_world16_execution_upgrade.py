"""Four bounded CPU metadata/native-payload tests; never a GPU certificate."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_abcd_research.checkpoint import atomic_torch,CHECKPOINT_CONTRACT
from starVLA.rl.d44_native_rl.config import defaults46
from starVLA.rl.d46_world16 import native,runtime
from starVLA.rl.d46_world16.execution_upgrade import prepare,FLAG_NAMES


def write_sealed(path,value):
    value=copy.deepcopy(value);value.pop('sha256',None);value['sha256']=fingerprint(value);atomic_json(path,value);return value


def fixture(tmp_path,monkeypatch,arm='D0'):
    old_source=tmp_path/'frozen_old';old_source.mkdir();(old_source/'fixture.py').write_text('old immutable source\n')
    old_sources={'fixture.py':file_sha(old_source/'fixture.py')};new_sources={'fixture.py':'b'*64}
    monkeypatch.setattr(native,'source_inventory',lambda:new_sources);monkeypatch.setattr(runtime,'source_inventory',lambda:new_sources)
    root=tmp_path/'oldrun';cp=root/'checkpoints/update000004_inner0';cp.mkdir(parents=True)
    cfg=defaults46(arm)
    tr=dict(schema=native.SCHEMA,source_world=8,target_world=16,origin_updates=2,source_qualification=True,target_qualification=True,
        source_run_identity_sha256='8'*64,common_optimizer_sha256='9'*64,control_plane=copy.deepcopy(native.CONTROL_PROTOCOL),
        source_checkpoint='/qualification/original_world8_READY',virtual_sampler_lanes=8,same_algorithm=True)
    if arm=='D3_FPO':tr['flow_energy_protocol']=copy.deepcopy(native.FPO_PROTOCOL)
    identity=write_sealed(cp/'identity.json',dict(schema='d44_native_rl_v1',architecture_id='D',world=16,recipe=cfg,qualification=True,
        qualification_fixture='d46_tiny_native_world_transition_CPU_v1',parent_model_sha256='tiny_native_qualification:CPU_payload_fixture',
        scope=dict(trainable_names=['action_expert.weight'],trainable_numel=2),source_files=old_sources,fresh_Adam=False,native_world_transition=tr))
    atomic_json(root/'identity.json',identity)
    transition=dict(schema=native.SCHEMA,target_identity_sha256=identity['sha256'],source_identity_sha256='8'*64,
        source_inventory=old_sources,common_optimizer_sha256='9'*64,origin_updates=2,qualification=True,source_qualification=True,
        source_world=8,target_world=16,control_plane=copy.deepcopy(native.CONTROL_PROTOCOL))
    if arm=='D3_FPO':transition['flow_energy_protocol']=copy.deepcopy(native.FPO_PROTOCOL)
    transition=write_sealed(root/'WORLD_TRANSITION.json',transition)
    parameter=torch.nn.Parameter(torch.tensor([.01,.02]));optimizer=torch.optim.AdamW([parameter],lr=1e-6)
    for _ in range(4):optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
    common=dict(adapters={'action_expert.weight':parameter.detach().clone()},optimizer=optimizer.state_dict(),scheduler={'last_epoch':4})
    atomic_torch(cp/'optimizer.pt',common)
    rows=[];lanes=[torch.Generator().manual_seed(42+r).get_state()for r in range(8)]
    for rank in range(16):
        state=dict(updates=4,collections=2,scene_cursor=128,next_inner=0,phase='READY',raw_scene_exposures=128,valid_scene_exposures=128,
            padding_scene_exposures=0,new_candidates=2048,valid_PG_candidates=2048,extra_B1_queries=0,adaptive_G32_queries=0,
            adaptive_kl={'collections':2,'beta':.001},evaluation_gates=[],journal_origin_updates=2,
            world16_virtual_sampler=dict(schema=native.STATE_SCHEMA,lanes=copy.deepcopy(lanes),committed_collections=2),history={'keep':rank},clock={'immutable':True})
        rng=dict(python=random.Random(42+rank).getstate(),numpy=np.random.RandomState(42+rank).get_state(),
            torch=torch.Generator().manual_seed(700+rank).get_state(),cuda=None,rollout_generator=torch.Generator().manual_seed(900+rank).get_state())
        own=dict(state=state,rollout=None,rng=rng,identity_sha256=identity['sha256'])
        atomic_torch(cp/f'rank{rank}.pt',own);rows.append(native.World16Boundary.validate_boundary(state,None,identity,common['optimizer'],common['scheduler']))
    atomic_json(cp/'transaction.json',dict(schema=CHECKPOINT_CONTRACT,attempt='CPU-fixture',ranks=rows))
    reseal(cp)
    return root,cp,old_source,identity,transition


def reseal(cp):
    files={p.name:file_sha(p)for p in sorted(cp.iterdir())if p.is_file()and p.name not in('files.json','COMPLETE')}
    atomic_json(cp/'files.json',files);atomic_json(cp/'COMPLETE',dict(schema=2,contract=CHECKPOINT_CONTRACT,phase='READY',world=16,updates=4,next_inner=0,files_sha256=file_sha(cp/'files.json')))


def profile(**kw):return dict.fromkeys(FLAG_NAMES,False)|kw


def exact(a,b):
    from tools.d46_world16.qualify_native import exact as check
    check(a,b)


def test_new_root_READY_and_qualification_only_guards(tmp_path,monkeypatch):
    root,cp,source,identity,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError):prepare(cp,root,source,execution_flags=profile(),qualification=True)
    with pytest.raises(ValueError,match='independent'):prepare(cp,root/'child',source,execution_flags=profile(),qualification=True)
    with pytest.raises(ValueError,match='never enter formal'):prepare(cp,tmp_path/'formal',source,execution_flags=profile(),qualification=False)
    with pytest.raises(ValueError,match='five boolean'):prepare(cp,tmp_path/'badflags',source,execution_flags=profile(fast_prefix='true'),qualification=True)
    value=json.loads((cp/'COMPLETE').read_text());value.update(phase='INNER_COMMITTED',next_inner=1);atomic_json(cp/'COMPLETE',value)
    with pytest.raises(ValueError,match='READY'):prepare(cp,tmp_path/'pending',source,execution_flags=profile(),qualification=True)


def test_optimizer_bytes_all16_RNG_virtual8_state_and_new_journal_origin(tmp_path,monkeypatch):
    root,cp,source,identity,transition=fixture(tmp_path,monkeypatch,'D3_FPO');oldhash={p.name:file_sha(p)for p in cp.iterdir()if p.is_file()}
    result=prepare(cp,tmp_path/'upgraded',source,execution_flags=profile(fast_prefix=True,fuse_fm_mc=True),qualification=True)
    dest=Path(result['checkpoint']);new=json.loads((dest/'identity.json').read_text())
    assert file_sha(dest/'optimizer.pt')==oldhash['optimizer.pt']
    assert new['native_world_transition']==identity['native_world_transition']and new['fresh_Adam']is False
    for rank in range(16):
        before=torch.load(cp/f'rank{rank}.pt',weights_only=False);after=torch.load(dest/f'rank{rank}.pt',weights_only=False)
        assert after['identity_sha256']==new['sha256']
        before['identity_sha256']=new['sha256'];before['state']['journal_origin_updates']=4;exact(before,after)
    assert {p.name:file_sha(p)for p in cp.iterdir()if p.is_file()}==oldhash
    rt=runtime.receipt(tmp_path/'upgraded',new)
    assert rt['origin_updates']==2 and rt['execution_upgrade_sha256']==result['execution_upgrade_sha256']
    state=torch.load(dest/'rank0.pt',weights_only=False)['state']
    attempt=runtime.World16Attempt(tmp_path/'upgraded',0,state,new,dest)
    assert attempt.record['verified_native_origin_updates']==4
    assert not(tmp_path/'upgraded/metrics_rank0.jsonl').exists()
    bad=copy.deepcopy(state);bad['updates']=5
    with pytest.raises(ValueError,match='history gap'):runtime.World16Attempt(tmp_path/'upgraded',0,bad,new,dest)


def test_changed_payload_frozen_source_and_upgrade_receipt_refused(tmp_path,monkeypatch):
    root,cp,source,identity,_=fixture(tmp_path,monkeypatch)
    original=(source/'fixture.py').read_bytes();(source/'fixture.py').write_text('tampered')
    with pytest.raises(ValueError,match='Old frozen source'):prepare(cp,tmp_path/'badsource',source,execution_flags=profile(),qualification=True)
    (source/'fixture.py').write_bytes(original)
    own=torch.load(cp/'rank9.pt',weights_only=False);own['rng']['rollout_generator'][0]^=1;atomic_torch(cp/'rank9.pt',own)
    with pytest.raises(ValueError,match='source16 payload'):prepare(cp,tmp_path/'badpayload',source,execution_flags=profile(),qualification=True)
    # A fully resealed but internally inconsistent source transaction is refused.
    own['state']['scene_cursor']+=64;atomic_torch(cp/'rank9.pt',own);reseal(cp)
    with pytest.raises(ValueError,match='transaction|clock'):prepare(cp,tmp_path/'badtx',source,execution_flags=profile(),qualification=True)
    own['state']['scene_cursor']-=64;atomic_torch(cp/'rank9.pt',own);reseal(cp)
    result=prepare(cp,tmp_path/'good',source,execution_flags=profile(),qualification=True)
    new=json.loads((Path(result['checkpoint'])/'identity.json').read_text());p=tmp_path/'good/EXECUTION_UPGRADE.json';value=json.loads(p.read_text());value['execution_upgrade']['flags']['rotary_cache']=True;write_sealed(p,value)
    with pytest.raises(ValueError,match='receipt/profile'):runtime.receipt(tmp_path/'good',new)


def test_legacy_world16_without_upgrade_keeps_unchanged_origin_and_receipt(tmp_path,monkeypatch):
    root,cp,source,identity,transition=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runtime,'source_inventory',lambda:identity['source_files'])
    assert runtime.receipt(root,identity)==transition
    own=torch.load(cp/'rank0.pt',weights_only=False);common=torch.load(cp/'optimizer.pt',weights_only=False)
    native.World16Boundary.validate_boundary(own['state'],None,identity,common['optimizer'],common['scheduler'])
    # Legacy histories still start at the original8->16 origin (2), not cp U4.
    rows=[dict(updates=u,run=identity['sha256'])for u in(3,4)]
    (root/'metrics_rank0.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in rows))
    attempt=runtime.World16Attempt(root,0,own['state'],identity,cp)
    assert attempt.record['verified_native_origin_updates']==2
