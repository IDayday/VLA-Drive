"""Behavioral tests for scheduling, full optimizer state and consumed-only resume."""
from copy import deepcopy
from types import SimpleNamespace, ModuleType
import sys
import random
import time
import numpy as np
import pytest
import torch
from torch import nn
from iqe.training.execution import EpochPlan, OrderedPrefetch, finite_gradients, collective_check, FrozenVisionAhead
from iqe.training.runtime_identity import resume_dependencies, validate_runtime
from iqe.training.checkpoint import capture_rng
from iqe.io import read_json, atomic_json


def batches(size, batch, seed, epoch):
    ids = np.random.RandomState(seed + epoch).permutation(size).tolist()
    return [ids[i:i+batch] for i in range(0,size,batch)]


def test_epoch_plan_reuses_shuffle_and_matches_tail_all_ranks():
    calls = []
    def build(*args):
        calls.append(args); return batches(*args)
    addresses = []
    for rank in range(4):
        plan = EpochPlan(10,8,5,rank,4,build)
        a = list(plan.iter_from(0,0,4))
        assert a[1].valid_count == (1 if rank<2 else 0)
        assert len(a[1].mine) == 1
        assert [(x.epoch,x.offset) for x in a] == [(0,0),(0,1),(1,0),(1,1)]
        addresses.append(a)
    assert len(calls) == 8  # once per epoch/rank, not once per update
    for step in range(4):
        valid = [x[step].mine[:x[step].valid_count] for x in addresses]
        assert sorted(i for row in valid for i in row) == sorted(addresses[0][step].entries)


def test_prefetch_resume_discards_generated_unconsumed_work_and_preserves_rng():
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    before = capture_rng()
    plan = EpochPlan(17,4,7,0,1,batches)
    produced = []
    def prepare(a):
        produced.append(a)
        return torch.tensor(a.mine)
    with OrderedPrefetch(plan.iter_from(0,0,8), prepare,2) as q:
        a, value = q.pop()
        b, second = q.pop()
        q.queue[0][1].result()  # force one prepared-but-unconsumed batch before interruption
    assert len(produced) > 2
    after = capture_rng()
    assert before['python']==after['python'] and torch.equal(before['cpu'],after['cpu'])
    assert np.array_equal(before['numpy'][1],after['numpy'][1])
    with OrderedPrefetch(EpochPlan(17,4,7,0,1,batches).iter_from(b.epoch,b.offset+1,6),prepare,1) as q:
        resumed,_ = q.pop()
    expected = list(EpochPlan(17,4,7,0,1,batches).iter_from(0,0,3))[-1]
    assert resumed == expected


def test_prefetch_failure_is_not_silently_skipped():
    def prepare(i):
        if i == 1: raise OSError('missing target cache')
        return i
    with OrderedPrefetch(range(4),prepare,2) as q:
        assert q.pop() == (0,0)
        with pytest.raises(OSError,match='missing target'): q.pop()
    with pytest.raises(ValueError,match='failure'): collective_check('cache unavailable')


def test_batched_finite_gradients_preserves_elementwise_rule():
    a,b = nn.Parameter(torch.zeros(2)),nn.Parameter(torch.zeros(2))
    assert finite_gradients([a,b])
    a.grad=torch.tensor([1e30,1e30])
    assert finite_gradients([a,b])  # a norm can overflow although all elements are finite
    b.grad=torch.tensor([0.,float('nan')]); assert not finite_gradients([a,b])
    b.grad=torch.tensor([float('inf'),0.]); assert not finite_gradients([a,b])


def test_legacy_unused_validation_is_explicit_and_no_other_change_is_allowed():
    old={'config':{'validation_every_steps':200,'lr':1e-5},'data':'same'}
    new={'config':{'validation_every_steps':0,'lr':1e-5},'data':'same'}
    with pytest.raises(ValueError,match='changed'): resume_dependencies(old,new)
    assert resume_dependencies(old,new,allow_legacy_validation_metadata=True)['kind']=='LEGACY_UNUSED_VALIDATION_FIELD'
    assert old['config']['validation_every_steps']==200
    for modified in ({**new,'data':'other'}, {'config':{**new['config'],'lr':2e-5},'data':'same'}):
        with pytest.raises(ValueError,match='changed'):
            resume_dependencies(old,modified,allow_legacy_validation_metadata=True)


def test_unqualified_runtime_cannot_start_full():
    runtime={'prefetch_depth':1,'frozen_vision_ahead':True,'qualification':None}
    with pytest.raises(ValueError,match='UNQUALIFIED'):
        validate_runtime(runtime,{'framework_contract_hash':'test'},8,mode='full')


def test_vision_handoff_one_use_prompt_guard_backward_and_frozen_mode():
    visual=nn.Linear(2,2).eval().requires_grad_(False)
    class Model:
        qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(visual=visual)))
        def _build_qwen_batch(self,obs,instructions):
            return visual(torch.stack([x['image'][0] for x in obs]))
    model=Model(); original=model._build_qwen_batch
    engine=FrozenVisionAhead(model,concurrent=False)
    engine.instructions=lambda obs:[o['lang'] for o in obs]
    obs=[{'image':[torch.tensor([1.,2.])],'lang':'go','state':np.ones((1,4))}]
    target=original(obs,['go'])
    prepared=engine.prepare(obs,1)
    head=nn.Linear(2,1)
    with engine.activate(prepared):
        actual=model._build_qwen_batch(obs,['go'])
        assert torch.equal(actual,target) and not actual.is_inference()
        head(actual).sum().backward()
    assert head.weight.grad is not None and visual.weight.grad is None
    with pytest.raises(ValueError,match='missing staged'): model._build_qwen_batch(obs,['go'])
    with engine.activate(engine.prepare(obs,1)):
        with pytest.raises(ValueError,match='mismatch'): model._build_qwen_batch(obs,['changed'])
    visual.train()
    with pytest.raises(ValueError,match='train'): engine.prepare(obs,1)
    engine.close()


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2,4),nn.Dropout(.25),nn.Linear(4,1))
    def forward(self,observations,targets,**kwargs):
        x=torch.stack([o['x'] for o in observations]); y=self.net(x)
        loss=(y-targets['ego']).square().mean()
        return {'loss':loss,'losses':{'ego_il':loss}}


def test_actual_base_trainer_prefetch_checkpoint_resume_exact(tmp_path,monkeypatch):
    from iqe.config import load_config
    from iqe.training import base_trainer as bt
    config=load_config('configs/iqe/main.yaml')
    config['query_base'].update(max_optimizer_steps=6,warmup_steps=1,global_batch_size=4,micro_batch_size=2)
    config['execution']['checkpoint_every_steps']=2
    config['execution']['loader_workers']=2
    class Dataset:
        index=[{'token':str(i)} for i in range(11)]
        def __getitem__(self,i):
            return {'x':torch.tensor([float(i)/11,1.])},{'ego':torch.tensor([float(i)/22])}
    monkeypatch.setattr(bt,'build_query_framework',lambda *a:TinyBase())
    monkeypatch.setattr(bt,'original_dataset',lambda *a,**kw:Dataset())
    def install(name,**values):
        m=ModuleType(name); m.__dict__.update(values);monkeypatch.setitem(sys.modules,name,m)
    install('tools.ddpolicy_vehicle.training_state',epoch_batches=batches)
    install('tools.foresight.student_state',optimizer_batch_counts=lambda *a:{})
    install('starVLA.dataloader.foresight_dataset',collate_training=lambda samples:
        ([s[0] for s in samples],{'ego':torch.stack([s[1]['ego'] for s in samples])}))
    contract={'framework_contract_hash':'unit','source_config':{'trainer':{'optimizer':{'betas':[.9,.95]}}},
              'query_architecture':{},'source_root':'unit','source_commit':'unit'}
    scenes=[SimpleNamespace(scene_id=str(i),split_role='incremental_fit',source_kind='original',target_provenance='gt') for i in range(11)]
    def run(folder,**kw):
        return bt.train_base(config,contract,scenes,tmp_path/folder,mode='smoke',device='cpu',**kw)
    baseline=run('serial')
    config['base_runtime']['prefetch_depth']=2
    whole=run('prefetch')
    paused=run('resume',stop_after=2)
    resumed=run('resume',resume=paused)
    states=[torch.load(p,weights_only=False) for p in [baseline,whole,resumed]]
    for a in states[1:]:
        assert a['epoch']==states[0]['epoch'] and a['offset']==states[0]['offset']
        for k,v in a['model'].items(): assert torch.equal(v,states[0]['model'][k])
        for k,v in a['optimizer']['state'].items():
            for key,value in v.items(): assert torch.equal(value,states[0]['optimizer']['state'][k][key])
        assert torch.equal(a['rng_by_rank'][0]['cpu'],states[0]['rng_by_rank'][0]['cpu'])
    import json
    rows=lambda folder:[json.loads(s) for s in (tmp_path/folder/'steps.jsonl').read_text().splitlines()]
    for a,b in zip(rows('serial'),rows('resume')):
        assert a['sample_ids']==b['sample_ids'] and a['losses']==b['losses'] and a['gradient_norm']==b['gradient_norm']
    assert read_json(tmp_path/'resume/RESUME_RESOLUTION.json')['next_sample_ids']==rows('serial')[2]['sample_ids']


def test_two_producer_stages_overlap_without_reordering():
    from threading import Event
    from iqe.training.execution import TrainingPrefetch
    cpu_second = Event()
    def load(i):
        if i == 1: cpu_second.set()
        return i
    def frozen(value):
        if value == 0: assert cpu_second.wait(2), 'CPU producer serialized behind frozen GPU stage'
        return value * 2
    with TrainingPrefetch(range(5),5,load,frozen,1) as pipeline:
        assert [pipeline.pop() for _ in range(5)] == [(i,2*i) for i in range(5)]


def test_local_checkpoint_staging_checksum_and_original_preservation(tmp_path):
    from iqe.io import file_hash
    from iqe.training.local_checkpoint import stage_checkpoint
    source=tmp_path/'source.pt'; source.write_bytes(b'original checkpoint test fixture')
    checksum=file_hash(source)
    staged=stage_checkpoint(source,checksum,tmp_path/'local')
    assert file_hash(staged)==file_hash(source)==checksum
    assert stage_checkpoint(source,checksum,tmp_path/'local')==staged
    staged.write_bytes(b'broken')
    with pytest.raises(ValueError,match='corrupt local'): stage_checkpoint(source,checksum,tmp_path/'local')
    assert file_hash(source)==checksum


def test_efficiency_uses_recent_updates_not_cumulative_average(tmp_path):
    import json
    from iqe.training.efficiency import training_efficiency
    rows=[{'optimizer_step':i+1,'compute_update_seconds':1. if i<10 else 4.,
           'seconds_per_update':1.,'actual_global_batch':32,'losses':{'il':.5},'gradient_norm':1.} for i in range(15)]
    (tmp_path/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"partial":')
    result=training_efficiency(tmp_path,1,2.,window=4)
    assert result['p50_update_seconds']==4. and not result['throughput_gate_passed']
    assert result['window_start_step']==12 and result['samples_per_compute_second']==8.
    for r in rows: del r['compute_update_seconds']
    (tmp_path/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='cumulative'): training_efficiency(tmp_path,1,2.,window=4)


def _fast_collective_worker(rank, rendezvous, output):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    collective_check(None)
    try:
        collective_check('rank1 missing cache' if rank==1 else None)
    except ValueError as error:
        atomic_json(str(output)+str(rank)+'.json', {'message':str(error)})
    else:
        raise AssertionError('all ranks must fail together')
    dist.destroy_process_group()


def test_fast_collective_propagates_single_rank_failure(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_fast_collective_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'rank')),nprocs=2,join=True)
    assert 'rank1 missing cache' in read_json(tmp_path/'rank0.json')['message']
    assert read_json(tmp_path/'rank0.json')==read_json(tmp_path/'rank1.json')


def test_deterministic_backend_migration_is_explicit_and_narrow():
    from iqe.training.runtime_identity import resume_dependencies
    before={'config':{'lr':.1,'validation_every_steps':200},'environment':{'torch':'unit','deterministic':False},'data':'fixed'}
    after={'config':{'lr':.1,'validation_every_steps':0},'environment':{'torch':'unit','deterministic':True},'data':'fixed'}
    with pytest.raises(ValueError):resume_dependencies(before,after,allow_legacy_validation_metadata=True)
    result=resume_dependencies(before,after,allow_legacy_validation_metadata=True,allow_deterministic_backend_migration=True)
    assert result['optimizer_state_preserved'] and not result['bitwise_legacy_continuation']
    changed=deepcopy(after);changed['config']['lr']=.2
    with pytest.raises(ValueError):resume_dependencies(before,changed,allow_legacy_validation_metadata=True,allow_deterministic_backend_migration=True)
