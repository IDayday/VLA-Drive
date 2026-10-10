from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import os
import random
import subprocess
import sys
import numpy as np
import pytest
import torch
from iqe.io import atomic_json, atomic_torch, read_json, file_hash, digest
from iqe.training.resume import resolve_resume, inspect_checkpoint
from iqe.training.randomness import initialization_rng, module_seed
from iqe.evaluation.retention import module_hash
from iqe.ablations.residual_v2 import ResidualExpertV2
from iqe.data.support_retrieval import retrieve, RetrievalIndex
from iqe.data.manifests import save_scenes
from iqe.config import load_config
from iqe.rounds import run_round
from conftest import TensorAdapter, scene
from test_two_round_pipeline import pipeline_fixture
from test_training_resume_ddp import network, train, cfg, loss_fn, fetch, denominator, plan, ConsumedSampler, capture_rng


def checkpoint(path, step, *, base=False, config=None, dependencies=None):
    dependencies = dependencies or {'data':'test', 'config':config or {}}
    value = {'optimizer_step':step,'dependencies':dependencies,'resume_boundary':'optimizer_step',
             'optimizer':{},'scheduler':{},'rng_by_rank':[]}
    if base:
        value.update(kind='iqe_query_base',model={},epoch=0,offset=step)
    else:
        value.update(schema_version=1,module={},sampler={},resolved_config=config or {})
    atomic_torch(path,value)
    atomic_json(str(path)+'.COMPLETE.json',{'checksum':file_hash(path),'optimizer_step':step,'dependencies':dependencies})


@pytest.mark.parametrize('base',[False,True])
def test_resume_newest_beats_stale_result_and_repeat_interrupt(tmp_path,base):
    a,b,c=[tmp_path/f'step_{s:06d}.pt' for s in (1000,2000,3000)]
    checkpoint(a,1000,base=base);checkpoint(b,2000,base=base)
    atomic_json(tmp_path/'result.json',{'checkpoint':str(a),'optimizer_steps':1000,'status':'STOPPED_AT_BOUNDARY'})
    assert resolve_resume(tmp_path,'auto')==str(b)
    assert read_json(tmp_path/'RESUME_RESOLUTION.json')['result_json_stale']
    checkpoint(c,3000,base=base)
    assert resolve_resume(tmp_path,'auto')==str(c)


def test_corrupt_partial_fallback_requires_policy_and_explicit_old_requires_fork(tmp_path):
    a=tmp_path/'step_000001.pt';b=tmp_path/'step_000002.pt'
    checkpoint(a,1);b.write_bytes(b'partial')
    with pytest.raises(ValueError,match='FALLBACK_REQUIRED'):resolve_resume(tmp_path,'auto')
    assert resolve_resume(tmp_path,'auto',allow_fallback=True)==str(a)
    with pytest.raises(ValueError,match='FORK_RUN_REQUIRED'):resolve_resume(tmp_path,a)
    fork=tmp_path/'fork'
    assert resolve_resume(fork,a)==str(a)
    assert read_json(fork/'RESUME_LINEAGE.json')['fork_run']
    assert b.read_bytes()==b'partial'
    checkpoint(b,2);b.write_bytes(b'corrupt')
    assert resolve_resume(tmp_path,'auto',allow_fallback=True)==str(a)
    assert any('checksum' in x.get('reason','') for x in read_json(tmp_path/'RESUME_RESOLUTION.json')['candidates'])


def test_resume_rejects_metadata_identity_and_changed_config(tmp_path):
    p=tmp_path/'step_000001.pt';checkpoint(p,1,config={'lr':.01})
    with pytest.raises(ValueError,match='config'):resolve_resume(tmp_path,'auto',expected_config={'lr':.02})
    r=read_json(str(p)+'.COMPLETE.json');r['optimizer_step']=2;atomic_json(str(p)+'.COMPLETE.json',r)
    with pytest.raises(ValueError,match='optimizer_step'):resolve_resume(tmp_path,'auto')


def test_two_interrupts_preserve_next_ids_rng_gradients_and_parameters(tmp_path):
    torch.manual_seed(6);full=network(.2);part=deepcopy(full)
    full_rows=train(full,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{'data':'test'},tmp_path/'full',seed=39,denominator_function=denominator)
    full_rng=capture_rng()
    train(part,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{'data':'test'},tmp_path/'part',seed=39,denominator_function=denominator,stop_after=1)
    stale=read_json(tmp_path/'part/result.json')
    train(part,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{'data':'test'},tmp_path/'part',seed=39,denominator_function=denominator,resume='auto',stop_after=2)
    atomic_json(tmp_path/'part/result.json',stale) # emulate crash after checkpoint, before result update
    tail=train(part,loss_fn,fetch,ConsumedSampler(plan()),cfg(),{'data':'test'},tmp_path/'part',seed=39,denominator_function=denominator,resume='auto')
    assert tail[0]['sample_ids']==full_rows[2]['sample_ids']
    assert tail[0]['loss']==full_rows[2]['loss']
    for a,b in zip(full.parameters(),part.parameters()):
        assert torch.equal(a,b) and torch.equal(a.grad,b.grad)
    assert torch.equal(full_rng['cpu'],capture_rng()['cpu'])
    assert read_json(tmp_path/'part/RESUME_RESOLUTION.json')['next_sample_ids']==tail[0]['sample_ids']


def test_initialization_restores_rng_and_is_process_independent(tmp_path):
    script='''from iqe.training.randomness import initialization_rng, module_seed
from iqe.scorer import TrajectoryScorer
from iqe.evaluation.retention import module_hash
import sys,torch
_ = torch.randn(int(sys.argv[2]))
with initialization_rng(module_seed(int(sys.argv[1]),'scorer',2)):
 m=TrajectoryScorer(8,8,('nc',),16,1,2,0.)
print(module_hash(m))
'''
    hashes=[subprocess.check_output([sys.executable,'-c',script,str(seed),str(draws)],text=True).strip() for seed,draws in [(3,0),(3,91),(4,0)]]
    assert hashes[0]==hashes[1] and hashes[0]!=hashes[2]
    before=capture_rng()
    with initialization_rng(33):
        random.random();np.random.rand();torch.randn(20)
    assert torch.equal(before['cpu'],capture_rng()['cpu']) and before['python']==capture_rng()['python']


def test_selector_constructor_and_router_expansion_identity(tmp_path,monkeypatch):
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    run_round(pipeline,1,steps=2)
    identity=read_json(pipeline.round_root(1)/'scorer/identity.json')
    torch.randn(99)
    fresh=pipeline.selector_module(1)
    assert module_hash(fresh)==identity['initialization']['initial_state_hash']
    run_round(pipeline,2,steps=2,selector_only=True)
    second=read_json(pipeline.round_root(2)/'scorer/identity.json')
    assert second['initialization']['parent_checkpoint_hash']==file_hash(pipeline.round_root(1)/'scorer/best.pt')
    from iqe.router import SceneRouter
    with initialization_rng(7):old=SceneRouter(8,8,['expert_0'],16)
    a=old.expanded(['expert_0','expert_1'],init_seed=19)
    torch.randn(34);b=old.expanded(['expert_0','expert_1'],init_seed=19)
    assert module_hash(a)==module_hash(b)
    assert torch.equal(a.head.weight[0],old.head.weight[0])


def test_true_residual_has_zero_delta_changes_and_depends_on_base(features):
    source=TensorAdapter().action.eval().requires_grad_(False);expert=ResidualExpertV2(source).eval()
    initial=module_hash(expert.initial)
    assert torch.count_nonzero(expert.delta(features).raw)==0
    assert torch.equal(expert(features).raw,source(features).raw)
    optimizer=torch.optim.SGD(expert.delta.parameters(),lr=.01)
    target=torch.randn_like(expert(features).raw)
    out=expert(features);loss=(out.raw-target).square().mean();loss.backward();optimizer.step()
    assert expert.delta(features).raw.abs().max()>0
    assert module_hash(expert.initial)==initial
    out1=expert(features)
    with torch.no_grad():expert.initial.heads[-1].bias.add_(.25)
    out2=expert(features)
    torch.testing.assert_close(out2.raw-out1.raw,torch.full_like(out1.raw,.25))
    torch.testing.assert_close(out2.physical,expert.trajectory_contract.physical(out2.raw))
    # Output supervision is exactly equivalent to residual target in raw coordinates.
    with torch.no_grad():base=expert.initial(features).raw
    torch.testing.assert_close((expert(features).raw-target).square().mean(),(expert.delta(features).raw-(target-base)).square().mean())


def test_retrieval_index_matches_reference_with_ties_and_exclusions():
    rows=[scene(i) for i in range(30)];vectors=np.random.default_rng(9).normal(size=(30,12));vectors[2]=vectors[3]
    index=RetrievalIndex(rows,vectors,'version',7)
    for i in (0,2,13):
        a,ar=retrieve(rows[i],vectors[i],rows,vectors,5,'version',7)
        b,br=index.query(rows[i],vectors[i],5)
        assert a==b and ar['candidates']==br['candidates']
    assert list(index.query_many(rows[:3],vectors[:3],5))==[index.query(s,v,5) for s,v in zip(rows[:3],vectors[:3])]


def test_nondefault_bootstrap_and_repetitions_rejected(tmp_path):
    from iqe.evaluation.oracle import oracle_report
    from iqe.data.mining import MiningThresholds
    c=load_config('configs/iqe/main.yaml')
    c['metric']['repetitions']=2
    import yaml
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='repetitions=1'):load_config(path)
    c['metric']['repetitions']=1;c['query_base']['validation_every_steps']=200;path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='query_base'):load_config(path)
    scores=np.array([[.1,.9],[.8,.9],[.7,.8]])
    comp={c:np.ones_like(scores) for c in ['no_at_fault_collisions','drivable_area_compliance']}
    valid={c:np.ones_like(scores,bool) for c in comp}
    gates={'minimum_logs_for_ci_gate':4,'oracle_min_mean_gain':.005,'oracle_min_new_successes':1}
    r=oracle_report(scores,np.ones_like(scores,bool),comp,valid,['expert_0','expert_1'],['a','b','c'],'expert_1',MiningThresholds(),gates,repetitions=17,seed=12)
    assert r['bootstrap']['seed']==12 and r['bootstrap']['repetitions']==17 and r['science']=='INCONCLUSIVE'


def test_storage_shortage_blocks_without_deletion(tmp_path):
    from iqe.storage import capacity_plan, require_capacity
    from iqe.io import BlockedError
    old=tmp_path/'existing.pt';old.write_bytes(b'protected')
    report=capacity_plan(tmp_path,100,10,100,20,50,free_bytes=100)
    assert report['remaining_saves']==8 and report['retention_dry_run']
    with pytest.raises(BlockedError,match='STORAGE_CAPACITY'):require_capacity(report)
    assert old.read_bytes()==b'protected'


def test_frozen_is_evaluation_candidate_not_accepted_pool(tmp_path):
    from iqe.registry import ExpertRegistry
    from test_cli_registry_export import event
    r=ExpertRegistry(tmp_path/'registry.json','contract')
    r.append(event());r.append(event(1,'training','training'));r.append(event(1,'frozen'))
    assert set(r.candidate_pool())=={'expert_0'}
    assert set(r.candidate_pool(include_frozen=True))=={'expert_0','expert_1'}
    r.append(event(1,'candidate_only'))
    assert set(r.candidate_pool())=={'expert_0','expert_1'}


def test_explicit_old_checkpoint_fork_restores_historical_best(tmp_path):
    module=network();settings=cfg();settings['validation_every_steps']=1
    train(module,loss_fn,fetch,ConsumedSampler(plan()),settings,{'data':'test'},tmp_path/'parent',seed=9,
          denominator_function=denominator,validate=lambda m,s:{'selection_key':[-s]})
    old=tmp_path/'parent/step_000001.pt'
    parent_hash=file_hash(tmp_path/'parent/step_000004.pt')
    rows=train(network(),loss_fn,fetch,ConsumedSampler(plan()),settings,{'data':'test'},tmp_path/'fork',seed=9,
        denominator_function=denominator,validate=lambda m,s:{'selection_key':[-s]},resume=old)
    assert rows[0]['optimizer_step']==2
    assert file_hash(tmp_path/'parent/step_000004.pt')==parent_hash
    assert read_json(tmp_path/'fork/result.json')['best_selection_key']==[-1]


def test_nondefault_no_replay_changes_actual_training_mask(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);p.config['scorer']['replay_old_candidates']=False
    run_round(p,1,steps=2)
    identity=read_json(p.round_root(1)/'scorer/identity.json')
    row=json.loads((p.round_root(1)/'scorer/steps.jsonl').read_text().splitlines()[0])
    assert not identity['old_new_replay']
    assert row['losses']['value']['denominator']==4 # one candidate per scene
    assert row['losses']['ranking']['denominator']==0


def test_router_rejected_rows_removed_without_renumbering():
    from iqe.router import SceneRouter
    old=SceneRouter(8,8,['expert_0','expert_1','expert_3'],16)
    new=old.for_pool(['expert_0','expert_1','expert_4'],init_seed=33)
    assert new.expert_ids==('expert_0','expert_1','expert_4')
    assert torch.equal(new.head.weight[:2],old.head.weight[:2])
