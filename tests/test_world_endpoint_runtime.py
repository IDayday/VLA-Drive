"""Endpoint collection/real-Scope loss and true8rank native recovery checks."""
import copy
import json
from datetime import timedelta
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_world_safety_exploration_groups import cfg as base_cfg
from world_postcot.test_scope_rl import make
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY,SCOPE_SCHEMA
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime
from starVLA.rl.qwen_world_safety_exploration import endpoint_runtime as E
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit
from starVLA.rl.qwen_world_safety_exploration.runtime import runtime_for_config
from starVLA.rl.qwen_world_safety_exploration.recovery import tensor_digest
from starVLA.rl.qwen_world_safety_exploration.endpoint_policy import endpoint_log_prob
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for


def cfg():
    c=base_cfg(8);c['research']['endpoint_policy']=copy.deepcopy(E.ENDPOINT_CONFIG)
    return c


def factory():
    actor,conditions,_,_=make(MEMORY)
    actor._loss=E.EndpointLoss(cfg())
    return actor,conditions


class CanonicalCPU:
    """Algebraic scoring mock only, never passed off as a driving evaluation."""
    protocol=dict(sha256='CPU_unit_canonical')
    def score(self,job):
        token,log,requested=job
        assert requested.shape==(8,8,3) and requested.dtype==np.float32
        result=[]
        for j,request in enumerate(requested):
            nc=dac=comfort=1.;ttc=0. if j==7 else 1.;ep=.78 if j==0 else (1. if j<7 else .9)
            score=nc*dac*(5*ep+5*ttc+2*comfort)/12
            result.append(dict(status='OK',token=token,log=log,protocol_sha256=self.protocol['sha256'],cache=dict(sha256='unit_context'),
                requested_trajectory=request.copy(),metrics=dict(no_at_fault_collisions=nc,drivable_area_compliance=dac,
                    time_to_collision_within_bound=ttc,ego_progress=ep,comfort=comfort,score=score,driving_direction_compliance=float(j!=1)),
                readouts=dict(canonical_PDMS=score),scoring_seconds=.0))
        return result


def collect(actor,conditions,rank=0):
    generator=torch.Generator(device='cpu').manual_seed(seed_for('simwam_source_rank_rollout',42,rank))
    events=[]
    r=E.collect_endpoint(actor,conditions.get('train',actor.device),'train','unit_log',CanonicalCPU(),generator,
        variant='R_CANONICAL',version=dict(run='unit_endpoint',update=0,adapter_sha256='same'),progress=lambda *a:events.append(a))
    GroupCredit(8).finalize(r)
    return r,generator,events


def identity(conditions):
    return dict(world=8,sha256='unit_endpoint',schema=SCOPE_SCHEMA,recipe=cfg(),
        projected_condition_identity=conditions.sha,score_protocol=CanonicalCPU.protocol)


def state(u=0):
    return dict(updates=u,collections=1,next_inner=u,phase='ROLLOUT_COMMITTED' if u==0 else 'INNER_COMMITTED',
        scene_cursor=8,raw_scene_exposures=8,new_candidates=64)


def test_real_scope_endpoint_fresh_ratio_gradient_and_true_correlated_density(tmp_path):
    torch.set_num_threads(2)
    actor,conditions=factory();r,generator,events=collect(actor,conditions)
    assert [(v[0],v[2]) for v in events]==[('generated',8),('scored',8)]
    p=r['endpoint_policy']
    assert r['stoch_idx'].numel()==0 and 'old_probability' not in r
    assert r['old_logp'].shape==(8,1) and r['old_logp'].dtype==torch.float64
    assert not torch.equal(p['raw_innovations'],p['correlated_noise'])
    assert r['chain'][:,0].unique(dim=0).shape[0]==8
    direct=endpoint_log_prob(p['action'].double(),p['old_mean'],p['sigma'],.8)[:,None]
    assert torch.equal(direct,r['old_logp'])
    assert torch.equal(p['action'],p['unrounded_action'].float())
    assert all(np.array_equal(row['requested_trajectory'],p['action'][j].numpy()) for j,row in enumerate(r['records']))
    builds=actor.current_memory_builds;frozen={n:v.clone() for n,v in actor.state_dict().items() if n not in actor.closure['trainable_names']}
    loss,metrics=actor(r)
    assert actor.current_memory_builds==builds+2 and actor._prepared_scope is None
    ratio=E.EndpointRuntimeMixin.policy_ratio(actor,r)
    torch.testing.assert_close(ratio,torch.ones_like(ratio),atol=1e-10,rtol=0)
    assert actor._probability_audit['current']['source_clamp_hit_rate'] is None
    assert metrics['endpoint_actual_scored_candidates']==8 and metrics['endpoint_unscored_clean_means']==8
    assert metrics['quality_strict_high_count']==7 and metrics['quality_direction_high_count']==6
    assert metrics['quality_positive_high_count']==6 and metrics['quality_negative_high_count']==1
    assert metrics['quality_high_positive_PPO_blocked']==0 and metrics['quality_high_weight_per_good']>0
    assert metrics['quality_active_negative_unsafe_weight']>0
    loss.backward()
    trainable=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in trainable)
    for prefix in ('action_expert.','scene_provider.','predictive_provider.'):
        assert sum(float(p.grad.abs().sum()) for n,p in trainable if n.startswith(prefix))>0
    for n,v in actor.state_dict().items():
        if n in frozen:assert torch.equal(v,frozen[n])
    assert r['chain'].grad is None and p['terminal_x'].grad is None and p['action'].grad is None
    import yaml
    file=tmp_path/'endpoint.yaml';file.write_text(yaml.safe_dump(cfg()))
    runtime=runtime_for_config(file);assert isinstance(runtime,E.EndpointRuntimeMixin)
    for key,value in [('rho',.9),('physical_std',[1.,1.,.1])]:
        wrong=cfg();wrong['research']['endpoint_policy'][key]=value
        with pytest.raises(ValueError):E.validate_endpoint_recipe(wrong)


def _native(rank,initfile,root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+initfile,rank=rank,world_size=8,timeout=timedelta(seconds=180))
    try:
        actor,conditions=factory();r,generator,events=collect(actor,conditions,rank)
        runtime=E.EndpointRuntimeMixin();runtime.credit=GroupCredit(8)
        ident=identity(conditions)
        kwargs=dict(boundary_validator=runtime.validate_boundary,rank_validator=runtime.validate_ranks)
        optimizer=ScopeRuntime().make_optimizer(actor,cfg())
        actor._loss.optimizer=optimizer;actor._loss.learning_rate=5e-5
        scheduler=actor._loss._build_scheduler('constant',50000,100)
        digest=tensor_digest({k:v for k,v in r.items() if k!='records'})
        runtime.validate_boundary(state(),r,ident)
        for field in ('old_logp','action','raw_innovations'):
            bad=copy.deepcopy(r)
            value=bad['old_logp'] if field=='old_logp' else bad['endpoint_policy'][field]
            value.reshape(-1)[0]+=.01
            with pytest.raises(ValueError):runtime.validate_boundary(state(),bad,ident)
        observations=[]
        with patch.object(torch.cuda,'is_available',lambda:False):
            for u in range(4):
                before={n:p.detach().clone() for n,p in actor.named_parameters() if p.requires_grad}
                adam=tensor_digest(optimizer.state_dict());schedule=tensor_digest(scheduler.state_dict())
                rng=generator.get_state().clone();torch_rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
                path=checkpoint.save(root,actor,optimizer,scheduler,generator,state(u),r,ident,**kwargs)
                # A genuinely fresh actor/Memory/reference graph and native Adam,
                # not merely perturbing weights on an already prepared actor.
                actor,conditions=factory();optimizer=ScopeRuntime().make_optimizer(actor,cfg())
                actor._loss.optimizer=optimizer;actor._loss.learning_rate=5e-5
                scheduler=actor._loss._build_scheduler('constant',50000,100)
                restored,r=checkpoint.restore(path,actor,optimizer,scheduler,generator,ident,**kwargs)
                assert restored==state(u) and torch.equal(generator.get_state(),rng) and torch.equal(torch.get_rng_state(),torch_rng)
                assert random.getstate()==py_rng and np.array_equal(np.random.get_state()[1],np_rng[1])
                assert all(torch.equal(p,before[n]) for n,p in actor.named_parameters() if p.requires_grad)
                assert tensor_digest(optimizer.state_dict())==adam and tensor_digest(scheduler.state_dict())==schedule
                assert tensor_digest({k:v for k,v in r.items() if k!='records'})==digest
                total,metrics=actor(r)
                ratio=runtime.policy_ratio(actor,r)
                if u==0:torch.testing.assert_close(ratio,torch.ones_like(ratio),atol=1e-10,rtol=0)
                total.backward()
                params=[p for p in actor.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                observations.append(dict(inner=u,loss=float(total.detach()),policy_loss=float(metrics['policy_loss']),
                    weighted_anchor_loss=float(metrics['kl_loss']),joint_ratio_min=float(ratio.min()),joint_ratio_max=float(ratio.max()),
                    local_gradient_norm=float(torch.stack([p.grad.norm() for p in params]).norm())))
                for p in params:dist.all_reduce(p.grad);p.grad.div_(8)
                torch.nn.utils.clip_grad_norm_(params,1.)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            ready=state();ready.update(updates=4,next_inner=0,phase='READY')
            path=checkpoint.save(root,actor,optimizer,scheduler,generator,ready,None,ident,**kwargs)
            loaded,none=checkpoint.restore(path,actor,optimizer,scheduler,generator,ident,**kwargs)
            assert loaded==ready and none is None
        (Path(root)/f'CPU_rank{rank}.json').write_text(json.dumps(dict(rank=rank,observations=observations,
            restored_boundaries=['pending0','pending1','pending2','pending3','READY4'],
            corruption_rejected=['old_logp','action','raw_innovations'],scoring='algebraic CPU mock; no driving evaluation'),indent=2))
        if rank==0:
            assert len(list(Path(path).glob('rank*.pt')))==8
            (Path(root)/'NATIVE_CPU_PASS').write_text('actual Scope actor endpoint: world8, pending0/1/2/3 READY4, fresh actor native Adam/RNG/material restore\n')
    finally:dist.destroy_process_group()


def test_actual_scope_eight_rank_native_pending_and_fresh_actor_restore(tmp_path):
    torch.set_num_threads(1)
    mp.start_processes(_native,args=(str(tmp_path/'gloo'),str(tmp_path/'native')),nprocs=8,join=True,start_method='fork')
    assert (tmp_path/'native/NATIVE_CPU_PASS').is_file()
