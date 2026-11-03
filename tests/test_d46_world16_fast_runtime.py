"""Small CPU entry/profile integration checks, not CUDA/world16 qualification."""
import ast
import copy
from pathlib import Path
import pytest
import torch
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.d46_world16.fast_runtime import FastWorld16Extension
from starVLA.rl.d46_world16.execution_upgrade import FLAG_NAMES,SCHEMA
from starVLA.rl.d46_world16 import runtime


def saved(profile):
    sources={'qualification_source.py':'a'*64}
    meta=dict(schema=SCHEMA,source_world16_checkpoint='/qualification/own_READY',source_identity_sha256='1'*64,
        source_checkpoint_files_sha256='2'*64,source_checkpoint_identity_file_sha256='3'*64,
        source_world_transition_sha256='4'*64,common_optimizer_sha256='5'*64,
        origin_updates=4,flags=profile,source_qualification=True,target_qualification=True,
        old_inventory_sha256='6'*64,new_inventory_sha256=fingerprint(sources))
    value=dict(world=16,recipe={'arm':'D3_FPO','inner_updates':2},qualification=True,fresh_Adam=False,
        scope={'unchanged':True},source_files=sources,execution_upgrade=meta,
        native_world_transition={'origin_updates':2,'target_qualification':True})
    value['sha256']=fingerprint(value);return value


def profile(**kw):return dict.fromkeys(FLAG_NAMES,False)|kw


def test_bound_identity_and_default_false_unsupported_flags(monkeypatch):
    ext=FastWorld16Extension();ext.saved_identity=saved(profile(fast_prefix=True,fuse_fm_mc=True))
    monkeypatch.setattr(runtime,'source_inventory',lambda:ext.saved_identity['source_files'])
    actual=copy.deepcopy(ext.saved_identity);actual.pop('execution_upgrade');actual.pop('native_world_transition');actual['fresh_Adam']=True
    assert ext.identity(actual)==ext.saved_identity
    bad=copy.deepcopy(actual);bad['scope']['unchanged']=False
    with pytest.raises(ValueError,match='closure'):ext.identity(bad)
    for key in('no_grad_prefix_cache','async_scoring'):
        ext.saved_identity=saved(profile(**{key:True}))
        with pytest.raises(ValueError,match=key):ext.execution_profile()
    ext.saved_identity={}
    assert ext.execution_profile()==profile()


def test_real_tiny_FPO_activated_flags_cache_and_fresh_energy_ratio(monkeypatch):
    from tools.d46_world16.qualify_native import make_actor,condition,SyntheticService
    from starVLA.rl.d44_native_rl.config import defaults46
    from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**kw:None)
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(42);cfg=defaults46('D3_FPO');actor=make_actor(cfg)
        original_state_keys=tuple(actor.state_dict());param_ids=[id(p)for p in actor.parameters()]
        global_rng=torch.get_rng_state().clone();ext=FastWorld16Extension()
        ext.saved_identity=saved(profile(fast_prefix=True,fuse_fm_mc=True,rotary_cache=True));ext.configure_actor(actor)
        assert torch.equal(global_rng,torch.get_rng_state())
        assert tuple(actor.state_dict())==original_state_keys and[id(p)for p in actor.parameters()]==param_ids
        assert actor.execution_fast_prefix is True and actor.execution_fuse_fm_mc is True
        assert len(ext._execution_caches)==2
        g=torch.Generator().manual_seed(42)
        scene=collect_scene(actor,condition(actor,0),'TRAIN_CPU_0','TRAIN_log',SyntheticService(),g,cfg,0.,{'run':'CPU','update':0,'collection':1})
        scene.update(advantages=torch.linspace(-1,1,16),beta=.001,expect_fresh=True)
        loss,metrics=actor(scene);loss.backward()
        assert torch.isfinite(loss)and metrics['fresh_ratio_error']==0
        assert metrics['fpo_joint_logratio_min']==0 and metrics['fpo_joint_logratio_max']==0
        assert all(torch.isfinite(p.grad).all()for p in actor.parameters()if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in actor.parameters()if p.requires_grad)
        assert all(cache.stats['rotary_hits']>0 and cache.stats['additive_hits']>0 for cache in ext._execution_caches)
        assert all(not p.requires_grad for p in actor.ref_action_expert.parameters())
        ext.close_execution_caches()
        assert not hasattr(actor.action_expert.decision,'_d46_invariant_cache')
    finally:torch.set_num_threads(previous_threads)


def test_legacy_actor_hook_is_opt_in_and_runs_after_clone_parity():
    from types import SimpleNamespace
    ext=FastWorld16Extension();ext.saved_identity={}
    actor=SimpleNamespace(execution_fast_prefix=False,execution_fuse_fm_mc=False)
    ext.configure_actor(actor)
    assert ext.actor_execution_profile==profile()and ext._execution_caches==()
    assert not actor.execution_fast_prefix and not actor.execution_fuse_fm_mc
    tree=ast.parse((Path(__file__).parents[1]/'tools/d44_native_rl/train.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    calls=sorted((n.lineno,ast.unparse(n.func))for n in ast.walk(main)if isinstance(n,ast.Call))
    after=next(line for line,name in calls if name=='extension.after_restore')
    configure=next(line for line,name in calls if name=='configure_actor')
    attempt=next(line for line,name in calls if name=='extension.make_attempt')
    assert after<configure<attempt
