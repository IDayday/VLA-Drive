"""One-factor sigma upper-bound extension; no GPU qualification matrix."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from test_qwen_world_fpo_training import recipe, collected, identity, state
from test_qwen_world_rescue_auxiliary import TinyActor
from starVLA.rl.qwen_world_research_algorithms import fpo_training as fpo
from starVLA.rl.qwen_world_research_algorithms import safety_efficiency as se
from starVLA.rl.qwen_world_research_algorithms.fpo_entry import FPORuntimeMixin
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest


def fulltime_recipe():
    cfg=recipe()
    cfg['research'].update(arm=fpo.FULLTIME_ARM,fpo=copy.deepcopy(fpo.FULLTIME_CONFIG))
    return cfg


def test_original_defaults_exact_draws_and_only_fulltime_upper_bound():
    cfg=fulltime_recipe()
    assert fpo.validate_recipe(cfg) is cfg
    assert fpo.CONFIG['sigma_max']==.8 and fpo.PROTOCOL['sigma_max']==.8
    assert fpo.protocol_for_recipe()==fpo.PROTOCOL
    assert fpo.protocol_for_recipe(recipe())==fpo.PROTOCOL
    _,out=collected()
    rng=torch.random.get_rng_state().clone()
    seed,old_sigma,old_noise=fpo._draws(out,0)
    same=fpo._draws(out,0,recipe())
    assert seed==same[0] and torch.equal(old_sigma,same[1]) and torch.equal(old_noise,same[2])
    generator=torch.Generator().manual_seed(seed)
    uniform=torch.rand(2,8,generator=generator,dtype=torch.float32)
    noise=torch.randn(2,8,8,3,generator=generator,dtype=torch.float32)
    assert torch.equal(old_sigma,.2+.6*uniform) and torch.equal(old_noise,noise)
    new_seed,new_sigma,new_noise=fpo._draws(out,0,cfg)
    assert new_seed==seed and torch.equal(new_noise,old_noise)
    assert torch.equal(new_sigma,.2+.8*uniform)
    assert bool((new_sigma>=.2).all() and (new_sigma<1).all() and (new_sigma>.8).any())
    assert torch.equal(rng,torch.random.get_rng_state())
    for arm,settings in [(fpo.ARM,fpo.FULLTIME_CONFIG),(fpo.FULLTIME_ARM,fpo.CONFIG),
                         ('PPO_OTHER',fpo.FULLTIME_CONFIG),
                         (fpo.FULLTIME_ARM,dict(fpo.FULLTIME_CONFIG,sigma_max=1.01)),
                         (fpo.FULLTIME_ARM,dict(fpo.FULLTIME_CONFIG,sigma_min=0.)),
                         (fpo.FULLTIME_ARM,dict(fpo.FULLTIME_CONFIG,mc_samples=4))]:
        bad=copy.deepcopy(cfg);bad['research'].update(arm=arm,fpo=settings)
        with pytest.raises(ValueError):fpo.validate_recipe(bad)


def test_fulltime_runtime_four_inner_native_eight_ranks_and_protocol_binding(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setenv('RANK','0')
    cfg=fulltime_recipe()
    pairs=[collected(rank) for rank in range(8)]
    rewards=torch.cat([r['rewards'].double() for _,r in pairs])
    components=torch.cat([se.canonical_components(r['records'])['components'] for _,r in pairs])
    material=torch.cat((rewards[:,:,None],components),-1)
    monkeypatch.setattr(se.dist,'is_available',lambda:True)
    monkeypatch.setattr(se.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(se.dist,'get_world_size',lambda:8)
    def gather(out,local):
        for rank in range(8):out[rank].copy_(material[rank:rank+1])
    monkeypatch.setattr(se.dist,'all_gather',gather)
    def check(error):
        if error:raise AssertionError(error)
    monkeypatch.setattr(checkpoint,'check_errors',check)
    class Parent:
        def load_recipe(self,args):return self.cfg
        def make_actor(self,cfg,args,device):
            self.actor=TinyActor(cfg);return self.actor,None,None
        def recipe_identity(self,cfg,args,condition_sha):return dict(recipe=cfg)
    class Runtime(FPORuntimeMixin,Parent):pass
    rt=Runtime();rt.cfg=cfg;rt.args=SimpleNamespace(output=str(tmp_path))
    assert rt.load_recipe(rt.args) is cfg
    actor,_,_=rt.make_actor(cfg,rt.args,torch.device('cpu'))
    out=pairs[0][1]
    rt.finalize_advantages(out)
    pairs[0]=(actor,out)
    protocol=fpo.protocol_for_recipe(cfg)
    assert protocol['sigma_max']==1. and protocol!=fpo.PROTOCOL
    scale=json.loads((tmp_path/'initial_FPO_scale_rank0.json').read_text())
    assert scale['protocol']==protocol and scale['sigma_max']>.8
    assert rt.recipe_identity(cfg,rt.args,'condition')['fpo_protocol']==protocol
    ident=identity();ident.update(recipe=cfg,fpo_protocol=protocol)
    summaries=[]
    for rank,(actor,out) in enumerate(pairs):
        if rank:
            se.finalize_safety_efficiency_advantages(out)
            fpo.attach_material(actor,out,rank=rank,cfg=cfg)
        path=tmp_path/f'fulltime_rank{rank}.pt'
        torch.save(out,path);saved=torch.load(path,weights_only=False)
        before=rollout_digest(saved)
        for inner in range(4):
            sm=fpo.validate_boundary(state(inner),saved,ident,
                optimizer=dict(state={0:dict(step=inner)},param_groups=[dict(params=[0])]),
                scheduler=dict(last_epoch=inner),rank=rank)
            if inner==0:summaries.append(sm)
        fpo.validate_material(saved,rank=rank,cfg=cfg)
        with pytest.raises(ValueError,match='identity/digest'):
            fpo.validate_material(saved,rank=rank)  # Cannot reinterpret new material as old .8.
        if rank==0:
            actor._loss=fpo.FPOTrainingLoss(cfg)
            for inner in range(4):
                actor.p.grad=None
                loss,metrics=actor._loss._policy_loss(actor,saved)
                assert metrics['fpo_fresh_checked']==float(inner==0)
                loss.backward()
                assert torch.isfinite(actor.p.grad) and actor.p.grad.abs()>0
                with torch.no_grad():actor.p.sub_(1e-5*actor.p.grad)
                assert rollout_digest(saved)==before
            fpo.validate_material(saved,rank=rank,cfg=cfg)
        bad=copy.deepcopy(saved)
        bad['fpo']['sigma'].reshape(-1)[0]=.99
        bad['fpo']['material_sha256']=fpo.material_digest(bad['fpo'])
        with pytest.raises(ValueError,match='draw changed'):
            fpo.validate_material(bad,rank=rank,cfg=cfg)
    fpo.validate_ranks(summaries)
    bad=copy.deepcopy(summaries);bad[3]['fpo_protocol']=copy.deepcopy(fpo.PROTOCOL)
    with pytest.raises(ValueError,match='rank protocol'):fpo.validate_ranks(bad)
    bad=copy.deepcopy(ident);bad['fpo_protocol']=copy.deepcopy(fpo.PROTOCOL)
    with pytest.raises(ValueError,match='explicit FPO identity'):
        fpo.validate_boundary(state(),pairs[0][1],bad,rank=0)
