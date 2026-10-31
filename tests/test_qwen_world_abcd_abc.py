"""ABC credit recovery/source loss plus fullTRAIN leakage/cursor contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from starVLA.rl.qwen_world_abcd_research import abc,data
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from starVLA.rl.qwen_world_compare.runtime import expected_recipe
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss


def cfg(arch='A'):
    c=expected_recipe(arch);c['research']={'abc_objective':dict(abc.ABC_OBJECTIVE_CONFIG)}
    return c


def make_material():
    x=torch.ones(8,8,4,dtype=torch.float64)
    x[:,:,3]=torch.linspace(.1,.8,8,dtype=torch.float64)
    x[0,4:,2]=0;x[1,:,0]=0;x[1,4:,1]=0;x[2,:,3]=1
    rows=[]
    for group in x:
        records=[]
        for nc,dac,ttc,ep in group.tolist():
            score=nc*dac*(5*ep+5*ttc+2)/12
            records.append(dict(status='OK',metrics=dict(no_at_fault_collisions=nc,
                drivable_area_compliance=dac,time_to_collision_within_bound=ttc,
                ego_progress=ep,comfort=1.,score=score),readouts=dict(canonical_PDMS=score)))
        rows.append(records)
    rewards=torch.cat([abc.canonical_components(r)['canonical_rewards'] for r in rows])
    return rows,torch.cat((rewards.double().unsqueeze(-1),x),-1)


def test_finalize_and_pending_recovery_validate_global64_not_raw_reward_std(monkeypatch):
    records,material=make_material()
    def gather(out,local):
        for r,v in enumerate(out):v.copy_(material[r:r+1])
    monkeypatch.setattr(abc,'dist',SimpleNamespace(is_available=lambda:True,is_initialized=lambda:True,
        get_world_size=lambda:8,all_gather=gather))
    body=dict(schema='qwen_world_compare_RL_v1',architecture_id='A',recipe=cfg(),world=8,
              projected_condition_identity='fixed-current')
    identity=dict(body,sha256=fingerprint(body));summaries=[];rollouts=[]
    state=dict(updates=0,collections=1,next_inner=0,scene_cursor=8,raw_scene_exposures=8,
               new_candidates=64,phase='ROLLOUT_COMMITTED')
    for rank in range(8):
        token=f'token{rank}'
        r=dict(records=records[rank],rewards=abc.canonical_components(records[rank])['canonical_rewards'],
            token=token,log='TRAINlog',cond_bg=dict(schema='qwen_world_compare_current_descriptor_v1',
                token=token,identity='fixed-current',repeats=8),chain=torch.zeros(8,11,8,3),
            timesteps=torch.arange(10),deltas=torch.full((10,),-.1),stoch_idx=torch.tensor([7,8,9]),
            old_logp=torch.zeros(8,3),old_probability=dict(source_clamped_logprob=torch.zeros(8,3,8,3)),
            behavior_version=dict(run=identity['sha256'],update=0,adapter_sha256='same'))
        abc.finalize_advantages(r)
        summaries.append(abc.validate_boundary(state,r,identity));rollouts.append(r)
    abc.validate_ranks(summaries)
    assert not bool(rollouts[2]['adv_mask'].any())  # successful EP1 saturation
    assert bool(rollouts[1]['adv_mask'].any())      # allfail varying cost has direction
    assert bool((rollouts[0]['advantages'][4:]<0).all())
    with pytest.raises(ValueError):abc.finalize_advantages(rollouts[0])
    corrupt=copy.deepcopy(summaries);corrupt[0]['advantages'][0]+=.1
    with pytest.raises(ValueError):abc.validate_ranks(corrupt)
    corrupt=copy.deepcopy(rollouts[0]);corrupt['canonical_rewards'][0,0]=.123
    with pytest.raises(ValueError):abc.validate_boundary(state,corrupt,identity)


class Toy(torch.nn.Module):
    grpo_kl_beta=50.;grpo_denoising_discount=1.;grpo_bc_coeff=0.;grpo_kl_type='x'
    grpo_use_bc_loss=False
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.linspace(-.02,.02,8*3*8*3,dtype=torch.float64).reshape(8,3,8,3))
    def action_chain_logprobs(self,*args,use_ref=False,return_terms=False,**kwargs):
        value=self.p*0 if use_ref else self.p
        logp=value.mean((-1,-2));terms=dict(mean=value,vel=value*2,std=torch.tensor([.05,.04,.03]))
        return (logp,terms) if return_terms else logp
    def kl_weight_matrix(self,*,horizon,action_dim,device,dtype):
        return torch.tensor([.2,1,1],device=device,dtype=dtype)[None].expand(horizon,action_dim)


def test_objective_preserves_original_PPO_anchor_value_gradient_and_rejects_science_changes():
    a=Toy();b=copy.deepcopy(a)
    r=dict(chain=torch.zeros(8,11,8,3),timesteps=torch.arange(10),deltas=torch.full((10,),-.1),
        stoch_idx=torch.tensor([7,8,9]),cond_bg={},old_logp=torch.zeros(8,3),
        advantages=torch.tensor([1.,-1.,.2,-.3,0.,.1,.5,-.5]),adv_mask=torch.ones(8),
        structured_advantage=dict(protocol=abc.STRUCTURED_PROTOCOL))
    x,_=SourceLoss(cfg())._policy_loss(a,r);y,_=abc.ABCSafetyLoss(cfg())._policy_loss(b,r)
    x.backward();y.backward();torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(a.p.grad,b.p.grad,rtol=0,atol=0)
    for arch in ('A','B','C'):abc.validate_recipe(cfg(arch))
    bad=cfg();bad['learning_rate']=1e-4
    with pytest.raises(ValueError):abc.validate_recipe(bad)
    with pytest.raises(ValueError):abc.validate_recipe(cfg('D'))


def fixture_data(root,monkeypatch):
    monkeypatch.setattr(data,'EXPECTED_TRAIN_COUNT',18);monkeypatch.setattr(data,'EXPECTED_TRAIN_PHYSICAL_LOGS',2)
    root=Path(root);train=root/'TRAIN';train.mkdir()
    tokens=[f't{i:015x}' for i in range(18)];dev='dev';test='test';embargo='embargo'
    logs={t:f'train{i%2}_00010_00020' for i,t in enumerate(tokens)}
    logs.update(dev='devlog_00010_00020',embargo='devlog_00021_00030',test='testlog_00010_00020')
    pop=root/'population.json';pop.write_text(json.dumps(dict(token_logs=logs)))
    roles=dict(schema='unit',population_sha256=file_sha(pop),train_tokens=tokens,dev_tokens=[dev],
               navtest_tokens=[test],embargoed_train_tokens=[embargo])
    part=root/'DATA_PARTITION.json';part.write_text(json.dumps(roles))
    m=dict(cohort='TRAIN',count=18,tokens=tokens,population_sha256=file_sha(pop),role_partition_sha256=fingerprint(roles))
    (train/'manifest.json').write_text(json.dumps(m));(train/'COMPLETE').write_text(json.dumps(dict(manifest_sha256=file_sha(train/'manifest.json'))))
    return train,part,pop


def test_fullTRAIN_cursor_resume_epoch_padding_and_immutable_manifest(tmp_path,monkeypatch):
    train,part,pop=fixture_data(tmp_path,monkeypatch);out=tmp_path/'FULL_TRAIN.json'
    data.write_full_train_manifest(out,train,partition_path=part,population_path=pop)
    a=data.FullTRAINTokenStream(out,8);b=data.FullTRAINTokenStream(out,8)
    assert a.epoch_size==24
    for pos in [0,7,16,17,18,23,24,35,47,48]:assert a.token_at(pos)==b.token_at(pos)
    assert sum(a.at(16,r)[1]['padding'] for r in range(8))==6
    assert len({a.at(16,r)[0] for r in range(8)})==8
    assert a.at(24,0)[0]==a.token_at(24)
    before=out.read_bytes();data.write_full_train_manifest(out,train,partition_path=part,population_path=pop)
    assert out.read_bytes()==before
    with pytest.raises(ValueError):a.at(3,0)


def test_fullTRAIN_rejects_resealed_DEV_embargo_leak(tmp_path,monkeypatch):
    train,part,pop=fixture_data(tmp_path,monkeypatch)
    roles=json.loads(part.read_text());roles['train_tokens'][0]='embargo';part.write_text(json.dumps(roles))
    m=json.loads((train/'manifest.json').read_text());m['tokens']=roles['train_tokens'];m['role_partition_sha256']=fingerprint(roles)
    (train/'manifest.json').write_text(json.dumps(m));(train/'COMPLETE').write_text(json.dumps(dict(manifest_sha256=file_sha(train/'manifest.json'))))
    with pytest.raises(ValueError):data.write_full_train_manifest(tmp_path/'bad.json',train,partition_path=part,population_path=pop)


SHARED_BASE_REPORTS={}


@pytest.mark.parametrize('architecture',['A','B','C','D'])
def test_shared_frozen_base_matches_old_actor_four_inners(architecture):
    """One actual tiny architecture each; no real Qwen checkpoint or scoring."""
    import importlib.util
    from starVLA.rl.qwen_world_abcd_research.actor import ResearchActor
    from starVLA.rl.qwen_world_abcd_research import d_objective as d
    from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
    from starVLA.rl.qwen_world_compare.deployment import prune_deployment
    from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear

    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        path=Path(__file__).parent/'world_compare/test_rl_adapter.py'
        spec=importlib.util.spec_from_file_location('_shared_base_tiny_fixture',path)
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        base=fixture.TinyDeployment(architecture)
        prune_deployment(base,copy_model=False)
        actual=cfg(architecture) if architecture!='D' else expected_recipe('D')
        if architecture=='D':
            actual.update(learning_rate=1e-6,research=dict(d_objective=copy.deepcopy(d.CONFIG)))
            actual['grpo']['sample']['randn_clip']=None
            actual['grpo']['train'].update(ppo_clip_range=.2,kl_beta=.001,kl_type='x_kl',
                kl_dim_weights=None,kl_horizon_weights=None)
        backend=copy.deepcopy(actual)
        if architecture=='D':backend['grpo']['sample']['randn_clip']=float('inf')
        old=FourFrameworkRecipeActor(copy.deepcopy(base),backend)
        new=ResearchActor(copy.deepcopy(base),actual)
        del base
        # Frozen graph nodes are independent. Their immutable Parameters alone
        # are shared; reference has no LoRA and cannot see current adapters.
        shared=0
        for name,refmodule in new.ref_action_expert.named_modules():
            current=new.action_expert.get_submodule(name) if name else new.action_expert
            if isinstance(current,LoRALinear):current=current.base
            assert current is not refmodule
            for key,p in refmodule.named_parameters(recurse=False):
                if p is None:continue
                q=current.get_parameter(key)
                assert p is q and not p.requires_grad
                shared+=p.numel()
        assert not any('lora_' in n for n,_ in new.ref_action_expert.named_parameters())
        old_params={n:p for n,p in old.named_parameters() if p.requires_grad}
        new_params={n:p for n,p in new.named_parameters() if p.requires_grad}
        assert set(old_params)==set(new_params)==set(new.closure['trainable_names'])
        expected_numel={'A':8192,'B':8192,'C':15232,'D':10752}[architecture]
        assert sum(p.numel() for p in new_params.values())==expected_numel
        for name in old_params:assert torch.equal(old_params[name],new_params[name])
        cond=fixture.condition(new.action_expert)
        gen0=torch.Generator().manual_seed(42);gen1=torch.Generator().manual_seed(42)
        chain0,times0,deltas0,steps0=old.sample_action_chain(cond,8,generator=gen0)
        chain1,times1,deltas1,steps1=new.sample_action_chain(cond,8,generator=gen1)
        assert all(torch.equal(x,y) for x,y in zip((chain0,times0,deltas0,steps0),(chain1,times1,deltas1,steps1)))
        assert torch.equal(gen0.get_state(),gen1.get_state())
        lp0,terms0=old.action_chain_logprobs(cond,chain0,times0,deltas0,stoch_idx=steps0,return_terms=True)
        old_probability=copy.deepcopy(old._probability_audit['current'])
        lp1,terms1=new.action_chain_logprobs(cond,chain1,times1,deltas1,stoch_idx=steps1,return_terms=True)
        assert torch.equal(lp0,lp1) and torch.equal(terms0['mean'],terms1['mean']) and torch.equal(terms0['std'],terms1['std'])
        with torch.no_grad():
            _,ref0=old.action_chain_logprobs(cond,chain0,times0,deltas0,stoch_idx=steps0,use_ref=True,return_terms=True)
            _,ref1=new.action_chain_logprobs(cond,chain1,times1,deltas1,stoch_idx=steps1,use_ref=True,return_terms=True)
        assert torch.equal(ref0['mean'],ref1['mean'])
        records=make_material()[0][0]
        r=dict(cond_bg=cond,chain=chain0.detach(),timesteps=times0,deltas=deltas0,stoch_idx=steps0,
            old_logp=lp0.detach(),old_probability=old_probability,records=records,
            rewards=abc.canonical_components(records)['canonical_rewards'],token='syntheticTRAIN',log='unit',
            behavior_version=dict(run='unit',update=0,adapter_sha256='same'))
        if architecture=='D':d.finalize_advantages(r)
        else:abc.finalize_advantages(r)
        r0=copy.deepcopy(r);r1=copy.deepcopy(r)
        old._loss=d.DActionJaTLoss(actual) if architecture=='D' else abc.ABCSafetyLoss(actual)
        new._loss=d.DActionJaTLoss(actual) if architecture=='D' else abc.ABCSafetyLoss(actual)
        if architecture=='D':
            assert actual['grpo']['sample']['randn_clip'] is None
            assert old.grpo_randn_clip==new.grpo_randn_clip==float('inf')
            z=torch.tensor([-8.,0.,8.]);assert torch.equal(z,z.clamp(-new.grpo_randn_clip,new.grpo_randn_clip))
        frozen={n:p.detach().clone() for n,p in new.action_expert.named_parameters() if not p.requires_grad}
        reference={n:p.detach().clone() for n,p in new.ref_action_expert.named_parameters()}
        buffers={n:p.detach().clone() for n,p in new.named_buffers()}
        opts=[torch.optim.AdamW(list(ps.values()),lr=actual['learning_rate'],betas=(.9,.95),
            eps=1e-8,weight_decay=.01) for ps in (old_params,new_params)]
        losses=[]
        for inner in range(4):
            for opt in opts:opt.zero_grad(set_to_none=True)
            loss0,_=old(r0);loss1,_=new(r1)
            assert torch.equal(loss0,loss1)
            loss0.backward();loss1.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new_params.values())
            for name in old_params:assert torch.equal(old_params[name].grad,new_params[name].grad)
            losses.append(float(loss1.detach()))
            for opt,ps in zip(opts,(old_params,new_params)):
                torch.nn.utils.clip_grad_norm_(list(ps.values()),1.);opt.step()
            for name in old_params:assert torch.equal(old_params[name],new_params[name])
            old.set_policy_version(f'inner{inner+1}');new.set_policy_version(f'inner{inner+1}')
            with torch.no_grad():
                _,a=old.action_chain_logprobs(cond,chain0,times0,deltas0,stoch_idx=steps0,use_ref=True,return_terms=True)
                _,b=new.action_chain_logprobs(cond,chain1,times1,deltas1,stoch_idx=steps1,use_ref=True,return_terms=True)
            assert torch.equal(a['mean'],b['mean']) and torch.equal(b['mean'],ref1['mean'])
        for name,p in new.action_expert.named_parameters():
            if name in frozen:assert torch.equal(p,frozen[name])
        for name,p in new.ref_action_expert.named_parameters():assert torch.equal(p,reference[name])
        for name,p in new.named_buffers():assert torch.equal(p,buffers[name])
        assert any(bool(p.detach().ne(0).any()) for n,p in new_params.items() if n.endswith('lora_B'))
        SHARED_BASE_REPORTS[architecture]=dict(status='PASS',tiny_trainable_numel=expected_numel,
            shared_frozen_parameter_numel=shared,sample_current_reference_terms_bit_equal=True,
            four_inner_losses=losses,four_inner_loss_grad_and_Adam_parameters_bit_equal=True,
            frozen_parameters_reference_and_buffers_unchanged=True,
            D_semantic_null_runtime_infinity_identity_clip=architecture=='D',GPU_used=False)
    finally:torch.set_num_threads(threads)


def test_new_checkpoint_save_restore_really_uses_runtime_validators(tmp_path,monkeypatch):
    from starVLA.rl.qwen_world_abcd_research import checkpoint
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def forbidden(*args,**kwargs):raise AssertionError('Original canonical fallback must not run')
    monkeypatch.setattr(checkpoint,'source_boundary',forbidden);monkeypatch.setattr(checkpoint,'source_ranks',forbidden)
    calls=[]
    class Hooks:
        def validate_boundary(self,state,rollout,identity,optimizer=None,scheduler=None):
            calls.append('boundary')
            return dict(phase=state['phase'],updates=state['updates'],next_inner=state['next_inner'],proof='runtime-hook')
        def validate_ranks(self,summaries):calls.append('ranks');return summaries
    actor=torch.nn.Linear(2,1);actor.device=torch.device('cpu')
    opt=torch.optim.AdamW(actor.parameters(),lr=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    actor(torch.ones(1,2)).sum().backward();opt.step();scheduler.step();opt.zero_grad(set_to_none=True)
    identity=dict(world=1,sha256='CPU-hook-test');state=dict(phase='INNER_COMMITTED',updates=1,next_inner=1)
    rollout=dict(marker=torch.tensor([7.]));generator=torch.Generator().manual_seed(42);rng=generator.get_state().clone()
    saved={n:p.clone() for n,p in actor.named_parameters()};hooks=Hooks()
    cp=checkpoint.save(tmp_path,actor,opt,scheduler,generator,state,rollout,identity,runtime=hooks)
    with torch.no_grad():
        for p in actor.parameters():p.add_(10)
    generator.manual_seed(99)
    restored,pending=checkpoint.restore(cp,actor,opt,scheduler,generator,identity,runtime=hooks)
    assert calls==['boundary','ranks','boundary','ranks']
    assert restored==state and torch.equal(pending['marker'],rollout['marker'])
    assert torch.equal(generator.get_state(),rng)
    assert all(torch.equal(p,saved[n]) for n,p in actor.named_parameters())
    assert all(s['step']==1 for s in opt.state.values()) and scheduler.state_dict()['last_epoch']==1
    SHARED_BASE_REPORTS['checkpoint']=dict(status='PASS',runtime_hooks_called=calls,
        fallback_forbidden=True,CPU_toy_roundtrip_restored_parameters_Adam_RNG_pending=True)
