"""Directed G8 parity / G16 collection, global credit and native-boundary checks."""
import copy
from types import SimpleNamespace

import pytest
import torch

from starVLA.rl.qwen_world_safety_exploration import credit as C, recovery as R
from starVLA.rl.qwen_world_safety_exploration.group_runtime import (
    GroupRuntime, validate_recipe, collect_group, SCHEMA, CAMPAIGN,
)
from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
from starVLA.rl.qwen_world_postcot.scope_contracts import expected_scope_recipe, MEMORY
from starVLA.rl.qwen_world_research_algorithms import safety_efficiency as old
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss


def cfg(g):
    c = expected_scope_recipe(MEMORY)
    c['grpo']['sample']['group_size'] = g
    c['logical_budget']['candidates'] = 100000*g
    c['research'] = dict(schema=SCHEMA, campaign_root=str(CAMPAIGN), arm='G'+str(g),
        policy_objective='ppo', data_policy='uniform_full', structured_advantage=C.STRUCTURED_SCHEMA)
    return c


def components(g):
    x = torch.ones(8, g, 4, dtype=torch.float64)
    x[:, :, 3] = torch.linspace(.1, 1., g, dtype=torch.float64)
    x[0, g//2:, 2] = 0
    x[1, 1:, 0] = 0
    x[2, :, 0] = 0  # true allfail costtie, EP differs
    x[3, :, 0] = 0; x[3, g//2:, 1] = 0
    x[4, :, 3] = 1  # successful saturation, C can still differ
    x[5, :, 3] = .37
    x[6, :, :3] = torch.tensor([0., .5, 1.])
    x[7, 1::2, 0] = .5
    return x


def records(x):
    a = []
    for j, (nc, dac, ttc, ep) in enumerate(x.tolist()):
        comfort = float(j % 2); score = nc*dac*(5*ep+5*ttc+2*comfort)/12
        a.append(dict(status='OK', metrics=dict(no_at_fault_collisions=nc,
            drivable_area_compliance=dac, time_to_collision_within_bound=ttc,
            ego_progress=ep, comfort=comfort, score=score, driving_direction_compliance=1.),
            readouts=dict(canonical_PDMS=score), scoring_seconds=.001))
    return a


def rollout(rank, g, rec):
    return dict(rewards=C.canonical_components(rec, g)['canonical_rewards'], records=rec,
        group_size=g, chain=torch.zeros(g,11,8,3), old_logp=torch.zeros(g,3),
        timesteps=torch.arange(10), deltas=-torch.ones(10)/10, stoch_idx=torch.tensor([7,8,9]),
        old_probability=dict(source_clamped_logprob=torch.zeros(g,3,8,3)),
        token='s'+str(rank), log='log'+str(rank),
        behavior_version=dict(run='test',update=0,adapter_sha256='same'),
        cond_bg=dict(schema='world_postcot_trainable_memory_observation_v1',
            token='s'+str(rank),identity='condition',repeats=g))


def identity(g):
    return dict(world=8,sha256='test',schema='world_postcot_cleanx_scope_RL_v1',
                projected_condition_identity='condition',recipe=cfg(g))


def state(g, u):
    return dict(updates=u,collections=1,next_inner=u,scene_cursor=8,
        raw_scene_exposures=8,new_candidates=8*g,
        phase='ROLLOUT_COMMITTED' if u == 0 else 'INNER_COMMITTED')


def gather(monkeypatch, material):
    calls=[]
    monkeypatch.setattr(C.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(C.dist,'get_world_size',lambda:8)
    def allgather(out, value):
        calls.append(value.clone())
        for rank in range(8): out[rank].copy_(material[rank:rank+1])
    monkeypatch.setattr(C.dist,'all_gather',allgather)
    return calls


def test_g8_exact_baseline_and_g16_global_math_and_targeted_guards(monkeypatch):
    for g in (8,16):
        validate_recipe(cfg(g)); credit=C.GroupCredit(g); x=components(g)
        recs=[records(v) for v in x]
        rewards=torch.cat([C.canonical_components(r,g)['canonical_rewards'] for r in recs])
        material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
        calls=gather(monkeypatch,material)
        saved=[]
        for rank in range(8):
            r=credit.finalize(rollout(rank,g,recs[rank])); saved.append(r)
            assert r['advantage_statistics']['normalization_population_candidates']==8*g
            assert torch.equal(r['rewards'],r['canonical_rewards'])
            assert not bool((r['advantages'].gt(0)&~r['structured_advantage']['feasible'].reshape(-1)).any()) if rank==0 else True
        assert len(calls)==8  # once per rank, never four times per inner
        assert saved[2]['adv_mask'].sum()==0 and saved[4]['adv_mask'].sum()==0
        # Independent scalar group calculation, with exact registered global scale.
        raw=[]
        for rows in x.tolist():
            f=[float(a==b==c==1) for a,b,c,e in rows]; n=sum(f)
            costs=[(3-a-b-c)/3 for a,b,c,e in rows]
            mean_ep=sum(v*row[3] for v,row in zip(f,rows))/n if n else 0
            raw.append([(v-n/g if n else sum(costs)/g-costs[j])+
                        (v*(rows[j][3]-mean_ep) if n>=2 else 0) for j,v in enumerate(f)])
        raw=torch.tensor(raw,dtype=torch.float64)
        want=(raw/(raw.std(unbiased=False)+1e-8)).float()
        torch.testing.assert_close(torch.stack([r['advantages'] for r in saved]),want,atol=1e-7,rtol=1e-6)
        if g==8:
            expected=old.safety_efficiency_advantages(rewards,x,all_material=material)
            actual=credit.advantages(rewards,x,all_material=material)
            for k in ('A_safe','A_EP','A_raw','advantages','adv_mask','normalization_std','global_std'):
                assert torch.equal(actual[k],expected[k]),k
        bad=cfg(g);bad['grpo']['sample']['sde_exclude_last_step']=True
        with pytest.raises(ValueError):validate_recipe(bad)
        bad=cfg(g);bad['grpo']['sample']['group_size']=32
        with pytest.raises(ValueError):validate_recipe(bad)


class ToyModel(torch.nn.Module):
    grpo_kl_beta=50.;grpo_denoising_discount=1.;grpo_use_bc_loss=False
    grpo_bc_coeff=.1;grpo_kl_type='x'
    def __init__(self,g):
        super().__init__();self.p=torch.nn.Parameter(torch.linspace(-.001,.001,g*3*8*3).reshape(g,3,8,3))
    def action_chain_logprobs(self,*args,use_ref=False,return_terms=False,**kw):
        v=self.p*0 if use_ref else self.p
        out=v.mean((-1,-2));terms=dict(mean=v,vel=v*2,std=torch.tensor([.05,.04,.03]))
        return (out,terms) if return_terms else out
    def kl_weight_matrix(self,*,horizon,action_dim,device,dtype):
        return torch.tensor([.2,1.,1.],device=device,dtype=dtype)[None,:].expand(horizon,action_dim)


def test_eight_rank_torch_roundtrip_inner_recovery_gradients_and_corruption(monkeypatch,tmp_path):
    for g in (8,16):
        credit=C.GroupCredit(g);x=components(g);recs=[records(v) for v in x]
        rewards=torch.cat([C.canonical_components(r,g)['canonical_rewards'] for r in recs])
        material=torch.cat((rewards.double().unsqueeze(-1),x),-1)
        gather(monkeypatch,material)
        rs=[credit.finalize(rollout(rank,g,recs[rank])) for rank in range(8)]
        model=ToyModel(g);loss=ResearchLoss(cfg(g));opt=torch.optim.AdamW(model.parameters(),lr=5e-5)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        original_id=id(rs[0]['advantages']);original_digest=R.tensor_digest(rs[0])
        for u in range(4):
            summaries=[R.validate_boundary(state(g,u),r,identity(g),credit=credit,
                        recipe_validator=validate_recipe) for r in rs]
            R.validate_ranks(summaries,credit=credit)
            path=tmp_path/f'G{g}_inner{u}.pt'
            torch.save(dict(rollout=rs[0],summary=summaries[0],state=state(g,u),optimizer=opt.state_dict(),scheduler=sched.state_dict()),path)
            loaded=torch.load(path,weights_only=False)
            got=R.validate_boundary(loaded['state'],loaded['rollout'],identity(g),
                loaded['optimizer'],loaded['scheduler'],credit=credit,recipe_validator=validate_recipe)
            assert got==loaded['summary']
            other=copy.deepcopy(model)
            value,_=loss._policy_loss(model,rs[0]);expected,_=SourceLoss(cfg(g))._policy_loss(other,rs[0])
            value.backward();expected.backward()
            assert torch.equal(value,expected) and torch.equal(model.p.grad,other.p.grad)
            assert torch.isfinite(model.p.grad).all() and model.p.grad.abs().sum()>0
            opt.step();sched.step();opt.zero_grad(set_to_none=True)
            assert id(rs[0]['advantages'])==original_id and R.tensor_digest(rs[0])==original_digest
        for key in ('new_candidates','scene_cursor'):
            bad=state(g,1);bad[key]+=1
            with pytest.raises(ValueError):R.validate_boundary(bad,rs[0],identity(g),credit=credit,recipe_validator=validate_recipe)
        bad=copy.deepcopy(rs[0]);bad['cond_bg']['repeats']=24-g
        with pytest.raises(ValueError):R.validate_boundary(state(g,1),bad,identity(g),credit=credit,recipe_validator=validate_recipe)
        bad=copy.deepcopy(rs[0]);bad['structured_advantage']['A_raw'][0,0]+=.01
        with pytest.raises(ValueError):R.validate_boundary(state(g,1),bad,identity(g),credit=credit,recipe_validator=validate_recipe)
        bad=identity(g);bad['recipe']['research']['structured_advantage']='different-credit'
        with pytest.raises(ValueError):R.validate_boundary(state(g,1),rs[0],bad,credit=credit,recipe_validator=validate_recipe)
        summaries[0]['advantages'][0]+=.1
        with pytest.raises(ValueError):R.validate_ranks(summaries,credit=credit)
        # Same historical recipe's G8/G16 tensors are never interchangeable.
        with pytest.raises(ValueError):R.validate_boundary(state(g,1),rs[0],identity(24-g),credit=C.GroupCredit(24-g),recipe_validator=validate_recipe)


class FakeModule(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,*a):
        self.calls+=1
        return SimpleNamespace(actor_memory=torch.ones(1,192,2),actor_valid=torch.ones(1,192,dtype=torch.bool))


def test_conditions_single_prefill_collection_horizon8_and_dynamic_readouts():
    for g in (8,16):
        qwen=FakeModule();scene=FakeModule();predictive=FakeModule()
        data=SimpleNamespace(tokens=['s'],sha='data',observation=lambda t:dict(proprio=[0.,0.,0.,0.]))
        conditions=GroupMemoryConditions(SimpleNamespace(qwen=qwen),data,'IL70',
            SimpleNamespace(ref_scene_provider=scene,ref_predictive_provider=predictive),
            group_size=g,processor=lambda observations:observations)
        descriptor=conditions.get('s','cpu')
        first=conditions.materialize(descriptor,torch.device('cpu'))
        second=conditions.materialize(descriptor,torch.device('cpu'))
        assert first['context'].shape==(g,192,2) and first['ego'].shape==(g,4)
        assert qwen.calls==scene.calls==predictive.calls==1
        assert second['base_memory'].data_ptr()==first['base_memory'].data_ptr()
        recs=records(components(g)[0]); seen=[]
        class Actor:
            device=torch.device('cpu');record_sampling_terms=False
            _noise_audit=[];_probability_audit={}
            def sample_action_chain(self,cond,horizon,*,generator):
                assert horizon==8 and cond['repeats']==g
                self._noise_audit.append(dict(G=g))
                return torch.zeros(g,11,8,3),torch.arange(10),-torch.ones(10)/10,torch.tensor([7,8,9])
            def action_chain_logprobs(self,*a,**kw):
                self._probability_audit['current']=dict(source_clamped_logprob=torch.zeros(g,3,8,3))
                return torch.zeros(g,3)
        def score(args):
            assert args[2].shape==(g,8,3)
            return recs
        events=[]
        r=collect_group(Actor(),descriptor,'s','log',SimpleNamespace(score=score,protocol=dict(sha256='p')),
            torch.Generator().manual_seed(42),group_size=g,variant='R_CANONICAL',version=dict(update=0),
            progress=lambda *a:events.append(a))
        assert r['chain'].shape==(g,11,8,3) and r['rewards'].shape==(1,g)
        assert [(e[0],e[2]) for e in events]==[('generated',g),('scored',g)]
        assert r['group_size']==g and len(r['records'])==g
