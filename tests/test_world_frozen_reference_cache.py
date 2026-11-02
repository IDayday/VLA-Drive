"""Execution-only reference reuse: directed CPU math/native tests, no GPU score."""
import copy
import io

import pytest
import torch

from test_world_performance_objectives import TinyActor, rollout
from test_qwen_world_lastdet_contract import recipe
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_research_algorithms import frozen_reference as C
from starVLA.rl.qwen_world_performance.awfm import AWFMLoss,attach_awfm_material
from starVLA.rl.qwen_world_performance.fpo_adapter import FPOAdapterLoss,attach_fpo_material


class Fixture(TinyActor):
    def __init__(self):
        super().__init__();self.conditions=None;self.reference_forwards=0;self.current_forwards=0
        self.ref_action_expert=torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False)
        with torch.no_grad():self.ref_action_expert.weight.fill_(.1)

    def action_chain_logprobs(self,*args,use_ref=False,**kwargs):
        self.reference_forwards+=int(use_ref);self.current_forwards+=int(not use_ref)
        return super().action_chain_logprobs(*args,use_ref=use_ref,**kwargs)


def identity(cfg,out):
    return dict(sha256=out['behavior_version']['run'],recipe=cfg,IL_parent=dict(model_sha256='IL70_fixture'),
        projected_condition_identity=out['cond_bg']['identity'],source_files={'original_math':'unchanged_fixture'})


def assert_equal(left,right):
    assert C.digest(left)==C.digest(right)


@pytest.mark.parametrize('loss_type,attach',[(ResearchLoss,None),(AWFMLoss,attach_awfm_material),(FPOAdapterLoss,attach_fpo_material)])
@pytest.mark.parametrize('kl_type',['x','x_kl','v'])
def test_four_inner_values_gradients_Adam_RNG_bitexact(loss_type,attach,kl_type):
    torch.manual_seed(42);cfg=recipe(False);cfg['grpo']['sample']['group_size']=16
    left=Fixture();right=copy.deepcopy(left);left.grpo_kl_type=right.grpo_kl_type=kl_type
    out=rollout(16)
    if attach:attach(left,out,rank=0)
    a,b=copy.deepcopy(out),copy.deepcopy(out);left._loss=loss_type(cfg);right._loss=loss_type(cfg)
    C.enable_reference_cache(right,identity(cfg,b))
    oa=torch.optim.AdamW([p for p in left.parameters() if p.requires_grad],lr=1e-5)
    ob=torch.optim.AdamW([p for p in right.parameters() if p.requires_grad],lr=1e-5)
    immutable=C.digest({k:out[k] for k in ('old_logp','advantages','adv_mask','chain')})
    rng=torch.get_rng_state().clone()
    for inner in range(4):
        left._performance_fresh_inner=right._performance_fresh_inner=inner
        x,mx=left._loss._policy_loss(left,a);y,my=right._loss._policy_loss(right,b)
        assert torch.equal(x,y);assert_equal(mx,my)
        x.backward();y.backward()
        for (n,p),(m,q) in zip(left.named_parameters(),right.named_parameters()):
            assert n==m
            if p.requires_grad:assert torch.equal(p.grad,q.grad)
            else:assert p.grad is None and q.grad is None
        oa.step();ob.step();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        assert_equal(left.state_dict(),right.state_dict());assert_equal(oa.state_dict(),ob.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
        assert C.digest({k:b[k] for k in ('old_logp','advantages','adv_mask','chain')})==immutable
    assert left.reference_forwards==4 and right.reference_forwards==1
    assert left.current_forwards==right.current_forwards==4


def test_pending_roundtrip_parent_ancestry_seal_and_detached_only():
    cfg=recipe(False);actor=Fixture();out=rollout();run=identity(cfg,out);C.enable_reference_cache(actor,run)
    before=C.digest(out);C.reference_terms(actor,out);assert actor.reference_forwards==1
    assert C.digest({k:v for k,v in out.items() if k!=C.KEY})==before
    data=io.BytesIO();torch.save(out,data);data.seek(0);restored=torch.load(data,weights_only=False)
    C.validate_reference_cache(restored,run)
    for k,v in restored[C.KEY]['terms'].items():assert not v.requires_grad and v.grad_fn is None
    C.reference_terms(actor,restored);assert actor.reference_forwards==1
    for key in ['chain','timesteps','deltas','stoch_idx']:
        bad=copy.deepcopy(restored);bad[key].reshape(-1)[0]+=1
        with pytest.raises(ValueError,match='ancestry'):C.validate_reference_cache(bad,run)
    for changed in [dict(run,sha256='different_run'),dict(run,IL_parent=dict(model_sha256='other_parent')),
                    dict(run,recipe=dict(cfg,action_infer_shift=6)),dict(run,source_files={'math':'different'})]:
        with pytest.raises(ValueError,match='ancestry|different run'):C.validate_reference_cache(restored,changed)
    bad=copy.deepcopy(restored);bad['cond_bg']['identity']='wrong'
    with pytest.raises(ValueError,match='condition'):C.validate_reference_cache(bad,run)
    bad=copy.deepcopy(restored);bad[C.KEY]['terms']['mean'][0,0,0,0]+=.1
    with pytest.raises(ValueError,match='seal'):C.validate_reference_cache(bad,run)
    with torch.no_grad():actor.ref_action_expert.weight.add_(.01)
    with pytest.raises(ValueError,match='frozen IL reference changed'):C.reference_terms(actor,restored)
    legacy=rollout();other=Fixture();C.enable_reference_cache(other,identity(cfg,legacy))
    assert C.validate_reference_cache(legacy,identity(cfg,legacy)) is None
    C.reference_terms(other,legacy);assert other.reference_forwards==1  # old pending constants built once


def test_actual_Scope_BF16_reference_reuse_keeps_current_memory_graph_each_inner():
    from test_world_performance_nominal import actual,primary,Service
    torch.set_num_threads(1)
    left,lc,cfg=actual();right,rc,_=actual();out=primary(left,lc,Service());a,b=copy.deepcopy(out),copy.deepcopy(out)
    C.enable_reference_cache(right,identity(cfg,b));rng=torch.get_rng_state().clone()
    before_left=left._reference_calls;before_right=right._reference_calls
    mb_left=left.current_memory_builds;mb_right=right.current_memory_builds
    oa=torch.optim.AdamW([p for p in left.parameters() if p.requires_grad],lr=1e-5)
    ob=torch.optim.AdamW([p for p in right.parameters() if p.requires_grad],lr=1e-5)
    for inner in range(4):
        x,mx=left(a);y,my=right(b);assert torch.equal(x,y);assert_equal(mx,my)
        x.backward();y.backward()
        for (n,p),(m,q) in zip(left.named_parameters(),right.named_parameters()):
            assert n==m
            if p.requires_grad:assert torch.equal(p.grad,q.grad)
        oa.step();ob.step();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        assert_equal(oa.state_dict(),ob.state_dict())
    assert left.current_memory_builds-mb_left==right.current_memory_builds-mb_right==4
    assert left._reference_calls-before_left==12 and right._reference_calls-before_right==3
    assert torch.equal(rng,torch.get_rng_state())


def test_cache_protocol_is_execution_only_not_scientific_identity():
    from starVLA.rl.simwam_recipe.lineage import scientific_identity,SCIENTIFIC_KEYS
    base={k:None for k in SCIENTIFIC_KEYS};base['recipe']=recipe(False)
    modified=copy.deepcopy(base);modified['execution_optimizations']={'frozen_IL_reference_cache':copy.deepcopy(C.PROTOCOL)}
    assert scientific_identity(base)==scientific_identity(modified)
