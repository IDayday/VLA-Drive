"""Independent mathematical fixtures, no native models/data/GPU execution."""
import math
from copy import deepcopy
import torch
from tools.st_world_policy import audit_flow_gradient_balance as audit
from starVLA.model.modules.flow_posttraining import FPOConfig, fpo_surrogate, group_centered_advantages


def test_rank_microbatch_fpo_gradient_matches_one_global_candidate_objective():
    # 32 scenes and 4 candidates; rank ownership and micro slices are the actual
    # registration shape, without a model or stochastic dropout comparison.
    g=torch.Generator().manual_seed(811)
    rewards=torch.rand((32,4),generator=g,dtype=torch.float64)
    old=torch.rand((32,4,50),generator=g,dtype=torch.float64)
    slopes=torch.randn((32,4,50),generator=g,dtype=torch.float64)
    advantages=group_centered_advantages(rewards,torch.ones_like(rewards,dtype=torch.bool))
    config=FPOConfig()
    parameter=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    full,_=fpo_surrogate(old.flatten(0,1), (old+parameter*slopes).flatten(0,1),
        advantages.flatten(),torch.ones(128,dtype=torch.bool),config,global_count=128,world_size=1)
    expected,=torch.autograd.grad(full,parameter)
    summed=torch.zeros_like(expected)
    for rank in range(4):
        ids=torch.arange(32)[rank::4]
        for micro in ids.split(4):
            parameter=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
            value,_=fpo_surrogate(old[micro].flatten(0,1),
                (old[micro]+parameter*slopes[micro]).flatten(0,1),advantages[micro].flatten(),
                torch.ones(16,dtype=torch.bool),config,global_count=128,world_size=1)
            grad,=torch.autograd.grad(value,parameter)
            summed+=grad
    torch.testing.assert_close(summed,expected,rtol=1e-14,atol=1e-14)


def test_statistics_match_explicit_combined_vector_and_directional_derivative():
    b=torch.tensor([1.,-2.,3.,-4.],dtype=torch.float64)
    p=torch.tensor([-2.,3.,-4.,5.],dtype=torch.float64)
    weight=.7
    report=audit.derived_statistics({'bc_squared_l2':float(b.square().sum()),
        'policy_squared_l2':float(p.square().sum()),'dot':float(b@p),'elements':4},weight)
    combined=b+weight*p
    assert math.isclose(report['combined_gradient_l2'],float(combined.norm()),rel_tol=1e-14)
    assert math.isclose(report['first_order_BC_change_under_combined_descent_per_unit_lr'],float(-b@combined),rel_tol=1e-14)
    assert math.isclose(report['first_order_policy_change_under_combined_descent_per_unit_lr'],float(-p@combined),rel_tol=1e-14)


def test_bf16_gradient_storage_is_widened_not_relabelled_fp32_backward():
    model=torch.nn.Linear(2,1,bias=False,dtype=torch.bfloat16)
    acc=audit.GradientSum(model)
    x=torch.tensor([[.003,.007]],dtype=torch.bfloat16)
    for kind,scale in [('bc',1.),('policy',-.3)]:
        model.zero_grad(set_to_none=True)
        (model(x).float().sum()*scale).backward()
        expected=model.weight.grad.float().clone()
        acc.add(kind,model)
        assert torch.equal(acc.sums[kind]['weight'],expected)
        assert acc.sums[kind]['weight'].dtype==torch.float32
    result=acc.report()
    assert result['execution_gradient_dtypes']==['torch.bfloat16']
    assert result['exact_ZeRO2_update_claim'] is False


def test_epoch_boundary_keeps_native_next_order():
    manifest={'entries':[{'scene_id':str(i)} for i in range(64)]}
    execution={'world_size':4,'global_batch':32,'micro_batch':4,'seed':42}
    ranks,cursor=audit.next_rank_entries(manifest,{'epoch':0,'offset':2},execution)
    direct,direct_cursor=audit.replay.next_scene_entries(manifest,{'epoch':1,'offset':0},{**execution,'world_size':1})
    assert cursor==direct_cursor
    assert [x for i in range(8) for x in [ranks[r][i] for r in range(4)]]==direct


def test_no_cuda_initialized():
    assert not torch.cuda.is_initialized()
