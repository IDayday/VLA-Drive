"""P0 exact values/strides/gradients and real Tiny D actor, CPU only."""
import torch
import pytest
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import DPrefix
from starVLA.rl.d44_native_rl.fast_prefix import fast_index_select


def prefix(names=True):
    torch.manual_seed(47)
    kv=tuple((torch.randn(3,2,5,4,requires_grad=True),torch.randn(3,2,5,4,requires_grad=True))for _ in range(28))
    return DPrefix(kv,torch.ones(3,5,dtype=torch.bool),torch.arange(3),torch.randn(3,32,4,requires_grad=True),('a','b','c')if names else())


def values(p):return [x for pair in p.layer_kv for x in pair]+[p.valid,p.next_positions,p.reasoning]


def test_selection_values_strides_storage_and_gradients_exact():
    for indices in(torch.zeros(16,dtype=torch.long),torch.tensor([2]*16),torch.tensor([2,0,1,2]),torch.empty(0,dtype=torch.long),torch.tensor([-1,-1])):
        p=prefix();a=p.index_select(indices);b=fast_index_select(p,indices)
        assert a.scene_identity==b.scene_identity
        for x,y in zip(values(a),values(b)):
            assert torch.equal(x,y)and x.stride()==y.stride()and x.storage_offset()==y.storage_offset()
            assert x.untyped_storage().nbytes()==y.untyped_storage().nbytes()
        leaves=[x for x in values(p)if x.requires_grad]
        l1=sum((i+1)*v.square().sum()for i,v in enumerate(values(a))if v.requires_grad)
        l2=sum((i+1)*v.square().sum()for i,v in enumerate(values(b))if v.requires_grad)
        ga=torch.autograd.grad(l1,leaves,retain_graph=True);gb=torch.autograd.grad(l2,leaves)
        assert all(torch.equal(x,y)for x,y in zip(ga,gb))
    for names in(True,False):
        for indices in(torch.tensor(0),torch.tensor([[0,1]]),torch.tensor([0.]),torch.tensor([99]),torch.tensor([0,-1])):
            p=prefix(names);errors=[]
            for flag in(False,True):
                try:fast_index_select(p,indices)if flag else p.index_select(indices)
                except Exception as error:errors.append((type(error),str(error)))
            assert len(errors)==2 and errors[0]==errors[1]


def test_profiler_removes_repeated_scalar_extraction():
    p=prefix();indices=torch.zeros(16,dtype=torch.long);counts=[]
    for flag in(False,True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])as prof:fast_index_select(p,indices)if flag else p.index_select(indices)
        counts.append(sum(e.count for e in prof.key_averages()if e.key=='aten::_local_scalar_dense'))
    assert counts[0]>=118 and counts[1]==0
    print('Actual CPU aten::_local_scalar_dense calls baseline/hoisted:',counts)


def test_real_Tiny_FPO_actor_loss_all_LoRA_gradients_bitexact(monkeypatch):
    from tools.d46_world16.qualify_native import make_actor,condition,SyntheticService
    from starVLA.rl.d44_native_rl.config import defaults
    from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**kw:None)
    torch.manual_seed(42);torch.set_num_threads(1);cfg=defaults('D3_FPO');cfg['checkpoint_policy']='all_off'
    actor=make_actor(cfg)
    with torch.no_grad():
        for n,p in actor.named_parameters():
            if n.endswith('lora_B'):p.fill_(.001)
    scene=collect_scene(actor,condition(actor,0),'TRAIN_CPU','TRAIN_log',SyntheticService(),torch.Generator().manual_seed(42),cfg,0.,{'run':'CPU','update':0,'collection':1})
    scene.update(advantages=torch.linspace(-1,1,16),beta=.001,expect_fresh=True)
    params=[p for p in actor.parameters()if p.requires_grad];results=[]
    for flag in(False,True):
        actor.execution_fast_prefix=flag
        loss,metrics=actor(scene)
        grads=torch.autograd.grad(loss,params)
        results.append((loss.detach(),grads,{k:v.detach()for k,v in metrics.items()}))
    assert torch.equal(results[0][0],results[1][0])
    assert all(torch.equal(x,y)for x,y in zip(results[0][1],results[1][1]))
    assert all(torch.equal(results[0][2][k],results[1][2][k])for k in results[0][2])
