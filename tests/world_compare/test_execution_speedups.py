import copy
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import pytest
import torch

from starVLA.model.modules.qwen_world_compare.execution import short_stream_checkpointing
from starVLA.model.modules.qwen_world_compare.independent_actor import ArchitectureA,ArchitectureC
from starVLA.model.modules.qwen_world_compare.shared_actor import ArchitectureB
from starVLA.rl.qwen_world_compare.prefetch import OrderedUpdatePrefetch


@pytest.mark.parametrize('architecture',list('ABCD'))
def test_short_checkpoint_policy_preserves_outputs_input_and_all_parameter_grads_and_adam(architecture):
    torch.set_num_threads(2);torch.manual_seed(42)
    if architecture=='D':
        spec=importlib.util.spec_from_file_location('tiny_D_oracle',Path(__file__).with_name('test_d_qwen_flow_adapter.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        reference=module.model(checkpointing=True);inputs=module.batch(vision=True,padded=True)
    elif architecture=='B':
        reference=ArchitectureB(hidden_dim=24,num_layers=4,num_heads=4,ffn_dim=72,activation_checkpointing=True)
    else:
        reference=({'A':ArchitectureA,'C':ArchitectureC}[architecture])(
            actor_config=dict(hidden_dim=24,num_layers=4,num_heads=4,ffn_dim=72,activation_checkpointing=True),
            world_config=dict(hidden_dim=12,num_layers=4,num_heads=3,ffn_dim=36,bridge_groups=2,activation_checkpointing=True))
    with torch.no_grad():
        for name,p in reference.named_parameters():
            if 'modulation'in name:p.add_(torch.randn_like(p)*.01)
    candidate=copy.deepcopy(reference)
    short_stream_checkpointing(SimpleNamespace(architecture_id=architecture,decision=candidate),False)
    state=torch.randn(2,8,3);sigma=torch.tensor([.23,.81]);scene=torch.randn(2,160,24);ego=torch.randn(2,8)
    valid=torch.ones(2,160,dtype=torch.bool);valid[1,-4:]=False
    def forward(model):
        if architecture=='D':prefix=model.prepare_prefix(model.encode_current(inputs))
        else:prefix=model.prepare_prefix(scene,valid,ego)
        pred=model.denoise_x0(state,sigma,prefix)
        post=model.replay_candidates(state,torch.tensor([0,1]),prefix)
        pre=model.pre_reasoning_states(prefix)
        return pred,pre,post
    rng=torch.get_rng_state().clone();left=forward(reference);sum(x.square().mean()for x in left).backward()
    assert torch.equal(rng,torch.get_rng_state())
    right=forward(candidate);sum(x.square().mean()for x in right).backward()
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    for (name,a),(other,b)in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name==other and a.dtype==b.dtype==torch.float32
        assert(a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=5e-5)
    opts=[torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)for m in(reference,candidate)]
    for opt in opts:opt.step()
    for a,b in zip(reference.parameters(),candidate.parameters()):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    assert set(reference.state_dict())==set(candidate.state_dict())


class Builder:
    def __init__(self):self.calls=0
    def build(self,tokens,positions,update,*,account=True):
        if tokens==['bad']:raise ValueError('bad data')
        if account:self.calls+=1
        return dict(tokens=tokens,noise=torch.randn(len(tokens),8,3,generator=torch.Generator().manual_seed(update*100+positions[0])))


def request(update,cursor):return [([str(cursor+i)],[i])for i in range(2)]


def test_prefetch_same_draws_calls_order_and_resume_without_global_rng_changes():
    torch.manual_seed(9);rng=torch.get_rng_state().clone();builder=Builder();reference=Builder()
    with OrderedUpdatePrefetch(builder,request,start_update=10,start_cursor=320)as pool:
        for update in(10,11):
            actual=pool.next(update,update*32)
            expected=[reference.build(t,p,update)for t,p in request(update,update*32)]
            for a,b in zip(actual,expected):assert a['tokens']==b['tokens']and torch.equal(a['noise'],b['noise'])
        assert builder.calls==reference.calls==4
    assert builder.calls==4 and torch.equal(rng,torch.get_rng_state())
    with OrderedUpdatePrefetch(builder,request,start_update=12,start_cursor=384)as pool:
        actual=pool.next(12,384);expected=[reference.build(t,p,12)for t,p in request(12,384)]
        assert all(torch.equal(a['noise'],b['noise'])for a,b in zip(actual,expected))
    assert builder.calls==reference.calls==6


def test_prefetch_exception_and_misordered_cursor_are_not_silently_skipped():
    builder=Builder()
    with OrderedUpdatePrefetch(builder,lambda u,c:[(['bad'],[0])],start_update=1,start_cursor=32)as pool:
        with pytest.raises(ValueError,match='cursor'):pool.next(2,64)
        with pytest.raises(ValueError,match='bad data'):pool.next(1,32)
    assert builder.calls==0


def test_database_reader_is_created_and_closed_on_worker_not_shared():
    import sqlite3
    import threading
    class SQLBuilder(Builder):
        def __init__(self):
            super().__init__();self.bank=SimpleNamespace(db=sqlite3.connect(':memory:'))
            self.bank.db.execute('create table t(v)');self.bank.db.execute('insert into t values(7)')
            self.thread=threading.get_ident();self.closed=False
        def state_dict(self):return dict(calls=self.calls,bank_hash='fixed')
        def build(self,*args,**kwargs):
            assert self.bank.db.execute('select v from t').fetchone()==(7,)
            return super().build(*args,**kwargs)
        def close(self):
            assert self.thread==threading.get_ident()
            self.bank.db.close();self.closed=True
    main=SQLBuilder();made=[]
    def factory():
        value=SQLBuilder();made.append(value);return value
    with pytest.raises(ValueError,match='worker-local'):
        OrderedUpdatePrefetch(main,request,start_update=1,start_cursor=32)
    with OrderedUpdatePrefetch(main,request,start_update=1,start_cursor=32,builder_factory=factory)as pool:
        assert len(pool.next(1,32))==2
    assert main.calls==2 and made[0].closed and made[0].thread!=threading.get_ident()
    assert main.bank.db.execute('select v from t').fetchone()==(7,)
    main.close()
