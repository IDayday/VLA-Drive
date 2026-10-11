import copy
import pytest
import torch
from iqe.performance.options import PerformanceOptions
from iqe.performance.master_optimizer import MasterAdamW
from iqe.performance.variants import load_adam


def test_strict_runtime_configuration():
    with pytest.raises(TypeError):PerformanceOptions(unknown=True)
    with pytest.raises(ValueError):PerformanceOptions(query_training_semantics='repeat7').validate()
    with pytest.raises(ValueError):PerformanceOptions(checkpoint_policy='ignored').validate()
    assert PerformanceOptions().validate().identity()['visual_precision']=='legacy_fp32'


def test_master_preserves_small_updates_and_exact_resume():
    p=torch.nn.Parameter(torch.ones(8,dtype=torch.bfloat16))
    opt=MasterAdamW([('w',p)],lr=1e-5,betas=(.9,.95))
    old=opt.masters[0].detach().clone()
    p.grad=torch.full_like(p,.1);opt.step()
    assert not torch.equal(old,opt.masters[0])
    assert torch.equal(p,opt.masters[0].bfloat16())
    assert opt.state[opt.masters[0]]['exp_avg'].dtype==torch.float32
    saved=copy.deepcopy(opt.state_dict())
    q=torch.nn.Parameter(torch.zeros_like(p));other=MasterAdamW([('w',q)],lr=1e-5,betas=(.9,.95))
    other.load_state_dict(saved)
    for o,v in [(opt,p),(other,q)]:v.grad=torch.full_like(v,.2);o.step()
    assert torch.equal(opt.masters[0],other.masters[0])
    assert torch.equal(p,q)
    assert torch.equal(opt.state[opt.masters[0]]['exp_avg'],other.state[other.masters[0]]['exp_avg'])
    bad=MasterAdamW([('wrong',torch.nn.Parameter(torch.zeros_like(p)))],lr=1e-5)
    with pytest.raises(ValueError,match='name/order'):bad.load_state_dict(saved)


def test_master_migration_uses_original_fp32_not_rounded_execution_weight():
    fp32=torch.tensor([1.00013,2.00017])
    p=torch.nn.Parameter(fp32.bfloat16());opt=MasterAdamW([('w',p)],lr=1e-5)
    opt.initialize_masters({'w':fp32})
    assert torch.equal(opt.masters[0],fp32)
    assert not torch.equal(opt.masters[0],p.float())


def test_optimizer_implementation_flags_survive_legacy_state_load():
    p=torch.nn.Parameter(torch.ones(3));old=torch.optim.AdamW([p],lr=.001)
    p.grad=torch.ones_like(p);old.step()
    new=torch.optim.AdamW([torch.nn.Parameter(torch.ones(3))],lr=.001)
    load_adam(new,copy.deepcopy(old.state_dict()),PerformanceOptions(optimizer='fused'))
    assert new.param_groups[0]['fused'] is True
    assert new.param_groups[0]['lr']==old.param_groups[0]['lr']


def test_frozen_payload_complete_slots_and_left_padding():
    from iqe.performance.vision_cache import FrozenVisionPayload,combine_payloads
    def payload(length):
        return FrozenVisionPayload(torch.arange(length),torch.arange(length).repeat(3,1),
            {'history':torch.tensor([0]),'foresight':torch.tensor([1]),'action':torch.tensor([length-1])},
            torch.ones(3,8), (torch.ones(3,8)*2,torch.ones(3,8)*3,torch.ones(3,8)*4),torch.tensor([[1,2,2]]*3))
    names=('history','foresight','action')
    a,b=payload(4),payload(6);ids,mask,pos,slots,im,deep=combine_payloads([a,b],99,'cpu',names)
    assert ids[0,:2].tolist()==[99,99]
    assert mask.sum(1).tolist()==[4,6]
    assert slots['foresight'].tolist()==[[3],[1]]
    assert slots['action'].tolist()==[[5],[5]]
    assert torch.equal(pos[:,0,2:],a.position_ids)
    assert im.shape==(6,8) and len(deep)==3
    a.slots.pop('foresight')
    with pytest.raises(ValueError,match='foresight'):a.validate(names)


def test_visual_observation_identity_rejects_gt_and_tracks_order():
    from PIL import Image
    from iqe.performance.vision_cache import observation_key
    obs={'image':[Image.new('RGB',(2,2),c) for c in ['red','green','blue']],'state':[0],'lang':'go'}
    identity={'precision':'fp32'}
    key=observation_key(obs,'go',identity)
    assert key!=observation_key(dict(obs,image=list(reversed(obs['image']))),'go',identity)
    assert key!=observation_key(obs,'stop',identity)
    assert key!=observation_key(obs,'go',{'precision':'bf16'})
    with pytest.raises(ValueError,match='whitelist'):observation_key(dict(obs,gt=1),'go',identity)


def _small_query(dropout=0.):
    from iqe.query_base import QueryActionBridge
    from omegaconf import OmegaConf
    config=OmegaConf.create({'framework':{'action_model':{'action_horizon':8},'qwenvl':{'vl_hidden_dim':16}}})
    options={'width':16,'layers':2,'ffn':32,'dt':.5,'scene_tokens':3,'compressor_layers':1,'compressor_heads':2,
             'heads':2,'dropout':dropout,'drop_path':dropout,'prev_weight':.4}
    return QueryActionBridge(config,options)


def test_actual_query_repeat_mean_loss_and_gradient_without_stochastic_layers():
    torch.manual_seed(5);one=_small_query();eight=copy.deepcopy(one)
    memory=torch.randn(3,8,16);target=torch.randn(3,8,4);ego=torch.randn(3,1,4)
    for m in (one,eight):m.ego_state=ego;m.sample_valid=torch.tensor([True,False,True]);m.eval()
    a=one(memory,target);b=eight(memory.repeat(8,1,1),target.repeat(8,1,1))
    a.backward();b.backward()
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    for (_,p),(_,q) in zip(one.named_parameters(),eight.named_parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)


def test_stochastic_query_repeat_is_not_a_bitwise_training_equivalence():
    torch.manual_seed(5);one=_small_query(.2);eight=copy.deepcopy(one)
    memory=torch.randn(3,8,16);target=torch.randn(3,8,4);ego=torch.randn(3,1,4)
    for m in (one,eight):m.ego_state=ego;m.sample_valid=torch.ones(3,dtype=torch.bool);m.train()
    rng=torch.get_rng_state();a=one(memory,target);after_one=torch.get_rng_state()
    torch.set_rng_state(rng);b=eight(memory.repeat(8,1,1),target.repeat(8,1,1))
    assert not torch.equal(a,b)
    # Different CUDA/CPU RNG engines may advance by call rather than element;
    # no cross-semantics RNG equality or inequality is required.
    assert torch.isfinite(a) and torch.isfinite(b)


def test_empty_query_mask_keeps_graph_and_all_finite_gradients():
    model=_small_query();model.ego_state=torch.zeros(2,1,4);model.sample_valid=torch.zeros(2,dtype=torch.bool)
    loss=model(torch.randn(2,8,16),torch.zeros(2,8,4));loss.backward()
    assert loss.item()==0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.expert.query.weight.grad is not None


def test_fingerprint_includes_master_and_scalar_optimizer_step():
    from iqe.performance.run import state_hash
    model=torch.nn.Linear(3,2).to(torch.bfloat16)
    opt=MasterAdamW(list(model.named_parameters()),lr=1e-5)
    for p in model.parameters():p.grad=torch.ones_like(p)
    opt.step();before=state_hash(model,opt)
    with torch.no_grad():opt.masters[0].view(-1)[0]+=1e-6
    after=state_hash(model,opt)
    assert before['model']==after['model']
    assert before['optimizer_tensors']!=after['optimizer_tensors']


def test_continuation_rejects_missing_or_stale_evidence(tmp_path):
    from iqe.performance.qualification import validate_continuation
    from iqe.io import atomic_json
    cert=tmp_path/'certificate.json'
    atomic_json(cert,{'status':'FAIL'})
    with pytest.raises(ValueError,match='passed'):validate_continuation(cert,PerformanceOptions(),{})
    atomic_json(cert,{'status':'PASS','options':PerformanceOptions().identity(),'implementation_hashes':{'old':'x'}})
    with pytest.raises(ValueError,match='implementation'):validate_continuation(cert,PerformanceOptions(),{})


def test_all_deepstack_outputs_required():
    from iqe.performance.vision_cache import FrozenVisionPayload
    p=FrozenVisionPayload(torch.arange(4),torch.arange(4).repeat(3,1),{'foresight':torch.tensor([1])},torch.ones(3,8),(),torch.ones(3,3,dtype=torch.long))
    with pytest.raises(ValueError,match='three DeepStack'):p.validate(['foresight'])
