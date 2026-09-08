import copy
import inspect
import textwrap
from types import SimpleNamespace as S
import numpy as np
import pytest
import torch
from navsim.agents.EpisodeDrive.planreg_v2.motion import CandidateKinematicsCodec,IntervalMotionEncoder,interval_selection
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics,TrajectoryNormalizer,unwrap_heading
from navsim.agents.EpisodeDrive.planreg_v2.initialization import shared_artifact,load_shared_bank
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import resolve_learning_rates,multiplier
from navsim.agents.EpisodeDrive.planreg_v2.predictor import ActionCausalPredictor
from navsim.agents.EpisodeDrive.planreg_v2.runtime import component_gradient_audit,validate_ttc_reduction
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent
from navsim.agents.EpisodeDrive.planreg_v2.targets import long_target


@pytest.mark.parametrize('field,value',[('time',0.),('time',-.5),('time',.5),('time',float('nan')),
                                     ('pose',float('nan')),('pose',float('inf'))])
def test_valid_bad_motion_is_not_silently_padding(field,value):
    poses=torch.zeros(2,3,8,3);times=torch.arange(1,9).expand(2,3,8).float()*.5
    if field=='time':times[1,2,1]=value
    else:poses[1,2,1,0]=value
    with pytest.raises(ValueError,match='batch/candidate/time'):
        CandidateKinematicsCodec()(poses,torch.zeros(2,4),times,torch.ones(2,3,8,dtype=torch.bool))


def test_padding_nan_motion_outputs_and_gradients_finite():
    p=torch.randn(1,2,8,3);p[:,:,3:]=float('nan');p.requires_grad_()
    t=torch.arange(1,9).float()*.5;t[3:]=float('nan')
    mask=torch.arange(8)<3
    result=CandidateKinematicsCodec()(p,torch.zeros(1,4),t,mask)
    model=IntervalMotionEncoder(16)
    outputs,valid=model(result['motion_sequence'],result['timestamps'],result['valid_mask'],[.5,1.5,4.])
    outputs.square().sum().backward()
    assert torch.isfinite(outputs).all() and torch.isfinite(p.grad).all()
    assert valid[0,0].tolist()==[True,True,False]
    assert p.grad[:,:,3:].count_nonzero()==0


def test_masked_unwrap_statistics_and_explicit_modes(tmp_path):
    x=torch.randn(8,3);mask=torch.ones(8,dtype=torch.bool);mask[3]=False;x[3]=float('nan')
    x[:,2]=torch.tensor([3.,3.1,-3.1,float('nan'),-2.9,-2.8,-2.7,-2.6])
    n=measured_statistics([('a',x,mask),('b',torch.zeros(8,3),torch.ones(8,dtype=torch.bool))],'train','unit')
    z=n.normalize(x,valid_mask=mask)
    assert torch.isfinite(z).all()
    torch.testing.assert_close(n.inverse(z,valid_mask=mask)[mask],x[mask])
    global_n=measured_statistics([('a',x,mask),('b',torch.zeros(8,3),torch.ones(8,dtype=torch.bool))],'train','unit',mode='global_zscore')
    assert torch.equal(global_n.std[0],global_n.std[7])
    fixed=TrajectoryNormalizer(torch.zeros(3),torch.tensor([30.,10.,1.]),n.metadata,mode='legacy_fixed_affine')
    fixed.save(tmp_path/'fixed.json')
    assert TrajectoryNormalizer.load(tmp_path/'fixed.json').metadata['mode']=='legacy_fixed_affine'
    assert n.metadata['raw_gt_sha256'] and n.metadata['valid_mask']==[True]*8


@pytest.mark.parametrize('failure',['short','backward','nonfinite','other_log','late'])
def test_long_invalid_preserves_scene(failure):
    times=torch.arange(11).float()*.5;poses=torch.zeros(11,3);valid=torch.ones(11,dtype=torch.bool)
    logs=['drive_a']*11
    if failure=='short':poses,times,valid=poses[:10],times[:10],valid[:10]
    if failure=='backward':times[4]=times[3]
    if failure=='nonfinite':poses[2,0]=float('nan')
    if failure=='other_log':logs[-1]='drive_b'
    if failure=='late':times[-1]=5.1
    output,ok=long_target(poses,times,valid,log_ids=logs)
    assert not ok and output.shape==(8,3)


@pytest.mark.parametrize('gb',[1,32,128])
def test_all_lr_groups_caps_language_exception_and_schedule(gb):
    rates=resolve_learning_rates(gb)
    assert rates['language_lora']['resolved_peak']==1e-5
    for name,row in rates.items():
        expected=1e-5 if name=='language_lora' else min(row['reference_lr']*(gb/32)**.5,row['cap'])
        assert row['resolved_peak']==expected
        assert expected*multiplier(0,21789)==expected*.01
        assert expected*multiplier(round(21789*.05),21789)==expected
        assert expected*multiplier(21788,21789)==expected*.1
    assert resolve_learning_rates(2*4*16)==resolve_learning_rates(8*2*8)


class Bank(torch.nn.Module):
    def __init__(self,wm=True):
        super().__init__();self.config={'register_init_std':.02};self.world_model_enabled=wm
        self.scene_memory=torch.nn.Linear(3,4)
        self.wm_predictor=torch.nn.Linear(3,4) if wm else None
    def trainable_state(self):return {n:p.detach().clone() for n,p in self.named_parameters()}


def test_shared_named_bank_controls_stale_identity_and_no_reinitialization():
    main=Bank();artifact=shared_artifact(main,0)
    control=Bank(False);load_shared_bank(control,artifact)
    for name,p in control.named_parameters():assert torch.equal(p,artifact['trainable_state'][name])
    old=copy.deepcopy(artifact);old['identity']['register_init_std']=1e-6
    with pytest.raises(ValueError,match='Stale'):load_shared_bank(control,old)
    old=copy.deepcopy(artifact);old['schema']='planreg_v2_shared_trainable_v1'
    with pytest.raises(ValueError,match='Stale'):load_shared_bank(control,old)


def test_relabeling_manifest_cannot_authorize_uniform_long_cache(tmp_path):
    import json
    from navsim.agents.EpisodeDrive.planreg_v2.data import InputOnlyV2Dataset
    from navsim.agents.EpisodeDrive.planreg_v2 import CACHE_SCHEMA,LONG_TARGET_VERSION
    item=tmp_path/'item.pt'
    torch.save(dict(schema=CACHE_SCHEMA,long_target_version=LONG_TARGET_VERSION,features={},
                    targets=dict(long_target_version='uniform_linear_v1')),item)
    manifest=dict(schema=CACHE_SCHEMA,long_target_version=LONG_TARGET_VERSION,split='train',records=[dict(token='unit',cache_path=str(item))])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    dataset=InputOnlyV2Dataset(path,'/unused/vlm')
    with pytest.raises(ValueError,match='Old uniform-long'):dataset[0]


def test_future_only_teacher_batch_matches_removed_current_encoding():
    class Teacher(torch.nn.Module):
        def forward(self,pixels):return pixels.mean((-2,-1))[:,None,:].repeat(1,16,86)[...,:256]
    obj=S(ema_teacher=Teacher())
    features=dict(pixel_values=[torch.randn(2,3,4,4),torch.randn(3,3,4,4)],
                  future_pixel_values=[[torch.randn(2,3,4,4) for _ in range(3)],
                                       [torch.randn(3,3,4,4) for _ in range(3)]])
    predictions={'visual_content':torch.zeros(2,48,256)}
    old=PlanRegV2Agent.encode_teacher(obj,features,predictions,include_current=True)
    new=PlanRegV2Agent.encode_teacher(obj,features,predictions)
    assert torch.equal(old[:,1:],new)


def assert_predictor_semantics(model):
    torch.manual_seed(55)
    z=torch.randn(2,16,32,requires_grad=True);target=torch.randn(2,3,16,32,requires_grad=True)
    actions=torch.randn(2,3,32,requires_grad=True);geometry=torch.randn(2,16,5)
    valid=torch.ones(2,16,dtype=torch.bool);semantic=torch.randn(2,16,32,requires_grad=True)
    tf,ro,steps=model.branches(z,target,actions,geometry,valid,semantic)
    for p in steps:p.retain_grad()
    changed=actions.detach().clone();changed[:,2]*=-10
    tf2,ro2,_=model.branches(z,target,changed,geometry,valid,semantic)
    torch.testing.assert_close(tf[:,:2],tf2[:,:2],atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(ro[:,:2],ro2[:,:2],atol=2e-6,rtol=1e-5)
    replacement=torch.randn_like(target)*torch.linspace(.1,3,32)
    _,other,_=model.branches(z,replacement,actions,geometry,valid,semantic)
    torch.testing.assert_close(ro,other,atol=0,rtol=0)
    (ro[:,-1]-torch.randn_like(ro[:,-1])).abs().mean().backward()
    assert z.grad.norm()>0 and all(p.grad is not None and p.grad.norm()>0 for p in steps[:2])
    assert target.grad is None


def test_tf_ro_semantics_and_separate_future_dependency():
    assert_predictor_semantics(ActionCausalPredictor(dim=32,layers=2,heads=4,ffn=64))


@pytest.mark.parametrize('mutation',['detach','teacher','noncausal'])
def test_predictor_reversion_mutations_are_detected(mutation,monkeypatch):
    model=ActionCausalPredictor(dim=32,layers=2,heads=4,ffn=64)
    function=type(model).forward if mutation=='noncausal' else type(model).branches
    source=textwrap.dedent(inspect.getsource(function))
    before,after={'detach':('states.append(prediction)','states.append(prediction.detach())'),
        'teacher':('states.append(prediction)','states.append(targets[:,i].detach())'),
        'noncausal':('causal_mask = block_ids[None,:] > block_ids[:,None]',
                     'causal_mask = torch.zeros_like(block_ids[None,:] > block_ids[:,None])')}[mutation]
    assert before in source
    namespace=dict(function.__globals__);exec(compile(source.replace(before,after),'<intentional production mutation>','exec'),namespace)
    monkeypatch.setattr(type(model),function.__name__,namespace[function.__name__])
    with pytest.raises(AssertionError):assert_predictor_semantics(model)


def test_component_gradients_are_separate_and_read_only():
    p=torch.nn.Parameter(torch.randn(4));p.grad=torch.ones(4);before=torch.get_rng_state().clone()
    losses=dict(trajectory_loss=p.square().sum(),scorer_loss=(p-1).square().sum(),wm_loss=(p+2).square().sum(),wm_weight=torch.tensor(.1))
    report=component_gradient_audit(losses,[('backbone.model.vision_model.q_lora_a.weight',p)])
    assert set(report['vision_lora']['norms'])=={'trajectory','scorer','weighted_wm'}
    assert torch.equal(p.grad,torch.ones(4)) and torch.equal(before,torch.get_rng_state())


def test_ttc_formal_all_valid_guard_is_conditional():
    scores=torch.zeros(2,64,7);scores[1,:,3]=1
    validate_ttc_reduction(scores,8)
    scores[0,0,3]=2
    validate_ttc_reduction(scores,1)  # original single-batch core supports its sentinel mask
    with pytest.raises(ValueError,match='synchronized rejection'):validate_ttc_reduction(scores,2)


def test_unequal_ttc_valid_count_full_batch_reference():
    from navsim.agents.EpisodeDrive.layers.losses.episode_drive_loss import EpisodeDriveLoss
    from navsim.agents.EpisodeDrive.score_module.scorer import DRIVOR_SCORE_HEAD_NAMES
    logits={n:torch.zeros(2,4,dtype=torch.float64) for n in DRIVOR_SCORE_HEAD_NAMES}
    logits['time_to_collision_within_bound']=torch.tensor([[3.,1.,2.,-1.],[-2.,-1.,0.,2.]],dtype=torch.float64,requires_grad=True)
    scores=torch.zeros(2,4,7,dtype=torch.float64);scores[...,3]=torch.tensor([[1.,2.,2.,2.],[1.,0.,1.,0.]])
    loss=EpisodeDriveLoss().score_loss(logits,None,None,None,scores.clone(),None,None,None,None)[0][1]
    valid=scores[...,3]!=2
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits['time_to_collision_within_bound'][valid],scores[...,3][valid])
    torch.testing.assert_close(loss,expected,atol=0,rtol=0)
    means=[]
    for i in range(2):
        means.append(EpisodeDriveLoss().score_loss({n:v[i:i+1] for n,v in logits.items()},None,None,None,scores[i:i+1].clone(),None,None,None,None)[0][1])
    assert abs(float((means[0]+means[1])/2-loss))>1e-3
    with pytest.raises(ValueError):validate_ttc_reduction(scores,2)


@pytest.mark.parametrize('mutation',['ego_before','uniform_long','linear_long'])
def test_r1_r2_production_reversions_are_detected(mutation,monkeypatch):
    from scripts.audit_planreg_v2_review import scorer_integration,long_parity
    from navsim.agents.EpisodeDrive.planreg_v2.action import V2ActionDecoder
    import navsim.agents.EpisodeDrive.planreg_v2.targets as targets
    if mutation=='ego_before':
        method=V2ActionDecoder.score
        source=textwrap.dedent(inspect.getsource(method)).replace(
            'self.scorer_attention(embedded, memory,','self.scorer_attention(embedded+ego[:,None], memory,').replace(
            'hidden = hidden + ego[:,None]','hidden = hidden')
        namespace=dict(method.__globals__);exec(compile(source,'<pre-ego mutation>','exec'),namespace)
        monkeypatch.setattr(V2ActionDecoder,'score',namespace['score'])
        with pytest.raises(AssertionError):scorer_integration()
    else:
        method=targets.long_target;source=inspect.getsource(method)
        if mutation=='uniform_long':
            source=source.replace('query = np.interp(q_index,np.arange(10),actual)',
                                  'query = np.arange(1,9)*actual[-1]/8.')
        else:
            source=source.replace("output = CubicSpline(actual,values,bc_type='not-a-knot',extrapolate=False)(query)",
                                  'output = np.stack([np.interp(query,actual,values[:,i]) for i in range(3)],-1)')
        namespace=dict(method.__globals__);exec(compile(source,'<long target mutation>','exec'),namespace)
        # Audit resolves its production callable at import; patch that binding
        # with the deliberately mutated production body, not a changed reference.
        monkeypatch.setitem(long_parity.__globals__,'long_target',namespace['long_target'])
        with pytest.raises(AssertionError):long_parity()
