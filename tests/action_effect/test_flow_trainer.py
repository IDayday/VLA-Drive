"""Small CPU behavioral fixtures, not pretrained S0/NAVSIM qualification."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from tools.action_effect.train_effect_flow import (FlowStep,FlowLossRecipe,
    balance_examples,RawActionEffectTeacher,load_flow_examples)
from tools.action_effect.context_data import batch_context
from tools.action_effect.training import run_training
from tools.action_effect.common import digest,artifact,write_json
from test_flow_internalization import model as tiny_model


@pytest.fixture(autouse=True)
def bounded_cpu():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixtures():
    g=torch.Generator().manual_seed(998)
    records={};examples=[]
    for i in range(4):
        raw=torch.randn(8,4,generator=g);raw[:,2]=0.;raw[:,3]=1.
        records[str(i)]={'scene_id':str(i),'source_log':'log'+str(i),'role':'fit',
            'condition':torch.randn(8,12,generator=g),'memory':torch.randn(10,12,generator=g),
            'ego':torch.zeros(1,4),'raw':raw,'observation_hash':'obs'+str(i)}
        from starVLA.dataloader.foresight_dataset import decode_ego
        records[str(i)]['physical']=decode_ego(raw)
        repair=i%2==0
        examples.append({'scene_id':str(i),'target_raw':raw+.1 if repair else raw.clone(),
            'is_repair':repair,'is_preserve':not repair,'effect_delta':torch.tensor([.2]),
            'effect_valid':torch.tensor([True])})
    return records,examples


def make_step(records,examples,micro=2):
    ng=torch.Generator().manual_seed(122);tg=torch.Generator().manual_seed(123)
    step=FlowStep(examples,records.__getitem__,recipe=FlowLossRecipe(microbatch_scenes=micro),
        device='cpu',noise_generator=ng,time_generator=tg,context_batcher=batch_context)
    return step,ng,tg


@pytest.mark.parametrize('arm',['B0','B1-trajectory'])
def test_repeat8_microbatch_denominators_rng_and_single_base_call(arm):
    records,examples=fixtures();a=tiny_model(arm);b=deepcopy(a)
    sa,nga,tga=make_step(records,examples,1);sb,ngb,tgb=make_step(records,examples,4)
    calls=[];original=b.base_velocity
    def counted(*args):calls.append(len(args[0]));return original(*args)
    b.base_velocity=counted
    ra=sa(a,[0,1,2,3],0);rb=sb(b,[0,1,2,3],0)
    assert calls==[32]  # one Base pass, no keep re-evaluation
    assert ra['FM_noise_presentations']==rb['FM_noise_presentations']==32
    assert torch.equal(nga.get_state(),ngb.get_state()) and torch.equal(tga.get_state(),tgb.get_state())
    for pa,pb in zip(a.adapter.parameters(),b.adapter.parameters()):
        assert torch.allclose(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)
    assert abs(ra['velocity']-rb['velocity'])<2e-6
    assert all(p.grad is None for p in a.base.parameters())


def test_balanced_reference_sampling_does_not_invent_unique_data():
    _,examples=fixtures();extra=deepcopy(examples[0]);extra['scene_id']='4';examples.append(extra)
    pool,report=balance_examples(examples)
    assert len(pool)==12 and report['references_per_class']==6
    assert sum(x['is_repair'] for x in pool)==sum(x['is_preserve'] for x in pool)==6
    assert report['unique_repair_scenes']==3 and report['unique_keep_scenes']==2
    assert len({id(x) for x in pool})==5 and report['unique_observations_added_by_oversampling']==0
    with pytest.raises(ValueError,match='BLOCKED_NO_REPAIR/KEEP'):
        balance_examples([examples[0]])


def test_reject_role_and_ambiguous_keep_targets_before_backward():
    records,examples=fixtures();step,_,_=make_step(records,examples);m=tiny_model()
    records['0']['role']='validation'
    with pytest.raises(ValueError,match='Validation/dev/test'):step(m,[0,1],0)
    records['0']['role']='fit';examples[0]['is_preserve']=True
    with pytest.raises(ValueError,match='exactly one'):step(m,[0,1],0)
    examples[0]['is_preserve']=False;examples[1]['target_raw']=examples[1]['target_raw']+.001
    with pytest.raises(ValueError,match='exact original raw'):step(m,[0,1],0)


class PhysicalTeacher(nn.Module):
    """Analytic physical fixture, never a learned/verified simulator teacher."""
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(2.),requires_grad=False)
        self.config=SimpleNamespace(mode='delta_feedback');self.eval()
    def forward(self,context,a,b):
        assert set(context)=={'memory','ego','mask'}
        return {'mean_delta':(b[...,0]-a[...,0]).mean(1,keepdim=True)*self.scale}


def test_training_only_physical_teacher_bridge_uses_original_normalizer_and_grad():
    records,_=fixtures();teacher=PhysicalTeacher();bridge=RawActionEffectTeacher(teacher)
    raw=records['0']['raw'][None];delta=torch.zeros_like(raw,requires_grad=True)
    context=batch_context([records['0']],'cpu');out=bridge(context,raw,delta)
    out.sum().backward()
    from starVLA.dataloader.foresight_dataset import X_STD
    assert torch.allclose(delta.grad[...,0],torch.full((1,8),2*X_STD/8))
    assert teacher.scale.grad is None and not bridge.training
    bridge.train();assert not teacher.training
    with pytest.raises(ValueError,match='Undefined sin/cos'):
        bridge(context,raw,torch.cat([torch.zeros_like(raw[...,:2]),-raw[...,2:]],-1))


def test_shared_loop_restores_both_flow_streams_and_terminal_no_extra_steps(tmp_path):
    records,examples=fixtures();a=tiny_model();b=deepcopy(a)
    cfg={'training':{'max_updates':4,'learning_rate':.001,'weight_decay':.01,
        'warmup_updates':1,'batch_scenes':4,'gradient_clip':1.,'validation_every':2}}
    sa,nga,tga=make_step(records,examples,2);sb,ngb,tgb=make_step(records,examples,2)
    def run(m,s,n,t,out,resume=False,stop_after=None):
        return run_training(m,output=out,identity={'source_commit':'fixture','arm':'B1-trajectory'},
            dataset_size=4,step_fn=s,cfg=cfg,device='cpu',seed=42,max_updates=4,
            stop_after=stop_after,resume=resume,stateful={'flow_noise':n,'flow_time':t})
    run(a,sa,nga,tga,tmp_path/'full')
    run(b,sb,ngb,tgb,tmp_path/'resume',stop_after=2)
    c=tiny_model();sc,ngc,tgc=make_step(records,examples,2)
    result=run(c,sc,ngc,tgc,tmp_path/'resume',resume=True)
    assert result['optimizer_step']==4 and result['status']=='COMPLETE'
    for k,v in a.adapter.state_dict().items():assert torch.equal(v,c.adapter.state_dict()[k])
    assert torch.equal(nga.get_state(),ngc.get_state()) and torch.equal(tga.get_state(),tgc.get_state())
    state=deepcopy(c.state_dict());rng=torch.get_rng_state().clone();ns=ngc.get_state().clone()
    done=run(c,sc,ngc,tgc,tmp_path/'resume',resume=True)
    assert done['segment_updates']==0 and done['optimizer_step']==4
    assert torch.equal(torch.get_rng_state(),rng) and torch.equal(ns,ngc.get_state())
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in state.items())


def test_cli_help_and_strict_teacher_arm_boundary():
    cmd=[sys.executable,'-m','tools.action_effect.train_effect_flow']
    result=subprocess.run(cmd+['--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--verify-head-only' in result.stdout
    result=subprocess.run(cmd+['--config','configs/action_effect/main.yaml','--context-manifest','unused',
        '--output','unused','--arm','B0','--repair-dataset','unused','--max-updates','1',
        '--effect-checkpoint','forbidden','--dry-run'],capture_output=True,text=True)
    assert result.returncode!=0 and 'Only B1-effect' in result.stderr


def test_actual_g0_adapter_role_join_raw_keep_and_missing_class(tmp_path):
    records,examples=fixtures();binding={'base_checkpoint':{'sha256':'base'},'identity':'binding'}
    reader=SimpleNamespace(binding=binding,rows={s:{'scene_id':s} for s in records})
    class Reader:
        def __init__(self):self.binding=binding;self.rows=reader.rows
        def __getitem__(self,s):return records[s]
    reg={'binding_identity':'binding'};reg['identity']=digest(reg);write_json(tmp_path/'REG.json',reg)
    write_json(tmp_path/'DONE.json',{'registration_identity':reg['identity']})
    rows=[]
    for s,r in records.items():
        keep=int(s)%2==1
        rows.append({'scene_id':s,'source_log':r['source_log'],'role':'fit','context_entry':reader.rows[s],
            'observation_hash':r['observation_hash'],'base_physical':r['physical'].tolist(),
            'target_physical':(r['physical']+.1).tolist(),'trainable':True,
            'target_kind':'identity_preservation' if keep else 'verified_local_repair',
            'base_labels':{'names':['score'],'units':['0..1'],'values':[.5],'valid':[True]},
            'selected_labels':{'names':['score'],'units':['0..1'],'values':[.6],'valid':[True]}})
    data={'schema':'action_effect_repair_dataset_v1','status':'COMPLETE','split':'train','deployment_inputs':False,
        'registration':artifact(tmp_path/'REG.json'),'registration_identity':reg['identity'],
        'population_completion':artifact(tmp_path/'DONE.json'),'base_checkpoint':binding['base_checkpoint'],'entries':rows}
    def save():
        data.pop('identity',None);data['identity']=digest(data);p=tmp_path/'DATA.json';p.write_text(json.dumps(data));return p
    loaded,report=load_flow_examples(save(),Reader())
    assert report['counts']['repairs']==2 and report['counts']['preserve']==2
    for ex in loaded:
        if ex['is_preserve']:assert torch.equal(ex['target_raw'],records[ex['scene_id']]['raw'])
    rows[0]['context_entry']={'scene_id':'wrong'}
    with pytest.raises(ValueError,match='observation/log/role'):load_flow_examples(save(),Reader())
