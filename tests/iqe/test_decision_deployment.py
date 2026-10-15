from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from iqe.contracts import FeatureBundle,TrajectoryBundle
from iqe.query_base import source_trajectory_contract
from iqe.decision_selection.deployment import DecisionSystem
from iqe.decision_selection.models import SceneRouter,SetSelector
from iqe.decision_selection.evaluation import deployment_choices
from iqe.decision_selection.bundle import head_from_payload


class Reader(nn.Module):
    def forward(self,q,a):return a


class Branch(nn.Module):
    def __init__(self,offset,invalid=False):
        super().__init__();self.offset=offset;self.invalid=invalid
    def forward(self,f):
        raw=f.scene[:,0,:4,None].transpose(1,2).expand(-1,8,-1).clone()+self.offset
        if self.invalid:raw[0]=float('nan')
        physical=raw[...,:3].clone()
        return TrajectoryBundle(raw,physical,(raw,),.5)


class Bridge(nn.Module):
    def __init__(self):
        super().__init__();self.q_former=Reader();self.expert=Branch(0)
        self.trajectory_contract=source_trajectory_contract()
    def encode_features(self,a,ego,ids,contract):
        scene=self.q_former(None,a)
        return FeatureBundle(scene,ego,torch.ones(scene.shape[:2],dtype=torch.bool),contract,tuple(ids))


class Adapter(nn.Module):
    def __init__(self):
        super().__init__();self.contract_hash='test';self.framework=nn.Module()
        self.framework.action_model=Bridge()
        network=SimpleNamespace(visual=nn.Identity(),language_model=nn.Identity())
        self.framework.qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(model=network))
        def encode(rows):
            x=torch.tensor([r['state'] for r in rows],dtype=torch.float32)
            a=network.language_model(network.visual(x[:,None].expand(-1,2,-1)))
            self.framework.action_model.ego_state=x
            return dict(W=a,action_queries=a)
        self.framework.encode_current=encode
        self.framework.build_planner_condition=lambda e:e['action_queries']
    def predict_s0(self,f):return self.framework.action_model.expert(f)


class Expert(nn.Module):
    def __init__(self,j,invalid=False):
        super().__init__();self.private_reader=True;self.q_former=Reader();self.expert=Branch(j,invalid)
    def forward(self,x):
        a=self.q_former(None,x['action'])
        return self.expert(FeatureBundle(a,x['current_ego'],torch.ones(a.shape[:2],dtype=torch.bool),'test',tuple(map(str,range(len(a))))))


class RoutedHead(nn.Module):
    def __init__(self):
        super().__init__();self.context=nn.Identity();self.expert_ids=tuple('expert_'+str(i) for i in range(4))
    def forward(self,c):
        c=self.context(c)
        return dict(decision_logits=torch.nn.functional.one_hot(c['current_ego'][:,0].long(),4).float())


def system(invalid=False):
    return DecisionSystem(Adapter(),{f'expert_{i}':Expert(i,invalid and i==3) for i in range(1,4)},RoutedHead(),'M_scene',[f'expert_{i}' for i in range(4)])


def observations():
    return [dict(image=object(),lang='current',state=[i,0,0,0],token=str(n)) for n,i in enumerate((3,1,3,0))]


def test_true_prerouting_grouping_order_and_labels():
    m=system();r=m(observations(),return_context=True)
    assert r['selected'].tolist()==[3,1,3,0]
    assert r['trajectory'][:,0,0].tolist()==[6,2,6,0]
    assert r['calls']==dict(visual=1,language=1,context_projections=1,
        reader={'expert_0':1,'expert_1':1,'expert_2':0,'expert_3':1},
        decoder={'expert_0':1,'expert_1':1,'expert_2':0,'expert_3':1},
        decoder_samples={'expert_0':1,'expert_1':1,'expert_2':0,'expert_3':2})
    dirty=[x|dict(GT=torch.tensor(float('nan')),score=object(),future_actors=object(),metric_cache=object()) for x in observations()]
    other=m(dirty,return_context=True)
    assert torch.equal(r['trajectory'],other['trajectory'])
    assert all(torch.equal(r['context'][k],other['context'][k]) for k in r['context'])
    m.train();assert all(not n.training for n in m.modules())


def test_invalid_selected_output_fallback_is_real_additional_base_call():
    r=system(invalid=True)(observations())
    assert r['primary_selected'].tolist()==[3,1,3,0]
    assert r['selected'].tolist()==[0,1,3,0]
    assert r['invalid_fallback'].tolist()==[True,False,False,False]
    assert r['calls']['decoder']['expert_0']==2
    assert r['calls']['decoder_samples']['expert_0']==2


def test_set_calls_every_branch_once_and_original_base_unchanged():
    torch.manual_seed(123)
    a=Adapter();experts={f'expert_{i}':Expert(i) for i in range(1,4)}
    head=SetSelector(4,4,source_trajectory_contract())
    m=DecisionSystem(a,experts,head,'M_set',[f'expert_{i}' for i in range(4)])
    rows=observations();r=m(rows);b=m(rows,base_only=True)
    assert r['calls']['visual']==r['calls']['language']==r['calls']['context_projections']==1
    assert all(v==1 for v in r['calls']['decoder'].values())
    assert torch.equal(r['candidates'][:,0],b['trajectory'])
    assert b['calls']['context_projections']==0
    assert b['calls']['decoder']==dict(expert_0=1,expert_1=0,expert_2=0,expert_3=0)


def test_bank_validity_not_used_to_choose_before_router_decoder():
    bank=dict(trajectories=np.ones((1,4,8,3),np.float32))
    bank['trajectories'][0,3]=np.nan
    logits=np.array([[0.,1.,2.,3.]],np.float32);ids=[f'expert_{i}' for i in range(4)]
    routed,a=deployment_choices('M_scene',logits,bank,ids)
    set_selected,b=deployment_choices('M_set',logits,bank,ids)
    assert a['primary_selected']==[3] and routed.tolist()==[0] and a['extra_Base_decoder_calls']==1
    assert set_selected.tolist()==[2] and b['extra_Base_decoder_calls']==0


def test_privileged_and_wrong_schema_cannot_be_exported():
    for p in (dict(kind='P_current'),dict(kind='M_scene',schema='iqe_cost_sensitive_M_set_research_v1')):
        with pytest.raises(ValueError):head_from_payload(p)
