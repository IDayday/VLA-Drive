"""Small CPU wiring contracts; no real S0 image/GPU qualification is claimed."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np
import pytest
import torch
from torch import nn

from tools.action_effect import qualify_online_deployment as online
from tools.action_effect.common import artifact,digest,sha256,write_json
from tools.action_effect.export_models import load_bundle
from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow
from starVLA.model.modules.action_effect.effect_model import EffectModelConfig
from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner,RefinerConfig
from starVLA.model.framework.effect_guided_refiner import EffectGuidedRefiner


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__();self.qwen_proj=nn.Linear(4,4);self.model=nn.Linear(4,4)
        self.config=SimpleNamespace();self.num_timestep_buckets=1000
        self.num_inference_timesteps=10;self.action_horizon=8;self.action_dim=4

    def forward_onestep(self,condition,state,times):
        return [self.model(state)+condition.mean(1)[:,None,:]*.01]

    def predict_action(self,condition,*,initial_noise):
        projected=self.qwen_proj(condition);out=initial_noise
        for i in range(10):
            times=torch.full((len(out),),i*100,device=out.device)
            out=out+.1*self.forward_onestep(projected,out,times)[0]
        return out


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Linear(1,1);self.language=nn.Linear(1,1);self.action_model=TinyHead()
        self.qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(visual=self.visual,language_model=self.language)))

    def encode_current(self,observations):
        assert all(set(o)=={'image','lang','state'} for o in observations)
        feature=self.language(self.visual(torch.ones(1,1))).reshape(1,1,1)
        return {'W':feature.expand(1,2,4),'action_queries':feature.expand(1,8,4)}

    def build_planner_condition(self,encoded):return encoded['action_queries']


def modules():
    torch.set_num_threads(1);torch.manual_seed(2)
    base=TinyBase().requires_grad_(False).eval()
    adapter=EffectFlowAdapter(EffectFlowAdapterConfig(condition_dim=4,hidden_dim=16,heads=2,layers=1))
    flow=EffectInternalizedFlow(base.action_model,adapter,arm='B1-trajectory').requires_grad_(False).eval()
    cfg=RefinerConfig(model=EffectModelConfig(context_dim=4,world_tokens=2,action_tokens=8,hidden_dim=16,layers=1,heads=2))
    refiner=EffectGuidedRefiner(CoefficientRefiner(cfg)).requires_grad_(False).eval()
    return base,refiner,flow


def observation():return {'image':[object(),object(),object()],'lang':'keep straight','state':np.zeros((1,4),np.float32),'token':'real_fixture'}


def test_once_shared_encoder_single_original_head_and_no_second_rollout():
    base,refiner,flow=modules();pipeline=online.OnlinePipeline(base,refiner=refiner,flow=flow)
    noise=torch.randn(1,8,4);outputs={}
    try:
        for mode in ('S0','A','B'):
            row=pipeline.run(mode,observation(),noise);outputs[mode]=online.frozen_output(row)
            assert row['calls']['visual']==row['calls']['language']==row['calls']['planner_projection']==1
            assert row['calls']['DiT']==10 and row['complete_generations']==1
            assert row['calls']['refiner']==int(mode=='A')
            assert row['calls']['velocity_adapter']==10*int(mode=='B')
        assert torch.equal(outputs['S0']['raw'],outputs['B']['raw']) # zero final adapter
        assert torch.equal(outputs['S0']['physical'],outputs['A']['physical']) # zero refiner
    finally:pipeline.close()
    assert not pipeline.handles


def test_future_labels_token_and_metric_cache_cannot_enter_any_route():
    base,refiner,flow=modules();pipeline=online.OnlinePipeline(base,refiner=refiner,flow=flow);noise=torch.randn(1,8,4)
    try:
        for mode in ('S0','A','B'):
            ordinary=online.frozen_output(pipeline.run(mode,observation(),noise))
            poison=online.frozen_output(pipeline.run(mode,online.poisoned(observation()),noise))
            assert online.compare_outputs(ordinary,poison)
    finally:pipeline.close()
    with pytest.raises(ValueError,match='delta-pose'):online.current_only({**observation(),'state':np.zeros((1,3))})


def test_wrong_head_or_training_aux_heads_rejected():
    base,refiner,flow=modules()
    flow.base.action_head=TinyHead()
    with pytest.raises(ValueError,match='exact original'):online.OnlinePipeline(base,flow=flow)
    aux=EffectFlowAdapter(EffectFlowAdapterConfig(condition_dim=4,hidden_dim=16,heads=2,layers=1,effect_dim=11))
    flow=EffectInternalizedFlow(base.action_model,aux,arm='B1-effect',effect_protocol='verified_pair_v1').requires_grad_(False).eval()
    with pytest.raises(ValueError,match='strip training-only'):online.OnlinePipeline(base,flow=flow)
    refiner.effect_teacher=nn.Linear(1,1)
    with pytest.raises(ValueError,match='only its private'):online.OnlinePipeline(base,refiner=refiner)


def test_repeated_visual_call_and_nonfinite_output_fail():
    base,refiner,flow=modules();original=base.encode_current
    def twice(rows):base.visual(torch.ones(1,1));return original(rows)
    base.encode_current=twice;pipeline=online.OnlinePipeline(base)
    try:
        with pytest.raises(ValueError,match='call contract'):pipeline.run('S0',observation(),torch.randn(1,8,4))
    finally:pipeline.close()
    with pytest.raises(ValueError,match='Noncanonical/nonfinite'):
        online.frozen_output({key:torch.tensor([float('nan')]) for key in ('W','A','condition','ego','raw','physical_before_refiner','physical')})


def test_injected_bundle_uses_same_head_without_opening_head_cache(tmp_path,monkeypatch):
    from tools.st_world_policy import base_binding
    from tools.action_effect import train_effect_flow
    base,_,flow=modules();payload={'schema':'action_effect_private_deploy_v1','route':'B','arm':'B1-effect',**flow.adapter.velocity_export()}
    private=tmp_path/'PRIVATE_HEAD.pt';torch.save(payload,private)
    binding=tmp_path/'BASE_BINDING.json';binding.write_text('{}')
    monkeypatch.setattr(base_binding,'read_binding',lambda p:{'binding_sha256':'fixture-base'})
    monkeypatch.setattr(train_effect_flow,'load_original_action_head',lambda *a,**kw:pytest.fail('must not read another head blob'))
    source='tools/action_effect/common.py'
    manifest={'schema':'action_effect_research_deployment_bundle_v1','route':'B','arm':'B1-effect',
        'base_binding':artifact(binding),'base_binding_hash':'fixture-base','private_head':artifact(private),
        'source_inventory':{source:sha256(source)},
        'effect_teacher_required':False,'simulator_required':False,'training_cache_required':False,'GT_required':False,'LiDAR_required':False,
        'original_action_head_receipt':{'path':'/unavailable/contexts/S0_ACTION_HEAD.COMPLETE.json','sha256':'not-read'},
        'original_action_head_contract':{'schema':'action_effect_original_action_head_v1','base_binding_hash':'fixture-base','strict_original_master':True}}
    manifest['identity']=digest(manifest);path=tmp_path/'BUNDLE.json';write_json(path,manifest)
    loaded=load_bundle(path,original_action_head=base.action_model,original_base_binding_hash='fixture-base')
    assert loaded.base.action_head is base.action_model and loaded.adapter.config.effect_dim==0
    with pytest.raises(ValueError,match='same verified original Base'):
        load_bundle(path,original_action_head=base.action_model,original_base_binding_hash='wrong')


def test_runtime_guards_and_latency_sample_boundaries():
    data=online.DeploymentReadGuard()
    for path in ('/data/ego/token.pt','/data/metric_cache/scene.pkl','/data/contexts/feature.pt','/data/PAIR_DATASET.json',
                 '/data/metric_cache.pkl','/data/future_latent_cache/scene.pt','/data/LiDAR/frame.bin'):
        with pytest.raises(PermissionError):data('open',(path,'r',0))
    data('open',('/data/current/token.json','r',0));data('open',('/code/tools/action_effect/context_data.py','r',0))
    imports=online.DeploymentImportGuard()
    for name in ('navsim.scorer','tools.action_effect.train_effect_model','tools.action_effect.context_data'):
        with pytest.raises(ImportError):imports.find_spec(name)
    with pytest.raises(ValueError,match='eight'):online.latency_summary([1.,2.,3.])
    assert online.latency_summary([.1]*8)['p95_seconds']==pytest.approx(.1)


def test_cli_dry_run_does_not_dispatch_and_refuses_three_samples(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(online.subprocess,'run',lambda *a,**kw:pytest.fail('dry run dispatched'))
    args=['--base-binding','/fixture/base','--cohort','/fixture/cohort','--scene-id','fixture',
          '--output',str(tmp_path/'out'),'--device','cuda:0','--modes','S0','--dry-run']
    assert online.main(args)==0 and json.loads(capsys.readouterr().out)['status']=='NOT_RUN'
    assert not (tmp_path/'out').exists()
    assert online.main(args+['--repeats','3'])==1
    assert '8..128' in capsys.readouterr().out
    assert online.main(args+['--worker'])==0
    assert json.loads(capsys.readouterr().out)['status']=='NOT_RUN'


def test_optional_prediction_join_is_after_policy_and_checks_identity(tmp_path,monkeypatch):
    from tools.st_world_policy import base_binding
    binding=tmp_path/'binding.json';binding.write_text('{}')
    monkeypatch.setattr(base_binding,'read_binding',lambda p:{'checkpoint':{'sha256':'real-base'}})
    reg={'schema':'action_effect_refiner_prediction_registration_v1','base_checkpoint':{'sha256':'real-base'},
         'scene_ids':['scene'],'arm':'S0'}
    reg['identity']=digest(reg);write_json(tmp_path/'REG.json',reg)
    row={'scene_id':'scene','observation_hash':'real-image-hash','physical':[[1.,2.,0.]]*8};row['identity']=digest(row)
    m={'schema':'action_effect_refiner_predictions_v1','status':'COMPLETE','registration':artifact(tmp_path/'REG.json'),
       'registration_identity':reg['identity'],'requested_scenes':1,'entries':[row]};m['identity']=digest(m)
    pin=tmp_path/'PREDICTIONS.json';write_json(pin,m)
    actual={'scene_id':'scene','observation_hash':'real-image-hash','base_binding':artifact(binding)}
    assert online.compare_prediction(pin,'S0',{'physical':row['physical']},actual)['exact_physical']
    with pytest.raises(ValueError,match='observation mismatch'):
        online.compare_prediction(pin,'S0',{'physical':row['physical']},{**actual,'observation_hash':'other-image'})
    m['registration_identity']='wrong';m['identity']=digest({k:v for k,v in m.items() if k!='identity'})
    wrong=tmp_path/'WRONG.json';write_json(wrong,m)
    with pytest.raises(ValueError,match='identity-bound'):
        online.compare_prediction(wrong,'S0',{'physical':row['physical']},actual)


def test_only_trusted_importlib_atomic_code_writes_are_exempt(tmp_path):
    code=tmp_path/'code';(code/'__pycache__').mkdir(parents=True)
    (code/'future_latent_head.py').write_text('VALUE=1\n')
    path=code/'__pycache__/future_latent_head.cpython-310.pyc.123456'
    guard=online.DeploymentReadGuard(code_roots=[code])
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    guard('open',(str(path),None,flags))
    assert len(guard.permitted_code_cache_writes)==1 and not guard.blocked
    for name,mode in ((path,os.O_RDONLY),(path,os.O_RDWR|os.O_CREAT),
                      (path.with_name('future_latent_head.cpython-310.pyc.BAD'),flags),
                      (code/'future_targets.pt',flags)):
        with pytest.raises(PermissionError):guard('open',(str(name),None,mode))
    outside=tmp_path/'data';(outside/'__pycache__').mkdir(parents=True)
    (outside/'future_latent_head.py').write_text('VALUE=1\n')
    with pytest.raises(PermissionError):guard('open',(str(outside/'__pycache__'/path.name),None,flags))


def test_rejected_resource_has_safe_stack_and_durable_diagnostic(tmp_path):
    guard=online.DeploymentReadGuard();imports=online.DeploymentImportGuard()
    with pytest.raises(PermissionError):guard('open',('/data/metric_cache.pkl','rb',os.O_RDONLY))
    with pytest.raises(ImportError):imports.find_spec('tools.action_effect.context_data')
    result=online.dependency_audit(tmp_path,guard,imports)
    assert result['status']=='FAIL' and (tmp_path/'DEPENDENCY_AUDIT.json').exists()
    detail=result['blocked_data_opens'][0]
    assert detail['stack'] and all(set(f)=={'file','line','function'} for f in detail['stack'])
    assert 'metric_cache.pkl' not in detail['path'] # Unknown external paths are hashed, never emitted raw.
    assert result['blocked_import_attempts'][0]['module']=='tools.action_effect.context_data'
