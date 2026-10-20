"""CPU metadata/command negatives; no real S0, CUDA or deployment result."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from tools.action_effect import prepare_final_online as prep
from tools.action_effect.common import artifact, digest, read_json, write_json


def saved_endpoint(tmp_path, arm):
    run=tmp_path/arm;run.mkdir()
    context=tmp_path/'CONTEXT.json';repair=tmp_path/'REPAIR.json'
    for p in (context,repair):p.write_text('{}')
    teacher={'checkpoint':{'path':'/own/g1/final.pt','sha256':'own'},'external_weights_loaded':False,'base_binding_hash':'base'}
    identity={'schema':'action_effect_small_training_v1','arm':arm,'seed':42,'max_updates':1000,
        'base_binding_hash':'base','context_artifact':artifact(context),'effect_teacher':teacher,
        'runtime':{'trainable_dtype':'float32','autocast':'disabled','deterministic':True,'matmul_tf32':False,'cudnn_tf32':False}}
    if arm=='A1':identity['data_artifact']=artifact(repair)
    else:identity.update(repair_data={'dataset':artifact(repair)},effect_protocol='verified_pair_v1')
    write_json(run/'IDENTITY.json',identity)
    checkpoint=run/'step_001000.pt';checkpoint.write_bytes(b'metadata fixture, not model evidence')
    receipt={'checkpoint':artifact(checkpoint),'optimizer_step':1000,'run_identity':digest(identity),
        'terminal':True,'termination_reason':'MAX_UPDATES','created_unix_ns':1}
    write_json(checkpoint.with_suffix('.COMPLETE.json'),receipt)
    write_json(run/'RESULT.json',{'schema':'action_effect_training_result_v1','status':'COMPLETE',
        'optimizer_step':1000,'termination_reason':'MAX_UPDATES','run_identity':digest(identity),'checkpoint':receipt})
    kwargs={'arm':arm,'seed':42,'base_hash':'base','context_pin':artifact(context),'repair_pin':artifact(repair)}
    return run,kwargs


def test_only_complete_explicit_final_endpoint_selected(tmp_path):
    run,kwargs=saved_endpoint(tmp_path,'A1')
    value=prep.endpoint(run,**kwargs)
    assert value['optimizer_step']==1000 and value['checkpoint']['path']==str(run/'step_001000.pt')
    (run/'step_009999.pt').write_bytes(b'newer unrelated file must not be scanned')
    assert prep.endpoint(run,**kwargs)==value
    # Existing shared trainer terminal resume returns a verified artifact pin.
    result=read_json(run/'RESULT.json');result['checkpoint']=result['checkpoint']['checkpoint']
    write_json(run/'RESULT.json',result,immutable=False)
    assert prep.endpoint(run,**kwargs)['checkpoint']==value['checkpoint']


@pytest.mark.parametrize('change',[
    {'arm':'A0'},{'seed':43},{'max_updates':16},{'base_binding_hash':'foreign'},
    {'context_artifact':{'sha256':'wrong'}},{'data_artifact':{'sha256':'wrong'}},
    {'runtime':{'trainable_dtype':'bfloat16'}},{'effect_teacher':{}},
    {'effect_teacher':{'external_weights_loaded':True,'base_binding_hash':'base','checkpoint':{'path':'external'}}},
])
def test_wrong_final_identity_fails_before_export(tmp_path,change):
    run,kwargs=saved_endpoint(tmp_path,'A1');d=read_json(run/'IDENTITY.json');d.update(change)
    write_json(run/'IDENTITY.json',d,immutable=False)
    with pytest.raises(ValueError):prep.endpoint(run,**kwargs)


def test_missing_or_paused_endpoint_waits_not_exported(tmp_path):
    with pytest.raises(prep.WaitingInputs):prep.endpoint(tmp_path/'missing',arm='A1',seed=42,base_hash='base',context_pin={},repair_pin={})
    run,kwargs=saved_endpoint(tmp_path,'A1');d=read_json(run/'RESULT.json');d['status']='PAUSED'
    write_json(run/'RESULT.json',d,immutable=False)
    with pytest.raises(prep.WaitingInputs):prep.endpoint(run,**kwargs)


def test_foreign_receipt_or_changed_checkpoint_rejected(tmp_path):
    run,kwargs=saved_endpoint(tmp_path,'A1');d=read_json(run/'RESULT.json')
    d['checkpoint']['run_identity']='other';write_json(run/'RESULT.json',d,immutable=False)
    with pytest.raises(ValueError,match='receipt/training'):prep.endpoint(run,**kwargs)
    (run/'step_001000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='stale'):prep.endpoint(run,**kwargs)


def test_legacy_proxy_effect_arm_rejected(tmp_path):
    run,kwargs=saved_endpoint(tmp_path,'B1-effect');d=read_json(run/'IDENTITY.json');d['effect_protocol']='legacy_endpoint_proxy'
    write_json(run/'IDENTITY.json',d,immutable=False)
    with pytest.raises(ValueError,match='verified-pair'):prep.endpoint(run,**kwargs)


def plan_fixture(tmp_path,monkeypatch,*,missing=False):
    from tools.st_world_policy import base_binding
    root=tmp_path/'root';root.mkdir();(root/'contexts').mkdir();(root/'g0/population_001024').mkdir(parents=True)
    for relative in ('contexts/MANIFEST_001024.json','g0/population_001024/REPAIR_DATASET.json','BASE_BINDING.json'):
        (root/relative).write_text('{}')
    cohort={'schema':'action_effect_observation_manifest_v1','split':'train','base_binding_hash':'base',
        'entries':[{'scene_id':'scene'}]};cohort['identity']=digest(cohort);write_json(root/'COHORT.json',cohort)
    campaign={'base_binding':artifact(root/'BASE_BINDING.json'),'base_checkpoint_binding_hash':'base','cohort':artifact(root/'COHORT.json')}
    campaign['identity']=digest(campaign);write_json(root/'CAMPAIGN_BINDING.json',campaign)
    config=tmp_path/'config.yaml';config.write_text('fixture')
    cfg={'artifact_root':str(root),'base_binding':str(root/'BASE_BINDING.json'),'cohort_size':1024,'training':{'max_updates':1000}}
    monkeypatch.setattr(prep,'load_config',lambda p:cfg)
    monkeypatch.setattr(base_binding,'read_binding',lambda p:{'binding_sha256':'base'})
    def fake_endpoint(run,**kw):
        if missing:raise prep.WaitingInputs('not yet')
        return {'arm':kw['arm'],'checkpoint':{'path':str(Path(run)/'final.pt')},'effect_teacher':{'checkpoint':{'path':'/same/g1'}}}
    monkeypatch.setattr(prep,'endpoint',fake_endpoint)
    args=SimpleNamespace(config=str(config),output=str(tmp_path/'closure'),a_run=None,b_run=None,seed=42,
        scene_id='scene',device='cuda:0',max_seconds=1800,export_max_seconds=1200,
        s0_predictions=None,a_predictions=None,b_predictions=None,phase='plan')
    return args


def test_plan_one_shared_fresh_worker_and_cpu_exports(tmp_path,monkeypatch):
    args=plan_fixture(tmp_path,monkeypatch);plan=prep.plan(args)
    assert plan['status']=='READY' and not Path(args.output).exists()
    for arm,route in [('A1','A'),('B1-effect','B')]:
        cmd=plan['export_commands'][arm]
        assert cmd[cmd.index('--route')+1]==route and cmd[-2:]==['--device','cpu']
    cmd=plan['online_command']
    assert cmd.count('tools.action_effect.qualify_online_deployment')==1
    assert cmd[cmd.index('--modes')+1:cmd.index('--warmup')]==['S0','A','B']
    assert cmd[cmd.index('--warmup')+1]=='2' and cmd[cmd.index('--repeats')+1]=='8'
    assert plan['budget']['total_online_predictions']==33 and plan['budget']['total_original_DiT_steps']==330
    assert plan['budget']['training_updates']==plan['budget']['simulator_calls']==0
    assert plan['closure_source']['tools/action_effect/qualify_online_deployment.py']


def test_waiting_plan_cannot_dispatch(tmp_path,monkeypatch):
    args=plan_fixture(tmp_path,monkeypatch,missing=True);args.phase='all'
    plan=prep.plan(args);assert plan['status']=='WAITING_INPUTS'
    monkeypatch.setattr(prep,'invoke',lambda *a,**kw:pytest.fail('must not run before both final results'))
    assert prep.execute(args,plan)==plan and not Path(args.output).exists()


def test_invoke_hides_gpu_only_for_exports(tmp_path,monkeypatch):
    captured=[]
    def run(cmd,**kw):captured.append((cmd,kw));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(prep.subprocess,'run',run)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    prep.invoke(['cpu-export'],cwd=tmp_path,output=tmp_path/'export',cpu=True,timeout=100)
    prep.invoke(['one-online-launcher'],cwd=tmp_path,output=tmp_path/'online',cpu=False,timeout=100)
    assert captured[0][1]['env']['CUDA_VISIBLE_DEVICES']==''
    assert captured[1][1]['env']['CUDA_VISIBLE_DEVICES']=='3'
    assert all(x[1]['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8' for x in captured)
    assert not torch.cuda.is_initialized()


def test_wrong_export_endpoint_rejected_before_module_load(tmp_path):
    write_json(tmp_path/'BUNDLE.json',{'schema':'action_effect_research_deployment_bundle_v1',
        'route':'B','arm':'B1-effect','identity':'bad'})
    with pytest.raises(ValueError):prep.validate_bundle(tmp_path/'BUNDLE.json',{'arm':'B1-effect'},
        route='B',base_pin={},base_hash='base')


def test_real_small_B_export_strips_auxiliary_heads_and_reuses_original_object(tmp_path,monkeypatch):
    from tools.st_world_policy import base_binding
    from tools.action_effect import evaluate_effect_flow,export_models
    from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
    from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow
    torch.set_num_threads(1)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.qwen_proj=torch.nn.Linear(4,4);self.config=SimpleNamespace()
            self.num_timestep_buckets=1000;self.num_inference_timesteps=10;self.action_horizon=8;self.action_dim=4
        def forward_onestep(self,c,state,t):return [state*.1+c.mean(1)[:,None,:]*.01]
        def predict_action(self,c,*,initial_noise):
            c=self.qwen_proj(c);state=initial_noise
            for i in range(10):state=state+.1*self.forward_onestep(c,state,i*100)[0]
            return state
    head=Head().requires_grad_(False).eval()
    adapter=EffectFlowAdapter(EffectFlowAdapterConfig(condition_dim=4,hidden_dim=16,heads=2,layers=1,effect_dim=11))
    original=EffectInternalizedFlow(head,adapter,arm='B1-effect',effect_protocol='verified_pair_v1').requires_grad_(False).eval()
    checkpoint=tmp_path/'step_001000.pt';checkpoint.write_bytes(b'fixture')
    original.provenance={'checkpoint':artifact(checkpoint),'optimizer_step':1000,'run_identity':'own-final','arm':'B1-effect'}
    binding=tmp_path/'BASE.json';binding.write_text('{}')
    receipt=tmp_path/'HEAD_RECEIPT.json';write_json(receipt,{'schema':'action_effect_original_action_head_v1',
        'base_binding_hash':'base','strict_original_master':True})
    write_json(tmp_path/'IDENTITY.json',{'head':{'head_receipt':artifact(receipt)}})
    monkeypatch.setattr(base_binding,'read_binding',lambda p:{'binding_sha256':'base'})
    def load(*a,**kw):
        assert kw['device']=='cpu' and kw['expected_base_binding_hash']=='base'
        return original
    monkeypatch.setattr(evaluate_effect_flow,'load_flow_policy',load)
    export_models.export(checkpoint,base_binding=binding,output=tmp_path/'deploy',route='B',device='cpu')
    manifest=read_json(tmp_path/'deploy/BUNDLE.json')
    payload=torch.load(tmp_path/'deploy/PRIVATE_HEAD.pt',map_location='cpu',weights_only=True)
    assert payload['config']['effect_dim']==0
    assert not any(k.startswith(('effect_head.','effect_projection.')) for k in payload['state_dict'])
    loaded=export_models.load_bundle(tmp_path/'deploy/BUNDLE.json',original_action_head=head,original_base_binding_hash='base')
    assert loaded.base.action_head is head and loaded.adapter.config.effect_dim==0
    condition=torch.randn(1,8,4);noise=torch.randn(1,8,4)
    with torch.no_grad():
        assert torch.equal(original.predict_action(condition,initial_noise=noise),loaded.predict_action(condition,initial_noise=noise))
    assert all(manifest[k] is False for k in ('effect_teacher_required','simulator_required','training_cache_required','GT_required','LiDAR_required'))


def test_cli_invalid_budget_never_runs(tmp_path,capsys):
    rc=prep.main(['--config','missing','--output',str(tmp_path/'out'),'--max-seconds','0','--phase','all'])
    assert rc==1 and json.loads(capsys.readouterr().out)['status']=='FAIL'
    assert not (tmp_path/'out').exists()
