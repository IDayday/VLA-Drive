"""Helper/negative tests only; these do not count as real GPU qualification."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from tools.action_effect import qualify_effect_resume as q
from tools.action_effect.common import artifact,digest,read_json,write_json


def history(step):
    return [{'optimizer_step':n,'losses':{'value':float(n)},'gradient_norm':.5,
             'lr':.0001,'consumed_scenes':32*n,'seconds':1.2} for n in range(1,step+1)]


def state(step=16):
    return {'model':{'weight':torch.tensor([.2,.3])},
        'optimizer':{'state':{0:{'step':torch.tensor(float(step)),
            'exp_avg':torch.tensor([.1,.2]),'exp_avg_sq':torch.tensor([.01,.02])}},'param_groups':[{'params':[0],'lr':.0001}]},
        'scheduler':{'last_epoch':step},'sampler':torch.Generator().manual_seed(1).get_state(),
        'stateful':{'pair_rng':torch.Generator().manual_seed(2).get_state()},
        'rng':{'numpy':np.random.RandomState(2).get_state(),'cpu':torch.Generator().manual_seed(3).get_state(),
               'python':(3,(1,2,3),None),'cuda':[]},'consumed_scenes':step*32,'optimizer_step':step,
        'run_identity':'fixture','termination_reason':'OPERATOR_PAUSED','terminal':False,'history':history(step)}


def test_all_state_components_and_actual_update_history_compared():
    left=state();right=deepcopy(left);right['history'][8]['seconds']=99.
    assert q.compare_checkpoint(left,right)['exact']
    right['optimizer']['state'][0]['exp_avg'][0]+=.01
    result=q.compare_checkpoint(left,right)
    assert not result['exact'] and any('exp_avg' in p for p in result['mismatches'])
    right=deepcopy(left);right['history'][8]['losses']['value']+=.01
    assert not q.compare_checkpoint(left,right)['exact']
    right=deepcopy(left);right['history']=[]
    with pytest.raises(ValueError,match='Missing actual update history'):q.compare_checkpoint(left,right)


def test_exact_comparison_rejects_nonfinite_schema_and_signed_zero():
    assert q.compare_exact(torch.tensor([0.]),torch.tensor([-0.]))
    assert q.state_hash(torch.tensor([0.]))!=q.state_hash(torch.tensor([-0.]))
    assert q.compare_exact(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))
    assert q.compare_exact(torch.tensor([float('nan')]),torch.tensor([float('nan')]))
    assert q.compare_exact(np.array([float('inf')]),np.array([float('inf')]))
    assert not q.compare_exact(state(),deepcopy(state()))


def checkpoint(root,step=9,foreign=False):
    root.mkdir(exist_ok=True);identity={'source':'unit fixture','recipe':'fixed1000'}
    write_json(root/'IDENTITY.json',identity)
    data=state(step);data['run_identity']='foreign' if foreign else digest(identity)
    path=root/f'step_{step:06d}.pt';torch.save(data,path)
    write_json(path.with_suffix('.COMPLETE.json'),{'checkpoint':artifact(path),'run_identity':digest(identity),
        'optimizer_step':step,'created_unix_ns':step})
    return path


def test_only_compatible_complete_checkpoint_from_named_run(tmp_path):
    a=tmp_path/'a';p=checkpoint(a)
    (a/'step_999999.pt').write_bytes(b'partial')
    assert q.at_step(a,9)[2]==p
    p.write_bytes(b'corrupt')
    assert q.checkpoints(a)==[]
    with pytest.raises(ValueError,match='Missing complete'):q.at_step(a,9)
    b=tmp_path/'b';checkpoint(b,foreign=True)
    assert q.checkpoints(b)==[]


def test_next_scene_pair_ids_reconstruct_actual_two_stream_sampling():
    data=SimpleNamespace(scenes=['scene0','scene1','scene2'],by_scene={
        'scene0':[{'identity':'pair00'},{'identity':'pair01'}],
        'scene1':[{'identity':'pair10'}],
        'scene2':[{'identity':'pair20'},{'identity':'pair21'},{'identity':'pair22'}]})
    sampler=torch.Generator().manual_seed(143);pair=torch.Generator().manual_seed(244)
    for _ in range(8):q.draw_batch(data,sampler,pair,32)
    before={'sampler':sampler.get_state(),'stateful':{'pair_rng':pair.get_state()}}
    expected=q.draw_batch(data,sampler,pair,32)
    after={'sampler':sampler.get_state(),'stateful':{'pair_rng':pair.get_state()}}
    result=q.next_batch_proof(data,before,after,deepcopy(after),seed=42,batch_size=32)
    assert result['exact'] and result['scene_pair_ids']==expected
    bad=deepcopy(before);bad['stateful']['pair_rng']=torch.Generator().manual_seed(9).get_state()
    assert not q.next_batch_proof(data,bad,after,after,seed=42,batch_size=32)['exact']


def test_budget_reserved_before_child_and_never_overspent(tmp_path,monkeypatch):
    args=SimpleNamespace(device='cpu',context_manifest='/fixture/context',pair_dataset='/fixture/pairs',arm='delta_feedback',seed=42)
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    class Failed:
        returncode=2
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:Failed())
    with pytest.raises(RuntimeError,match='child failed'):
        q.run_segment(cfg,args,tmp_path/'continuous16',16,resume_requested=False,output=tmp_path)
    assert read_json(tmp_path/'DISPATCH_BUDGET.json')['scheduled_updates']==16
    with pytest.raises(RuntimeError,match='child failed'):
        q.run_segment(cfg,args,tmp_path/'continuous16',16,resume_requested=True,output=tmp_path)
    assert read_json(tmp_path/'DISPATCH_BUDGET.json')['scheduled_updates']==32
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:pytest.fail('must not dispatch past budget'))
    with pytest.raises(ValueError,match='budget exhausted'):
        q.run_segment(cfg,args,tmp_path/'continuous16',16,resume_requested=True,output=tmp_path)


def test_completed_target_reused_without_child_dispatch(tmp_path,monkeypatch):
    checkpoint(tmp_path/'run',step=16)
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:pytest.fail('must not retrain completed target'))
    result=q.run_segment(None,None,tmp_path/'run',16,resume_requested=True,output=tmp_path)
    assert result['status']=='REUSED_COMPLETE' and result['new_updates']==0


def test_completion_reuse_preserves_result_pin_and_validates_artifacts(tmp_path):
    for name,body in [('IDENTITY.json',{'id':'unit'}),('QUALIFICATION_CONFIG.json',{'fixed_horizon':1000})]:
        write_json(tmp_path/name,body)
    checkpoint_file=tmp_path/'checkpoint.pt';checkpoint_file.write_bytes(b'unit checkpoint')
    result={'status':'PASS','exact_resume_this_runtime':True,'checkpoints':{'final':artifact(checkpoint_file)},
        'identity':artifact(tmp_path/'IDENTITY.json'),'qualification_config':artifact(tmp_path/'QUALIFICATION_CONFIG.json'),
        'commands':[],'new_updates_this_invocation':32}
    write_json(tmp_path/'RESULT.json',result)
    write_json(tmp_path/'COMPLETE.json',{'status':'PASS','identity':artifact(tmp_path/'IDENTITY.json'),'result':artifact(tmp_path/'RESULT.json')})
    before=(tmp_path/'RESULT.json').read_bytes()
    reused=q.reuse_complete(tmp_path)
    assert reused['reused_complete'] and reused['new_updates_this_invocation']==0
    assert (tmp_path/'RESULT.json').read_bytes()==before
    checkpoint_file.write_bytes(b'different')
    with pytest.raises(ValueError,match='stale dependency'):q.reuse_complete(tmp_path)


def test_dry_run_keeps_horizon_and_no_subprocess_training(tmp_path,monkeypatch):
    config=Path(__file__).resolve().parents[2]/'configs/action_effect/main.yaml'
    context=tmp_path/'context.json';context.write_text('{}')
    pairs=tmp_path/'pairs.json';pairs.write_text('{}')
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:pytest.fail('dry run must not train'))
    monkeypatch.setattr(q.subprocess,'check_output',lambda *a,**kw:'fixed-source-sha\n')
    args=SimpleNamespace(config=str(config),context_manifest=str(context),pair_dataset=str(pairs),
        output=str(tmp_path/'out'),resume=False,dry_run=True,device='cpu',seed=42,arm='delta_feedback')
    result=q.qualify(args)
    assert result['status']=='NOT_RUN'
    cfg=read_json(tmp_path/'out/QUALIFICATION_CONFIG.json')
    assert cfg['training']['max_updates']==1000 and cfg['training']['warmup_updates']==50
    assert cfg['training']['validation_every']==9
