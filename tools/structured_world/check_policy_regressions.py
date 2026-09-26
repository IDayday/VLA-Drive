"""Real Qwen regression checks, with declared tolerances and no optimizer updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from runtime import load_baseline,load_dataset,seed_all
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy


def main():
    p=argparse.ArgumentParser()
    for n in ['checkpoint','vlm','data-root','manifest','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--ledger');a=p.parse_args();seed_all(42)
    agent=load_baseline(a.checkpoint,a.vlm);agent.model.requires_grad_(False)
    ds=load_dataset(agent,a.manifest,a.data_root,2);samples=[ds[i] for i in range(2)]
    policy=StructuredWorldPolicy(agent.model,{'enabled':True,'agent_tokens':64}).cuda().eval()
    report={'tolerances':{'bf16_hidden':{'atol':.02,'rtol':.02},'fp32_hidden':{'atol':1e-5,'rtol':1e-5}},'checks':{}}
    original=agent.model.action_model.predict_action;captured=[]
    def capture(x,*args,**kwargs):
        print('ACTION_BOUNDARY',x.dtype,next(agent.model.action_model.parameters()).dtype,torch.is_autocast_enabled(),torch.get_autocast_dtype('cuda'),flush=True)
        captured.append(x.detach().clone())
        return original(x,*args,**kwargs)
    agent.model.action_model.predict_action=capture
    seed_all(123);agent.predict([samples[0]])
    legacy=captured[-1]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):new,_=policy.encode_conditions([samples[0]],include_world=False)
    difference=float((new.float()-legacy.float()).abs().max())
    report['checks']['shared_original_condition']={'legacy_dtype':str(legacy.dtype),'new_dtype':str(new.dtype),'max_abs':difference}
    try:torch.testing.assert_close(new.float(),legacy.float(),atol=.02,rtol=.02);report['checks']['shared_original_condition']['status']='PASS'
    except AssertionError as error:report['checks']['shared_original_condition'].update(status='FAIL',error=str(error))
    agent.model.action_model.predict_action=original
    clean={k:samples[0][k] for k in ['image','lang','state','token']}
    poisoned=dict(clean,WorldTargets={'future':'arbitrary'},world_targets=torch.randn(999),action=np.full((8,4),np.nan))
    seed_all(77);first=policy.predict_action([clean])
    seed_all(77);second=policy.predict_action([poisoned])
    np.testing.assert_array_equal(first['normalized_actions'],second['normalized_actions'])
    report['checks']['target_independence']='PASS'
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        single,_=policy.encode_conditions([samples[0]])
        batch,_=policy.encode_conditions(samples)
    difference=float((single.float()-batch[:1].float()).abs().max())
    try:torch.testing.assert_close(single.float(),batch[:1].float(),atol=.02,rtol=.02);status='PASS'
    except AssertionError:status='FAIL'
    report['checks']['padding']={'status':status,'max_abs':difference}
    # Strict reload of all newly introduced modules (original policy remains the same audited checkpoint).
    state={k:v.detach().clone() for k,v in policy.state_dict().items() if not k.startswith('baseline.')}
    clone=StructuredWorldPolicy(agent.model,policy.world_config).cuda().eval()
    for name,module in [('reader',clone.reader),('heads',clone.heads)]:
        module.load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')},strict=True)
    seed_all(77);reloaded=clone.predict_action([clean])
    np.testing.assert_array_equal(first['normalized_actions'],reloaded['normalized_actions'])
    report['checks']['new_modules_save_load']='PASS'
    # Real visual unfreezing: gradient traverses the live image encoder, no cached hidden state.
    policy.world_config['vision_trainable']=True
    visual=agent.model.qwen_vl_interface.model.model.visual
    visual.requires_grad_(True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _,pred=policy.encode_conditions([samples[0]])
        loss=pred['boxes'].float().square().mean()
    loss.backward()
    norm=sum(float(p.grad.float().square().sum()) for p in visual.parameters() if p.grad is not None)**.5
    report['checks']['visual_unfreeze']={'status':'PASS' if norm>0 else 'FAIL','gradient_norm':norm}
    if a.ledger:
        from budget import reserve,record
        reserve(a.ledger,'visual_unfreeze_update',1,vars(a))
        parameter=next(p for p in visual.parameters() if p.grad is not None and float(p.grad.abs().max())>0)
        before=parameter.detach().clone()
        optimizer=torch.optim.SGD(visual.parameters(),lr=.001)
        optimizer.step();record(a.ledger,'visual_unfreeze_update',1,status='complete')
        changed=float((parameter-before).abs().max())
        assert changed>0
        report['checks']['visual_unfreeze']['parameter_max_update']=changed
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
