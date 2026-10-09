"""Real cached-bank Router training and true pre-routing audit, isolated from serving."""
import argparse
from pathlib import Path
import torch
from iqe.config import load_config
from iqe.pipeline import Pipeline,stack_features
from iqe.data.sources import load_observation
from iqe.io import read_json,atomic_json
from iqe.resources import qualify


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    qualify('cpu');torch.set_num_threads(8)
    pipeline=Pipeline(load_config(a.config),mode='smoke',max_samples=32,device='cpu');pipeline.use_locked_base()
    model=pipeline.model(1)
    # Verify the lossless parallel official-validation implementation on the
    # exact real expert and locked scene list already evaluated serially.
    previous=read_json(pipeline.round_root(1)/'expert/validation_000002.json')
    with torch.no_grad():parallel=pipeline.expert_validation(1)(model.experts['expert_1'],2)
    assert previous==parallel
    result=pipeline.train_selector(1,router=True,steps=2)
    pipeline.calibrate(1,'selector_cal',router=True)
    evaluation=pipeline.evaluate(1,'dev_report',router=True)
    module,rule,_=pipeline.calibrated_selector(1,router=True)
    model.router=module;model.set_trainable_stage('inference');model.eval()
    calls={'shared_encoder':0,'decoder_rows':{eid:0 for eid in model.experts}}
    def shared_hook(*args):calls['shared_encoder']+=1
    def decoder_hook(eid):
        def record(module,args,output):calls['decoder_rows'][eid]+=len(args[0].scene)
        return record
    handles=[model.adapter.framework.action_model.q_former.register_forward_hook(shared_hook)]
    handles += [module.register_forward_hook(decoder_hook(eid)) for eid,module in model.experts.items()]
    scenes=pipeline.scenes(['stage_val'])
    with torch.no_grad():
        trajectory,ids=model.forward_prerouted(None,rule,features=stack_features([pipeline.cached(s) for s in scenes]))
    assert calls['shared_encoder']==0 and sum(calls['decoder_rows'].values())==len(scenes)
    assert all(calls['decoder_rows'][eid]==ids.count(eid) for eid in calls['decoder_rows'])
    with torch.no_grad():online,online_ids=model.forward_prerouted([load_observation(scenes[0],pipeline.contract)],rule)
    assert calls['shared_encoder']==1 and sum(calls['decoder_rows'].values())==len(scenes)+1
    assert torch.isfinite(online).all() and torch.isfinite(trajectory).all()
    for handle in handles:handle.remove()
    atomic_json(pipeline.root/'REAL_ROUTER.json',{'status':'PASS','optimizer_steps':result['optimizer_steps'],
        'real_old_new_bank':True,'selected_cached':ids,'selected_online':online_ids,'calls':calls,
        'parallel_validation_equals_original_serial':True,'selector_cal_only':True,
        'dev_report_selected_mean_01':evaluation['selected_mean_01'],'science':'UNTESTED','formal_active_updated':False})


if __name__=='__main__':main()
