#!/usr/bin/env python
"""Run bounded real-image CPU smoke from an actually trained diagnostic Query base.

This does not qualify GPU throughput, statistical gain, or a formal deployment.
"""
import argparse
from pathlib import Path
import torch
from iqe.config import load_config
from iqe.pipeline import Pipeline
from iqe.rounds import run_round
from iqe.io import atomic_json,read_json
from iqe.resources import qualify


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--steps',type=int,default=2)
    args=p.parse_args();config=load_config(args.config)
    assert 1<=args.steps<=32
    qualify('cpu');torch.set_num_threads(config['execution']['num_threads'])
    pipeline=Pipeline(config,mode='smoke',max_samples=32,device='cpu');pipeline.use_locked_base()
    def mark(stage):
        atomic_json(pipeline.root/'PROGRESS.json',{'stage':stage,'science':'UNTESTED','formal_active_updated':False})
        print(stage,flush=True)
    mark('real_image_feature_cache')
    pipeline.cache_features(['incremental_fit','stage_val','selector_cal','dev_report'])
    mark('base_singleton_and_online_cached_parity')
    from iqe.data.sources import load_observation
    from iqe.evaluation.retention import audit_frozen
    scene=pipeline.scenes(['stage_val'])[0]
    model=pipeline.model()
    f=pipeline.cached(scene).to('cpu')
    with torch.no_grad():
        original=model.adapter.framework.predict_action([load_observation(scene,pipeline.contract)])
        wrapped=model.forward_candidates(None,features=f)
    error=float((original-wrapped.trajectories[:,0]).abs().max())
    assert error<=config['tolerances']['fp32_cpu']['physical_max_abs']
    atomic_json(pipeline.root/'REAL_K1.json',{'physical_max_abs':error,'real_images':True,'checkpoint':pipeline.contract['query_checkpoint']})
    for number in (1,2):
        mark('round_'+str(number))
        run_round(pipeline,number,resume=True,steps=args.steps)
    mark('bundle_reload_and_prediction_check')
    from iqe.export import load_bundle
    loaded,rule,metadata=load_bundle(pipeline.round_root(2)/'bundle',shared_adapter=model.adapter)
    with torch.no_grad():out,ids,_=loaded.predict(None,rule,features=f)
    assert torch.isfinite(out).all()
    mark('real_NAVSIM_Agent_input_and_trajectory')
    from iqe.evaluation.agent_audit import real_agent_input
    from iqe.agent import IQEAgent
    agent_input,agent_evidence=real_agent_input(scene,pipeline.contract,pipeline.config['data']['raw_log_root'])
    agent=IQEAgent(str(pipeline.round_root(2)/'bundle'),device='cpu',allow_smoke=True)
    agent.model,agent.rule,agent.metadata=loaded,rule,metadata
    actual=agent.compute_trajectory(agent_input)
    agent_error=float((torch.from_numpy(actual.poses)-out[0].cpu()).abs().max())
    assert agent_error<=config['tolerances']['fp32_cpu']['physical_max_abs']
    atomic_json(pipeline.root/'REAL_AGENT.json',agent_evidence|{'physical_max_abs':agent_error,'finite':True})
    mark('optimizer_boundary_restore')
    # Completed stage resume must verify/reuse every artifact, including current bank.
    resumed=run_round(pipeline,2,resume=True,steps=args.steps)
    assert all(s['status']=='REUSED' for s in resumed['stages'])
    result={'status':'COMPLETE','science':'UNTESTED','base_optimizer_steps':pipeline.contract['base_optimizer_steps'],
        'expert_and_scorer_steps_each':args.steps,'rounds':2,'real_images':True,'real_official_scores':True,
        'selected_expert':ids,'GPU_used':False,'formal_active_updated':False,'role_counts':{r:len(pipeline.scenes([r])) for r in ('incremental_fit','stage_val','selector_cal','dev_report')},
        'stage_resume':resumed,'bundle':str(pipeline.round_root(2)/'bundle')}
    atomic_json(pipeline.root/'REAL_TWO_ROUND_RESULT.json',result)
    print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':main()
