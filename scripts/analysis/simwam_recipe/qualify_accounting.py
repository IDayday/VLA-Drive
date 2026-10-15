"""CPU checks for resolved recipe locking and resumed physical cost accounting."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from omegaconf import OmegaConf
from scripts.analysis.simwam_recipe.cost_ledger import invocation_cost,evaluation_cost
from scripts.analysis.simwam_recipe.summarize_training import read_journal
from starVLA.rl.simwam_recipe.contracts import load_recipe
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args();checks=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);cfg=load_recipe('configs/flow_rl/simwam_source_ddp.yaml')
        canonical=load_recipe('configs/flow_rl/simwam_canonical_ddp.yaml');canonical['variant']='R_SOURCE';canonical['grpo']['reward']['progress_weight']=10.
        assert canonical==cfg;checks.append('exact_two_allowed_variant_differences')
        for key in ('learning_rate','lr_warmup_steps','max_steps','weight_decay','optimizer_eps',
                    'gradient_clipping_effective','DDP_bucket_cap_mb','seed','evaluation_seed','action_infer_shift'):
            changed=copy.deepcopy(cfg);changed[key]+=1
            path=root/'recipe.yaml';OmegaConf.save(OmegaConf.create(changed),path)
            try:load_recipe(path)
            except ValueError:checks.append('locked_'+key)
            else:raise AssertionError('Unlocked '+key)
        for group,key in (('lora','r'),('lora','alpha'),('train','kl_beta'),('train','num_inner_epochs'),('train','adv_eps'),('sample','min_std'),('sample','group_size'),('reward','ttc_weight')):
            changed=copy.deepcopy(cfg);changed['grpo'][group][key]+=1
            path=root/'recipe.yaml';OmegaConf.save(OmegaConf.create(changed),path)
            try:load_recipe(path)
            except ValueError:checks.append('locked_'+group+'.'+key)
            else:raise AssertionError('Unlocked '+key)
        log=root/'metrics.jsonl';log.write_text('{"updates":1}\n{"updates":1}\n')
        try:read_journal(log)
        except ValueError:checks.append('duplicate_updates_rejected')
        else:raise AssertionError('Silent deduplication')
        records=[dict(rank=0,status='ABORTED',start_unix=10,start_update=0,end_update=8,
                      worker_wall_seconds=20,seconds_save=1,generated_candidates=32,scored_candidates=24),
                 dict(rank=0,status='COMPLETE',start_unix=40,start_update=4,end_update=12,
                      worker_wall_seconds=4,seconds_save=1,generated_candidates=16,scored_candidates=16)]
        for i,r in enumerate(records):
            folder=root/'invocations'/str(i);folder.mkdir(parents=True)
            atomic_json(folder/'rank0.json',r)
        cost=invocation_cost(root,done=12)
        assert cost['invocation_effective_updates']==8 and cost['invocation_updates_per_second']==2
        assert cost['cumulative_worker_wall_seconds']==24 and cost['aborted_invocation_worker_wall_seconds']==20
        assert cost['physical_generated_candidates']==48;checks.append('resumed_delta_rate_and_failed_work_counted')
        evalroot=root/'eval'
        for i,r in enumerate([dict(start_unix=0,end_unix=12,status='ABORTED',new=1,reused=0,physical_predictions=4,total_worker_wall_seconds=12,model_load_seconds=3,generation_seconds=6,scoring_seconds=3),
                              dict(start_unix=20,end_unix=30,status='ASSIGNED_COMPLETE',new=2,reused=1,physical_predictions=4,total_worker_wall_seconds=10,model_load_seconds=2,generation_seconds=4,scoring_seconds=4)]):
            folder=evalroot/'invocations'/str(i);folder.mkdir(parents=True)
            atomic_json(folder/'rank0.json',r)
        cost=evaluation_cost(evalroot,dict(scenes=3,modes=['E_NATIVE','E_SHIFT5']),[])
        assert cost['logical_single_candidate_evaluations']==6 and cost['cumulative_physical_predictions']==8
        assert cost['cumulative_worker_wall_seconds']==22 and cost['job_wall_seconds']==30 and cost['model_load_seconds']==5
        assert not cost['complete_cost_accounting']
        checks.append('evaluation_resume_preserves_loading_retry_worker_and_job_time')
        legacy=invocation_cost(root/'absent',done=10000,latest_wall=100)
        assert legacy['invocation_updates_per_second'] is None;checks.append('missing_legacy_cost_is_unknown_not_fabricated')
    result=dict(status='PASS',checks=checks);atomic_json(Path(a.output),result);print(json.dumps(result))


if __name__=='__main__':main()
