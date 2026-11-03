"""Parallel CPU aggregation of fully finished, immutable official evaluations.

Only validated file reads are memoized. Official scene validation, canonical
summary, bootstrap and paired analysis remain unchanged. No GPU or scoring.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import concurrent.futures
import fcntl
import json
import multiprocessing
from pathlib import Path
import time
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

BATCH=Path('/mnt/project/d46-priority-rl-20261103/operations/EARLY_EVAL_20261103')


def read(p):return json.loads(Path(p).read_text())


def aggregate(spec,baseline):
    import torch
    torch.set_num_threads(1)
    from tools.qwen_world_compare import evaluate as official
    original=official.load_evaluation;cache={}
    def validated_once(root,*,population=None):
        if population is not None:raise ValueError('Use the immutable registered population')
        key=str(Path(root).resolve())
        if key not in cache:cache[key]=original(root)
        return cache[key]
    official.load_evaluation=validated_once
    # Imported after memoization so both consumers reuse the exact validated
    # immutable rows, rather than re-reading/re-hashing them three times.
    from tools.d44_native_rl.analyze import analyze
    output=Path(spec['output']);summary=official.summarize(output,baseline)
    assert summary['status']=='COMPLETE'and summary['scenes']==12146 and summary['physical_logs']==44
    assert summary['invalid_scenes']==0 and summary['qualification']is False
    existing=output.parent/'SUMMARY.json'
    if existing.exists()and read(existing)!=summary:
        raise ValueError('Official summary changed; preserve the original')
    if not existing.exists():atomic_json(existing,summary)
    paired=analyze(output,baseline);paired['analysis_schema']='D46_complete_B1_paired_gain_loss_v1'
    gain=paired['gain_loss'];gain['fixed_reference_cohorts']={k.replace('IL44','IL46'):v for k,v in gain['fixed_reference_cohorts'].items()}
    gain['limits']=gain['limits'].replace('IL44','IL46');paired.pop('sha256',None);paired['sha256']=fingerprint(paired)
    target=output.parent/'PAIRED_ANALYSIS.json'
    if target.exists()and read(target)!=paired:raise ValueError('Existing paired analysis changed')
    if not target.exists():atomic_json(target,paired)
    return dict(summary=str(existing),paired_analysis=str(target),PDMS=summary['PDMS'],validated_dataset_loads=len(cache))


def main():
    plan=read(BATCH/'PLAN.json');state=read(BATCH/'STATE.json')
    body=dict(plan);digest=body.pop('sha256');assert fingerprint(body)==digest==state['plan_sha256']
    with(BATCH/'OWNER.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for spec in plan['cases']:
            case=state['cases'][spec['id']]
            for rank in range(plan['world']):
                jobs=case['jobs'].get(str(rank),[]);assert jobs
                assert read(Path(jobs[-1]['job'])/'RESULT.json')['returncode']==0
                assert read(Path(spec['output'])/f'rank{rank}.json')['status']=='COMPLETE'
        state.update(owner_pid=os.getpid(),aggregation='parallel_CPU_immutable_validated_read_cache',at=time.time())
        atomic_json(BATCH/'STATE.json',state)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:
            futures={pool.submit(aggregate,spec,plan['baseline']):spec['id']for spec in plan['cases']}
            for future in concurrent.futures.as_completed(futures):
                key=futures[future];value=future.result()
                state['cases'][key].update(status='COMPLETE',completed_shards=plan['world'],active_shards=0,**value)
                state.update(at=time.time(),complete=all(c['status']=='COMPLETE'for c in state['cases'].values()))
                atomic_json(BATCH/'STATE.json',state)
                print(json.dumps(dict(case=key,status='COMPLETE',PDMS=value['PDMS'])),flush=True)
        state['terminal']=True;atomic_json(BATCH/'STATE.json',state)


if __name__=='__main__':main()
