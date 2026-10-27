"""Finite real8GPU qualification6 -> exact inner resume16 -> fresh formal50k.

One owner per registered scope. Native admission, checkpoint transactions and
trainer are reused. No workload is evicted, no failed run is silently retried.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import allowed_hosts,file_sha
from starVLA.rl.qwen_world_postcot.resources import MODEL_PYTHON,OPERATOR_PYTHON,probe


def verify_qualification(root,cfg):
    import torch
    from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY
    identity=json.loads((root/'identity.json').read_text());rows=[]
    if identity['purpose']!='qualification_not_formal_initialization':raise ValueError('Wrong qualification purpose')
    common=torch.load(root/'checkpoints/update000016_inner0/optimizer.pt',map_location='cpu',weights_only=False)
    names=[n for n in identity['parameter_contract']['trainable_names']]
    # Parameter order is taken from the exact registered optimizer groups, not
    # assumed to match alphabetic closure names.
    source_state={}
    for rank in range(8):
        result=json.loads((root/f'result_rank{rank}.json').read_text())
        if (not result['qualification'] or result['updates']!=16 or result['collections']!=4
                or result['new_candidates']!=256 or result['status']!='STOPPED_AT_SAVED_BOUNDARY'
                or result['frozen_SHA_before']!=result['frozen_SHA_after']
                or result['frozen_observation_before']!=result['frozen_observation_after']):
            raise ValueError('Real scope/freeze qualification failed on rank'+str(rank))
        parity=json.loads((root/f'initial_ODE_parity_rank{rank}.json').read_text())
        if parity['status']!='PASS' or parity['normalized_full_chain_max_abs']!=0:
            raise ValueError('Scope zero-init no longer matches actual IL ODE')
        records=[json.loads(v) for v in (root/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        if [v['updates'] for v in records]!=list(range(1,17)):raise ValueError('Resume journal is not continuous')
        for row in records:
            if row['collection']!=(row['updates']-1)//4+1:raise ValueError('A repeated inner became a fresh rollout')
        held=[row for row in records if row['updates'] in (5,6,7,8)]
        if any(row['chain_identity']!=held[0]['chain_identity'] for row in held):
            raise ValueError('Mid-inner resume regenerated behavior or old probabilities')
        for p in (root/'behavior').glob(f'collection*/rank{rank}.pt'):
            behavior=torch.load(p,map_location='cpu',weights_only=False)
            if behavior['qualification_sampling_check']['status']!='PASS':raise ValueError('Actual mean/std mismatch')
        rows.append(result)
    adapters=common['adapters']
    if set(adapters)!=set(names):raise ValueError('Checkpoint lost a trainable memory/LoRA tensor')
    groups=common['optimizer']['param_groups'];moments=common['optimizer']['state']
    task_signals={g['name']:sum(bool(torch.count_nonzero(moments[p]['exp_avg'])) for p in g['params']) for g in groups}
    if any(n==0 for n in task_signals.values()):raise ValueError('A complete parameter group has no actual task gradient')
    if cfg['adaptation']==MEMORY:
        for result in rows:
            for prefix in ('scene_provider.','predictive_provider.'):
                if not any(changed for n,changed in result['changed_adapters'].items() if n.startswith(prefix)):
                    raise ValueError('A current memory provider did not change')
    return dict(status='PASS',adaptation=cfg['adaptation'],updates=16,fresh_global_rollouts=4,new_candidates=256,
        mid_inner_resume='update6 inner2 -> update7/8 same chain and old probability; complete16',
        initial_IL_full_ODE_parity='exact on all8ranks',sampling_recompute_mean_std='exact7/8/9',
        frozen_Qwen_and_permanent_reference='unchanged on all8ranks',
        task_moment_nonzero_tensors_by_group=task_signals,
        changed_tensors_by_rank=[sum(r['changed_adapters'].values()) for r in rows],
        trainable_numel=identity['parameter_contract']['trainable_numel'],
        trainable_tensors=len(names),peak_GPU_GiB=max(r['peak_GPU_bytes'] for r in rows)/2**30,
        qualification_weights_used_for_formal_initialization=False)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--registration',required=True);a=parser.parse_args()
    cfg=json.loads(Path(a.registration).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(root/'OWNER_RESULT.json').exists():raise ValueError('Finished attempt must not be repeated')
    source=Path(cfg['source']);host=cfg['host'];scope=cfg['adaptation']
    if(file_sha(source/cfg['config'])!=cfg['recipe_sha256'] or file_sha(cfg['parent_selection'])!=cfg['parent_selection_file_sha256']):
        raise ValueError('Registered config or parent selection changed')
    selected=json.loads(Path(cfg['parent_selection']).read_text())['selected']
    state=dict(status='STARTING',pid=os.getpid(),start_unix=time.time(),adaptation=scope,stages=[],formal_run=cfg['formal_run'])
    def publish():atomic_json(root/'OWNER_STATE.json',state)
    publish()
    def run_stage(name,command):
        if host not in allowed_hosts(admit_long_job=True):raise PermissionError('Host permission changed')
        receipt=root/'jobs'/name
        if receipt.exists():raise FileExistsError('Do not repeat an invocation')
        first=probe(host);time.sleep(2);second=probe(host)
        physical=[g['uuid'] for g in second['gpus']]
        if physical!=cfg['gpu_uuids'] or physical!=[g['uuid'] for g in first['gpus']]:raise ValueError('GPU mapping changed')
        if any(g['free_MiB']<cfg['minimum_free_MiB'] for g in second['gpus']):raise RuntimeError('Measured headroom changed')
        receipt.mkdir(parents=True)
        row=dict(stage=name,status='RUNNING',host=host,command=command,receipt=str(receipt),dispatch_unix=time.time())
        atomic_json(receipt/'INTENT.json',row);state['status']=name;state['stages'].append(row);publish()
        worker=[OPERATOR_PYTHON,'-m','tools.qwen_world_postcot.job_worker','--receipt-root',str(receipt),
            '--gpus','0,1,2,3,4,5,6,7','--expected-gpu-uuids-json',json.dumps(physical),
            '--minimum-free-mib',str(cfg['minimum_free_MiB']),'--command-json',json.dumps(command),'--cwd',str(source)]
        if cfg.get('share_min_free_mib'):
            worker+=['--share-min-free-mib',str(cfg['share_min_free_mib'])]
        invocation=['env','PYTHONPATH='+str(source),'CUBLAS_WORKSPACE_CONFIG=:4096:8',*worker]
        if host!=socket.gethostname().removesuffix('-worker-0'):
            invocation=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(invocation)]
        with(receipt/'transport.log').open('ab',buffering=0)as log:
            child=subprocess.Popen(invocation,cwd=source,stdout=log,stderr=subprocess.STDOUT)
            row['transport_pid']=child.pid;publish();code=child.wait()
        result=json.loads((receipt/'RESULT.json').read_text()) if(receipt/'RESULT.json').exists() else None
        row.update(returncode=code,result=result,end_unix=time.time(),status='EXITED' if code==0 and result and result['returncode']==0 else 'FAILED')
        publish()
        if row['status']!='EXITED':raise RuntimeError('Native stage failed; sealed boundary/failure evidence retained: '+name)
    base=[MODEL_PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','-m','tools.qwen_world_postcot.train_scope_rl',
        '--config',cfg['config'],'--manifest',cfg['manifest'],'--data',cfg['data'],'--il-checkpoint',selected['checkpoint']]
    qual=Path(cfg['qualification_run'])
    try:
        mid=qual/'checkpoints/update000006_inner2'
        if cfg.get('qualification_resume_checkpoint'):
            previous=Path(cfg['prior_resource_refusal']);refused=json.loads(previous.read_text())
            if (Path(cfg['qualification_resume_checkpoint']).resolve()!=mid.resolve()
                    or refused.get('status')!='REFUSED_RESOURCE' or refused.get('child_not_started') is not True
                    or cfg['qualification_resume_files_sha256']!=file_sha(mid/'files.json')):
                raise ValueError('Continuation requires exact legal checkpoint and positive pre-child refusal evidence')
            atomic_json(root/'CONTINUATION_RECEIPT.json',dict(status='RESOURCE_ONLY_CONTINUATION',
                previous_refusal=str(previous),previous_refusal_sha256=file_sha(previous),checkpoint=str(mid),
                checkpoint_files_sha256=file_sha(mid/'files.json'),training_source_unchanged=str(source),
                optimizer_RNG_old_chain_and_advantages_retained=True))
        else:
            run_stage('QUAL6',base+['--output',str(qual),'--qualification','--stop-after-update','6','--save-every','6'])
            archived={str(rank):json.loads((qual/f'result_rank{rank}.json').read_text()) for rank in range(8)}
            atomic_json(root/'QUAL6_RESULTS.json',archived)
        seal=json.loads((mid/'COMPLETE').read_text())
        if seal['updates']!=6 or seal['next_inner']!=2 or seal['phase']!='INNER_COMMITTED':raise ValueError('No legal mid-inner checkpoint')
        run_stage('QUAL16_RESUME',base+['--output',str(qual),'--qualification','--stop-after-update','16','--save-every','8','--resume',str(mid)])
        acceptance=verify_qualification(qual,cfg);atomic_json(root/'RL_ENGINEERING_ACCEPTANCE.json',acceptance)
        run_stage('FORMAL50000',base+['--output',cfg['formal_run'],'--parent-selection',cfg['parent_selection']])
        results=[json.loads((Path(cfg['formal_run'])/f'result_rank{rank}.json').read_text()) for rank in range(8)]
        if any(r['status']!='COMPLETE' or r['qualification'] or r['updates']!=50000 or r['collections']!=12500 for r in results):
            raise ValueError('Formal terminal budget incomplete')
        state.update(status='COMPLETE',end_unix=time.time());publish();atomic_json(root/'OWNER_RESULT.json',state)
    except BaseException as error:
        state.update(status='FAILED_REVIEW_REQUIRED',error=repr(error),end_unix=time.time());publish();atomic_json(root/'OWNER_RESULT.json',state);raise

if __name__=='__main__':main()
