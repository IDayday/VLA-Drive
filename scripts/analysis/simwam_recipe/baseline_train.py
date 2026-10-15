"""Fill only missing single-native-ODE TRAIN scores for the fixed hard subset.

G16 census is never used to decide membership. One scene-bound seed2101 root,
original uniform 10-step Euler, fixed SFT and canonical isolated scoring.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import torch
from starVLA.rl.constrained_flow_noise.loading import load_action_closure
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from starVLA.rl.simwam_recipe.contracts import *
from starVLA.rl.simwam_recipe.condition import ProjectedConditions
from starVLA.rl.simwam_recipe.reward_readouts import RewardService

STOP=False
def stop(*_):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',default=ROOT);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=24);p.add_argument('--max-new',type=int)
    p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=1)
    p.add_argument('--qualify-only',action='store_true',help='Only existing original-observation seed2101 scenes; require exact equality')
    a=p.parse_args();host=require_dispatch();configure_numerics()
    for s in (signal.SIGUSR1,signal.SIGTERM):signal.signal(s,stop)
    torch.cuda.set_device(0)
    root=Path(a.root);out=root/'SFT_NATIVE_TRAIN_seed2101';out.mkdir(parents=True,exist_ok=True)
    if not 0<=a.lane<a.lanes:raise ValueError('Invalid deterministic subpartition')
    name=f'rank{a.shard:02}' if a.lanes==1 else f'rank{a.shard:02}_lane{a.lane}of{a.lanes}'
    lock=(out/f'shard{a.shard:02}.lock').open('a')
    fcntl.flock(lock,(fcntl.LOCK_EX if a.lanes==1 else fcntl.LOCK_SH)|fcntl.LOCK_NB)
    lane_lock=(out/(name+'.lock')).open('a');fcntl.flock(lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pop=json.loads(Path(POPULATION).read_text());tokens=pop['train_tokens']
    if not 0<=a.shard<a.shards or set(tokens)&(set(pop['dev_tokens'])|set(pop['navtest_tokens'])):raise ValueError('Invalid TRAIN population')
    policy,cfg,sft=load_action_closure(dtype=torch.float32);policy.eval().requires_grad_(False)
    head=policy.action_model
    store=ProjectedConditions(root/'projected_condition',head)
    service=RewardService(Path.cwd(),workers=1)
    identity=dict(schema='simwam_SFT_hard_screen_native_v1',SFT_sha256=SFT_SHA256,scene_population=pop['identity_sha256'],
        tokens_sha256=fingerprint(tokens),seed=2101,root_namespace='original_ode_initial',root_dtype='bfloat16',
        sampler='unchanged_original_10_step_uniform_ODE',condition=store.sha,score=service.protocol['sha256'],
        no_group_aggregation=True,hard_threshold=.9,shards=a.shards)
    identity['sha256']=fingerprint(identity)
    with (out/'manifest.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=identity:raise ValueError('Screen identity changed')
        atomic_json(out/'manifest.json',identity)
    # Legitimate prior single-candidate seed2101 scores are reused only after
    # exact native output equality; older all-BF16 caches/G16 are never reused.
    old=Path('/mnt/project/flow-safe-improvement-20261013/evaluations/original_export')
    sample_root=old/'SFT_TRAIN_D5_samples';score_root=old/'SFT_TRAIN_D5_v1'
    old_sample=json.loads((sample_root/'manifest.json').read_text())
    old_scores=json.loads((score_root/'manifest.json').read_text())
    assert old_sample['behavior_sha256']==SFT_SHA256 and old_sample['mode']=='ode' and old_sample['seeds'][0]==2101
    assert old_scores['protocol_sha256']==service.protocol['sha256']
    started=time.monotonic();fresh=reused=old_reused=0;checks=[];pending=[]
    def publish(token,action,records,elapsed,reuse=None):
        nonlocal fresh,old_reused
        if len(records)!=1:raise ValueError('Exactly one candidate required')
        path=out/'scenes'/token[:2]/(token+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(token=token,log=pop['token_logs'][token],action=action.cpu(),records=records,
                     identity_sha256=identity['sha256'],generation_seconds=elapsed,reused_source=reuse)
        tmp=path.with_suffix('.incomplete');torch.save(payload,tmp);os.replace(tmp,path)
        atomic_json(path.with_suffix('.json'),dict(token=token,log=pop['token_logs'][token],identity_sha256=identity['sha256'],
                    payload_sha256=file_sha(path),metrics=records[0]['metrics'],reused_source=reuse))
        if reuse:old_reused+=1
        else:fresh+=1
    try:
        assigned=tokens[a.shard::a.shards][a.lane::a.lanes]
        if a.qualify_only:
            assigned=[t for t in assigned if (sample_root/'scenes'/t[:2]/(t+'.pt')).exists()][:4]
            if len(assigned)<2:raise ValueError('Need at least two original-observation comparison scenes')
        for token in assigned:
            if STOP or (a.max_new is not None and fresh+old_reused>=a.max_new):break
            require_dispatch()
            path=out/'scenes'/token[:2]/(token+'.pt')
            if path.with_suffix('.json').exists():
                seal=json.loads(path.with_suffix('.json').read_text())
                if seal['identity_sha256']!=identity['sha256'] or seal['payload_sha256']!=file_sha(path):raise ValueError('Changed screen record')
                reused+=1;continue
            t=time.monotonic();condition=store.get(token)
            key=seed_for('original_ode_initial',2101,token)
            root_noise=torch.randn((1,8,4),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(key))
            # Match original VLAAgent exactly. This environment really enables
            # CUDA FP32 autocast, including casting the BF16 initial root for
            # the first Linear. enabled=False is NOT equivalent for this input.
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float32):action=head._euler_sample(root_noise,condition).float()
            if not torch.isfinite(action).all():raise ValueError('Nonfinite native policy')
            elapsed=time.monotonic()-t
            previous=sample_root/'scenes'/token[:2]/(token+'.pt')
            if previous.exists():
                seal=json.loads(previous.with_suffix('.json').read_text());assert seal['payload_sha256']==file_sha(previous)
                row=torch.load(previous,map_location='cpu',weights_only=True)
                if not torch.equal(action.cpu(),row['normalized_actions'][:1]):raise ValueError('Native cache/original observation mismatch')
                prior=score_root/'scenes'/token[:2]/(token+'.pt');sseal=json.loads(prior.with_suffix('.json').read_text())
                assert sseal['payload_sha256']==file_sha(prior)
                scored=torch.load(prior,map_location='cpu',weights_only=True);records=scored['feedback'][:1]
                checks.append(dict(token=token,exact_original_observation_output=True))
                publish(token,action,records,elapsed,str(prior))
            else:
                requested=decode_request(action.cpu().numpy(),act_norm=int(sft.datasets.vla_data.act_norm))
                pending.append((token,action.cpu(),service.submit((token,pop['token_logs'][token],requested)),elapsed))
            if len(pending)>=2:
                tok,act,future,sec=pending.pop(0);publish(tok,act,future.result(),sec)
            if (fresh+old_reused)%16==0:
                atomic_json(out/(name+'.json'),dict(status='RUNNING',pid=os.getpid(),host=host,lane=a.lane,lanes=a.lanes,
                    new_scores=fresh,reused_in_place=reused,reused_original_scores=old_reused,elapsed=time.monotonic()-started))
        for tok,act,future,sec in pending:publish(tok,act,future.result(),sec)
        atomic_json(out/(name+'.json'),dict(status='QUALIFICATION_COMPLETE' if a.qualify_only else ('PAUSED' if STOP or a.max_new else 'ASSIGNED_COMPLETE'),
            pid=os.getpid(),host=host,new_scores=fresh,reused_in_place=reused,reused_original_scores=old_reused,
            lane=a.lane,lanes=a.lanes,execution_source_sha256=file_sha(__file__),
            exact_original_comparisons=checks,elapsed=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    finally:service.close()


if __name__=='__main__':main()
