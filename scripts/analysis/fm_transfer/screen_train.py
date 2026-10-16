"""Own completed SFT, one native seed2101 trajectory per locked TRAIN scene.

Frozen observation-cache inference is independently qualified against the raw
entry. This screen only fixes source-recipe difficult-scene membership; it is
never reported as raw-observation DEV/Navtest deployment evaluation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import uuid
import torch
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.fm_transfer.rl_adapter import load_supervised_policy
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.simwam_recipe.contracts import RAW_CONDITION, CONDITION_SHA, POPULATION, seed_for, file_sha, fingerprint
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request

STOP=False
def stop(*_):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--sft-checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=24);p.add_argument('--max-scenes',type=int)
    p.add_argument('--qualification',action='store_true')
    a=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    if not 0<=a.shard<a.shards or (a.qualification and (a.max_scenes is None or not 1<=a.max_scenes<=8)):
        raise ValueError('Invalid shard or unbounded qualification')
    if a.max_scenes is not None and not a.qualification:raise ValueError('Partial screens are qualification only')
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'rank{a.shard}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    invocation=uuid.uuid4().hex;receipt=out/'invocations'/invocation/f'rank{a.shard}.json'
    receipt.parent.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    status=dict(schema='fm_transfer_screen_invocation_v1',status='STARTING',host=host,pid=os.getpid(),
        invocation=invocation,rank=a.shard,start_unix=time.time(),new=0,reused=0,
        actual_new_inferences=0,model_load_seconds=0.,generation_seconds=0.,scoring_wait_seconds=0.,
        worker_scoring_seconds=0.,qualification=a.qualification)
    atomic_json(receipt,status);service=None;pending=[]
    try:
        begin=time.monotonic()
        policy,sft=load_supervised_policy(a.sft_checkpoint,qualification=a.qualification)
        policy.eval().requires_grad_(False)
        store=ConditionStore(RAW_CONDITION)
        if store.identity_sha!=CONDITION_SHA:raise ValueError('Changed Qwen execution')
        status['model_load_seconds']=time.monotonic()-begin
        pop=json.loads(Path(POPULATION).read_text());tokens=pop['train_tokens']
        if set(tokens)&(set(pop['dev_tokens'])|set(pop['navtest_tokens'])):raise ValueError('Split leakage')
        service=RewardService(Path.cwd(),workers=1)
        # Same original decode, independently checked by the raw deployment gate.
        from omegaconf import OmegaConf
        from starVLA.rl.constrained_flow_noise.loading import DEFAULT_SOURCE
        act_norm=int(OmegaConf.load(DEFAULT_SOURCE).datasets.vla_data.act_norm)
        contract=dict(SFT_model_sha256=sft['model_sha256'],SFT_identity_sha256=sft['identity_sha256'],
            architecture=sft['config']['architecture'],seed=2101,grid='native',threshold=.9,
            statistic='one canonical v1.1 score per scene')
        identity=dict(schema='fm_transfer_single_native_TRAIN_screen_v1',new_SFT=sft,
            screen_contract=contract,population_identity=pop['identity_sha256'],
            tokens_sha256=fingerprint(tokens),scene_count=len(tokens),condition_identity=store.identity_sha,
            protocol=service.protocol,root_namespace='original_ode_initial',root_dtype='torch.bfloat16',
            action_encoding='original eight-by-four',action_batch=1,steps=10,act_norm=act_norm,
            shards=a.shards,qualification_only=a.qualification,source_sha256=file_sha(__file__))
        identity['sha256']=fingerprint(identity)
        with (out/'manifest.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX)
            if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=identity:
                raise ValueError('Changed screen population/model/protocol')
            atomic_json(out/'manifest.json',identity)
        status.update(status='RUNNING',identity_sha256=identity['sha256']);atomic_json(receipt,status)
        assigned=tokens[a.shard::a.shards]
        if a.qualification:assigned=assigned[:a.max_scenes]
        def publish(item):
            token,action,requested,future=item
            wait=time.monotonic();records=future.result();status['scoring_wait_seconds']+=time.monotonic()-wait
            if len(records)!=1 or records[0]['status']!='OK':raise ValueError('A screen needs exactly one valid candidate; no error-as-zero')
            status['worker_scoring_seconds']+=sum(r['scoring_seconds'] for r in records)
            path=out/'scenes'/token[:2]/(token+'.pt')
            atomic_torch(path,dict(token=token,log=pop['token_logs'][token],identity_sha256=identity['sha256'],
                normalized_actions=action,requested_trajectory=requested,records=records,invocation=invocation))
            atomic_json(path.with_suffix('.json'),dict(token=token,log=pop['token_logs'][token],
                identity_sha256=identity['sha256'],payload_sha256=file_sha(path),metrics=records[0]['metrics']))
            status['new']+=1
            if status['new']%16==0:
                atomic_json(receipt,status);atomic_json(out/f'rank{a.shard}.json',status)
        for token in assigned:
            if STOP:break
            require_dispatch();path=out/'scenes'/token[:2]/(token+'.pt')
            if path.with_suffix('.json').exists():
                seal=json.loads(path.with_suffix('.json').read_text())
                if seal['identity_sha256']!=identity['sha256'] or file_sha(path)!=seal['payload_sha256']:
                    raise ValueError('Changed existing screen result')
                status['reused']+=1;continue
            begin=time.monotonic();raw=store.get(token).cuda().float()
            key=seed_for('original_ode_initial',2101,token)
            noise=torch.randn((1,8,4),device='cuda',dtype=torch.bfloat16,
                generator=torch.Generator(device='cuda').manual_seed(key))
            with torch.no_grad(),torch.autocast('cuda',enabled=False):action=policy.sample_ode(raw,noise,grid='native')
            torch.cuda.synchronize();status['generation_seconds']+=time.monotonic()-begin
            if not torch.isfinite(action).all():raise ValueError('Nonfinite native trajectory; retain failure and stop')
            status['actual_new_inferences']+=1
            action=action.cpu();requested=decode_request(action.numpy(),act_norm=act_norm)
            pending.append((token,action,requested,service.submit((token,pop['token_logs'][token],requested))))
            if len(pending)>=4:
                publish(pending[0]);pending.pop(0)
        while pending:
            publish(pending[0]);pending.pop(0)
        status['status']='STOPPED_AT_SCENE_BOUNDARY' if STOP else 'QUALIFICATION_COMPLETE' if a.qualification else 'ASSIGNED_COMPLETE'
    except BaseException:
        import traceback
        status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        # Failed/unpublished requests still consumed real inference/scoring.
        # Keep their identities; never hide them in the logical scene count.
        status['unpublished_scoring_requests']=[item[0] for item in pending]
        if service is not None:service.close()
        status.update(end_unix=time.time(),total_worker_wall_seconds=time.monotonic()-started)
        atomic_json(receipt,status);atomic_json(out/f'rank{a.shard}.json',status)


if __name__=='__main__':main()
