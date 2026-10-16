"""Finite CPU/Gloo failure-injection tests of the real checkpoint code."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
# This entry is explicitly CPU-only, including RNG bookkeeping. Do not let
# the generic checkpoint helper initialize an unrelated local CUDA context.
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from starVLA.rl.simwam_recipe import checkpoint as ck
from starVLA.rl.simwam_recipe.recovery_state import validate_boundary,validate_ranks,boundary_phase
from starVLA.rl.simwam_recipe.source_advantages import advantages
from starVLA.rl.simwam_recipe.invocation import Invocation
from starVLA.rl.flow_grpo.transactions import atomic_json


def fixture(rank,world):
    identity=dict(sha256='qualification-identity',world=world,recipe={'max_steps':20000,'variant':'R_SOURCE'})
    state=dict(updates=0,collections=0,next_inner=0,scene_cursor=0,raw_scene_exposures=0,new_candidates=0,phase='READY')
    model=torch.nn.Linear(2,1);model.device=torch.device('cpu')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    gen=torch.Generator().manual_seed(42+rank)
    return identity,state,model,opt,sched,gen


def rollout(rank,world,identity):
    # Shape/probability contracts match the production payload; policy math is mocked.
    raw=torch.zeros(8,3,8,4)
    r=dict(cond_bg={'context':torch.ones(8,1,2)},chain=torch.zeros(8,11,8,4),
        timesteps=torch.arange(10),deltas=torch.ones(10),stoch_idx=torch.tensor([7,8,9]),
        old_logp=raw.mean((-1,-2)),old_probability={'source_clamped_logprob':raw},
        rewards=torch.arange(8).float()[None,:]/8+rank*.01,records=[{}]*8,token='t'+str(rank),log='log',
        behavior_version=dict(run=identity['sha256'],update=0,adapter_sha256='same-adapter'))
    stats=advantages(r['rewards'],all_rewards=torch.cat([torch.arange(8).float()[None,:]/8+k*.01 for k in range(world)]))
    r.update({k:stats[k] for k in ('advantages','adv_mask','global_std','raw_group_std')})
    r['advantage_statistics']={}
    return r


def worker(rank,root,init):
    dist.init_process_group('gloo',init_method='file://'+init,world_size=2,rank=rank)
    ck.init_control_group();root=Path(root)
    identity,state,model,opt,sched,gen=fixture(rank,2)
    base=ck.save(root,model,opt,sched,gen,state,None,identity)
    seal=(root/'latest.json').read_bytes();passed=[]
    def rejects(name,fn):
        caught=False
        try:fn()
        except (ValueError,RuntimeError,FileExistsError):caught=True
        if not all(ck.gather(caught)):raise AssertionError('Failure did not reach all ranks: '+name)
        if (root/'latest.json').read_bytes()!=seal:raise AssertionError('Failure advanced latest: '+name)
        passed.append(name)
    for case,bad_rank in [('rank0_before_sampling',0),('nonrank_scoring_after_partial_chain',1),('before_advantage_commit',0)]:
        s=dict(state,phase='COLLECTING');r=rollout(rank,2,identity) if rank!=bad_rank else None
        if case=='before_advantage_commit':r=rollout(rank,2,identity);r.pop('advantages')
        rejects(case,lambda:ck.save(root,model,opt,sched,gen,s,r,identity,tag=case))
    bad=dict(state,phase='ROLLOUT_COMMITTED',collections=1,scene_cursor=2,raw_scene_exposures=2,new_candidates=16)
    r=rollout(rank,2,identity);r.pop('advantages')
    rejects('unfinalized_even_if_phase_forged',lambda:ck.save(root,model,opt,sched,gen,bad,r,identity,tag='unfinalized'))
    for point in ('rank_write','before_complete'):
        def fault(p,worker):
            if p==point and worker==(1 if point=='rank_write' else 0):raise OSError('injected disk/publication fault')
        rejects(point,lambda:ck.save(root,model,opt,sched,gen,state,None,identity,tag=point,fault=fault))
        # Same logical checkpoint can be retried despite an orphaned incomplete attempt.
        path=ck.save(root,model,opt,sched,gen,state,None,identity,tag=point)
        seal=(root/'latest.json').read_bytes()
        rejects('duplicate_'+point,lambda:ck.save(root,model,opt,sched,gen,state,None,identity,tag=point))
        if rank==0 and not list((root/'checkpoints').glob(point+'.incomplete.*')):raise AssertionError('Failure evidence lost')
    for i in (1,2,3):
        opt.zero_grad();model(torch.ones(1,2)).sum().backward();opt.step();sched.step()
        state.update(updates=i,collections=1,next_inner=i,scene_cursor=2,raw_scene_exposures=2,new_candidates=16,phase='INNER_COMMITTED')
        r=rollout(rank,2,identity)
        path=ck.save(root,model,opt,sched,gen,state,r,identity)
        before=copy.deepcopy(opt.state_dict());rng=gen.get_state().clone()
        with torch.no_grad():
            for p in model.parameters():p.add_(100)
        restored,rr=ck.restore(path,model,opt,sched,gen,identity)
        assert restored==state and torch.equal(rr['chain'],r['chain']) and torch.equal(gen.get_state(),rng)
        assert all(torch.equal(v['exp_avg'],opt.state_dict()['state'][k]['exp_avg']) for k,v in before['state'].items())
        passed.append('inner'+str(i)+'_restore')
    # Same update is never silently deduplicated. Rollback archives the entire old file.
    logroot=root/'journal';logroot.mkdir(exist_ok=True)
    rows=[dict(updates=i,collection=1,inner_completed=i,behavior=dict(run=identity['sha256'],update=0)) for i in (1,2,3)]
    (logroot/f'metrics_rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    rollback=dict(state,updates=1,next_inner=1)
    inv=Invocation(logroot,rank,rollback,identity,path)
    assert len((logroot/f'metrics_rank{rank}.jsonl').read_text().splitlines())==1
    assert (inv.directory/f'aborted_metrics_rank{rank}.jsonl').exists()
    inv.finish(rollback,'TEST_COMPLETE');passed.append('journal_rollback_preserves_aborted_attempt')
    atomic_json(root/f'result_rank{rank}.json',dict(status='PASS',checks=passed))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    if torch.cuda.is_available():raise RuntimeError('CPU qualification must not acquire a GPU')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    mp.spawn(worker,args=(str(out),str(out/'gloo-init')),nprocs=2,join=True)
    rows=[json.loads((out/f'result_rank{r}.json').read_text()) for r in range(2)]
    result=dict(status='PASS',world=2,backend='CPU Gloo',CUDA_visible=False,checks=rows[0]['checks'],all_ranks=rows)
    atomic_json(out/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
