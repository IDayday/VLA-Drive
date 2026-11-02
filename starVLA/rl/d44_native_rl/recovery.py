"""Batch-aware boundary validation on existing atomic checkpoint transactions."""
import json
import os
import time
import uuid
from pathlib import Path
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
from starVLA.rl.qwen_world_abcd_research import checkpoint
from starVLA.rl.flow_grpo.transactions import atomic_json


def material_digest(batch):
    keys=('condition','chain','old_mean','ref_mean','std','advantages','behavior_version','teacher',
          'positive_state','positive_sigma','nominal_chain','nominal_sigma_last','fm_state','fm_sigma','fm_target','old_energy',
          'factors','beta','timesteps','deltas','padding','epoch','records','teacher_record')
    return tensor_digest([{k:r[k]for k in keys if k in r}for r in batch['scenes']])


class Boundary:
    @staticmethod
    def validate_boundary(state,batch,identity,optimizer=None,scheduler=None,**_):
        m=identity['recipe']['inner_updates'];u=state['updates'];i=state['next_inner'];c=state['collections']
        pending=batch is not None
        if state['phase']not in ('READY','ROLLOUT_COMMITTED','INNER_COMMITTED','COMPLETE'):
            raise ValueError('COLLECTING/failure evidence is not a resumable boundary')
        if not 0<=i<m or (pending and (u!=(c-1)*m+i or batch['behavior_update']!=u-i))or(not pending and(u!=c*m or i)):
            raise ValueError('Batch/optimizer/inner ancestry mismatch')
        if pending:
            if len(batch['scenes'])!=identity['recipe']['global_scenes']//identity['world']or batch['digest']!=material_digest(batch):
                raise ValueError('Saved multi-scene chain/old probabilities/advantages changed')
            if any(r['chain'].requires_grad or r['old_mean'].requires_grad or r['advantages'].requires_grad for r in batch['scenes']):
                raise ValueError('Behavior material must be detached')
        dual=state['adaptive_kl']
        if dual['collections']!=c-int(pending):raise ValueError('Dual updated more/less than once per fresh collection')
        if optimizer is not None:
            opt=optimizer if isinstance(optimizer,dict)else optimizer.state_dict()
            steps=[int(v['step'])for v in opt['state'].values()if 'step'in v]
            if (max(steps,default=0)!=u):raise ValueError('Adam optimizer steps differ from the committed update')
        if scheduler is not None:
            sch=scheduler if isinstance(scheduler,dict)else scheduler.state_dict()
            if sch['last_epoch']!=u:raise ValueError('Scheduler update count differs')
        return dict(run=identity['sha256'],phase=state['phase'],updates=u,collections=c,scene_cursor=state['scene_cursor'],
            next_inner=i,pending=pending,digest=batch['digest']if pending else None,
            valid_scene_exposures=state['valid_scene_exposures'],new_candidates=state['new_candidates'],dual=dual)

    @staticmethod
    def validate_ranks(rows):
        keys=('run','phase','updates','collections','scene_cursor','next_inner','pending','valid_scene_exposures','new_candidates','dual')
        if any({k:r[k]for k in keys}!={k:rows[0][k]for k in keys}for r in rows):raise ValueError('Ranks disagree on batch commit')
        return rows


class Attempt:
    def __init__(self,root,rank,state,identity,resume):
        self.root=Path(root);self.rank=rank;self.started=time.monotonic();self.id=checkpoint.gather(uuid.uuid4().hex if rank==0 else None)[0]
        self.path=self.root/'invocations'/self.id/f'rank{rank}.json';self.path.parent.mkdir(parents=True,exist_ok=True)
        self.record=dict(schema='d44_invocation_v1',id=self.id,rank=rank,run=identity['sha256'],status='RUNNING',
            started=time.time(),start_update=state['updates'],resume=str(resume)if resume else None,
            physical_candidates=0,seconds_save=0.,GPU_time_is_shared_worker_wall=True)
        journal=self.root/f'metrics_rank{rank}.jsonl'
        if journal.exists():
            rows=[json.loads(x)for x in journal.read_text().splitlines()];prefix=[r for r in rows if r['updates']<=state['updates']]
            if [r['updates']for r in prefix]!=list(range(1,state['updates']+1)):raise ValueError('Effective journal gap/duplicate')
            if any(r['run']!=identity['sha256']for r in rows):raise ValueError('Foreign journal identity')
            suffix=[r for r in rows if r['updates']>state['updates']]
            if suffix:
                archive=self.path.parent/f'aborted_metrics_rank{rank}.jsonl';archive.write_text(''.join(json.dumps(x)+'\n'for x in suffix))
                tmp=journal.with_suffix('.reconciled');tmp.write_text(''.join(json.dumps(x)+'\n'for x in prefix));os.replace(tmp,journal)
        elif state['updates']:raise ValueError('Missing effective learning journal')
        atomic_json(self.path,self.record)

    def add_cost(self,**values):
        # Physical work is append-attempt accounting, independent of committed
        # logical samples and never rolled back with a checkpoint.
        for key,value in values.items():self.record[key]=self.record.get(key,0)+value
        self.record['last_physical_work_unix']=time.time()
        atomic_json(self.path,self.record)

    def finish(self,state,status):
        elapsed=time.monotonic()-self.started;delta=state['updates']-self.record['start_update']
        self.record.update(status=status,end_update=state['updates'],worker_wall_seconds=elapsed,
            invocation_updates=delta,updates_per_second=delta/elapsed if elapsed else None)
        atomic_json(self.path,self.record)
