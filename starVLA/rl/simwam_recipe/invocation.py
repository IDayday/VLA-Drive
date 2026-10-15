"""Append-only attempt provenance and rollback receipts, separate from RL data."""
import json
import os
from pathlib import Path
import time
import uuid
from .contracts import file_sha
from .checkpoint import gather
from starVLA.rl.flow_grpo.transactions import atomic_json


def validate_rows(rows, boundary):
    if [r['updates'] for r in rows] != list(range(1, boundary+1)):
        raise ValueError('Effective journal has a gap/duplicate; manual lineage audit required')
    for r in rows:
        u=r['updates']; c=(u-1)//4+1; i=(u-1)%4+1
        if r['collection']!=c or r['inner_completed']!=i or r['behavior']['update']!=(c-1)*4:
            raise ValueError('Journal behavior/collection/inner ancestry mismatch')


class Invocation:
    def __init__(self, root, rank, state, identity, resume=None):
        self.root=Path(root);self.rank=rank;self.started=time.monotonic()
        self.id=gather(uuid.uuid4().hex if rank==0 else None)[0]
        self.directory=self.root/'invocations'/self.id
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/f'rank{rank}.json'
        self.record=dict(schema='simwam_invocation_v2',invocation=self.id,rank=rank,
            run=identity['sha256'],start_unix=time.time(),status='RUNNING',
            start_update=state['updates'],start_collection=state['collections'],start_cursor=state['scene_cursor'],
            resume_checkpoint=str(resume) if resume else None,
            resume_files_sha256=file_sha(Path(resume)/'files.json') if resume else None,
            last_legal_checkpoint=str(resume) if resume else None,
            attempted_collections=0,generated_candidates=0,scored_candidates=0,
            seconds_generation=0.,seconds_scoring=0.,seconds_save=0.,
            GPU_occupancy='worker elapsed time; shared GPU, not exclusive GPU compute')
        journal=self.root/f'metrics_rank{rank}.jsonl'
        if journal.exists():
            raw=journal.read_bytes();rows=[];bad_tail=None
            lines=raw.splitlines(keepends=True)
            for idx,line in enumerate(lines):
                try:rows.append(json.loads(line))
                except (ValueError,UnicodeDecodeError):
                    if idx!=len(lines)-1:raise ValueError('Corrupt non-tail journal')
                    bad_tail=line.hex()
            prefix=[r for r in rows if r['updates']<=state['updates']]
            validate_rows(prefix,state['updates'])
            if any(r['behavior']['run']!=identity['sha256'] for r in rows):
                raise ValueError('Journal belongs to another scientific run')
            suffix=[r for r in rows if r['updates']>state['updates']]
            if suffix or bad_tail:
                archive=self.directory/f'aborted_metrics_rank{rank}.jsonl'
                with archive.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                atomic_json(self.directory/f'rollback_rank{rank}.json',dict(
                    schema=2,original_sha256=file_sha(archive),resume_checkpoint=str(resume),
                    retained_updates=state['updates'],aborted_updates=[r['updates'] for r in suffix],
                    incomplete_tail=bad_tail,reason='rollback to sealed checkpoint; original journal retained',
                    ancestry=[{k:r.get(k) for k in ('updates','collection','inner_completed','behavior','chain_identity','invocation')} for r in suffix]))
                tmp=journal.with_name(journal.name+'.reconciled-'+self.id)
                with tmp.open('x') as f:
                    for row in prefix:f.write(json.dumps(row,allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                os.replace(tmp,journal)
        elif state['updates']:
            raise ValueError('Cannot resume without the effective learning history')
        # Rollback cannot overwrite old behavior. Preserve every unpublished suffix.
        for p in sorted((self.root/'behavior').glob(f'collection*/rank{rank}.pt')):
            collection=int(p.parent.name.removeprefix('collection'))
            if collection>state['collections']:
                target=self.directory/'aborted_behavior'/p.parent.name/p.name
                target.parent.mkdir(parents=True,exist_ok=True)
                os.rename(p,target)
                atomic_json(target.with_suffix('.receipt.json'),dict(sha256=file_sha(target),
                    previous_path=str(p),resume_checkpoint=str(resume),status='UNCOMMITTED_OR_ROLLED_BACK'))
        atomic_json(self.path,self.record)

    def event(self, kind, seconds=0., candidates=0):
        if kind=='collection_start':self.record['attempted_collections']+=1
        elif kind=='generated':
            self.record['generated_candidates']+=candidates;self.record['seconds_generation']+=seconds
        elif kind=='scored':
            self.record['scored_candidates']+=candidates;self.record['seconds_scoring']+=seconds
        self.record['last_event']=kind;self.record['heartbeat_unix']=time.time()
        self.record['observed_worker_wall_seconds']=time.monotonic()-self.started
        atomic_json(self.path,self.record)

    def saved(self, checkpoint, seconds):
        self.record['last_legal_checkpoint']=checkpoint;self.record['seconds_save']+=seconds
        self.event('checkpoint_published')

    def finish(self,state,status,error=None):
        elapsed=time.monotonic()-self.started
        delta=state['updates']-self.record['start_update']
        self.record.update(status=status,end_unix=time.time(),end_update=state['updates'],
            end_collection=state['collections'],end_cursor=state['scene_cursor'],
            worker_wall_seconds=elapsed,invocation_updates=delta,
            invocation_updates_per_second=delta/elapsed if delta>0 and elapsed>0 else None,
            error=error,phase=state['phase'])
        atomic_json(self.path,self.record)
