"""Bounded native checkpoint payload queue; latest/best/in-use points are pinned."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid


def rotate(root,*,keep=10,pinned=()):
    root=Path(root).resolve();directory=root/'checkpoints'
    if keep<1:raise ValueError('Retain at least one native recovery point')
    if not directory.is_dir():return {'kept':[],'removed':[]}
    with(root/'checkpoint_retention.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not(root/'latest.json').is_file():
            # The native writer creates checkpoints/ before publishing its
            # first atomic latest pointer. Keep every payload until then.
            return {'status':'WAITING_INITIAL_LATEST','kept':[],'removed':[]}
        latest=json.loads((root/'latest.json').read_text())['checkpoint']
        protected={Path(p).resolve()for p in pinned}|{Path(latest).resolve()}
        rows=[]
        for path in directory.iterdir():
            if path.is_symlink()or not path.is_dir()or not re.fullmatch(r'update\d+_inner\d+',path.name):continue
            try:seal=json.loads((path/'COMPLETE').read_text())
            except(OSError,ValueError):continue
            if seal.get('phase')not in('READY','INNER_COMMITTED','ROLLOUT_COMMITTED','COMPLETE'):continue
            if not(path/'optimizer.pt').is_file()or not all((path/f'rank{r}.pt').is_file()for r in range(seal['world'])):
                raise ValueError('Published native checkpoint has missing recovery payloads: '+str(path))
            rows.append((seal['updates'],seal['next_inner'],path))
        rows.sort(reverse=True);recent={p for _,_,p in rows[:keep]};removed=[]
        for updates,inner,path in rows[keep:]:
            if path.resolve()in protected:continue
            # One atomic namespace operation precedes deletion; incomplete
            # saves and protected live consumers are never candidates.
            staged=directory/('.retired-'+path.name+'-'+uuid.uuid4().hex)
            os.rename(path,staged);shutil.rmtree(staged)
            removed.append(dict(name=path.name,updates=updates,next_inner=inner,at=time.time()))
        if removed:
            with(root/'retired_checkpoints.jsonl').open('a')as f:
                for item in removed:f.write(json.dumps(item)+'\n')
        value=dict(at=time.time(),recent_limit=keep,recent=[str(p)for _,_,p in rows if p in recent],
            pinned=[str(p)for _,_,p in rows if p.resolve()in protected],removed=removed)
        tmp=root/'retained_checkpoints.json.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,root/'retained_checkpoints.json')
        return value
