"""Finite event-driven official-model diagnostic queue; no foreign process control."""
import fcntl
import json
import os
from pathlib import Path
import time

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts
from tools.d44_native_rl.launch import launch
from tools.d46_augustus_ops.event_wait import wait_for_events

ROOT=Path('/mnt/project/d46-aug-migration-20261104')
SOURCE=Path(__file__).resolve().parents[2]


def read(p,default=None):
    return json.loads(Path(p).read_text())if Path(p).is_file()else default


def main():
    directory=ROOT/'upstream/probes';directory.mkdir(exist_ok=True)
    with(directory/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(directory/'STATE.json',dict(stage='WAITING_VERIFIED_WEIGHTS',created=time.time(),cases={}))
        atomic_json(directory/'OWNER.json',dict(pid=os.getpid(),source=str(SOURCE),at=time.time()))
        specifications=[('SFT_inference','SFT',0),('stage1_inference','RL/stage1_94.0',0),('SFT_one_action_update','SFT',1)]
        while not(directory/'STOP_OWNER').exists():
            # Authorization is re-read even while waiting; nothing launches on
            # any other host, or from an unverified partial download.
            if 'training-vlawm-tsinghua-zt5'not in allowed_hosts(admit_long_job=True):
                state['stage']='RESOURCE_WITHDRAWN';atomic_json(directory/'STATE.json',state);return
            if time.time()-state['created']>8*3600:
                state['stage']='DEFERRED_DOWNLOAD_OR_RESOURCE';atomic_json(directory/'STATE.json',state);return
            download=read(ROOT/'upstream/DOWNLOAD_RESULT.json',{})
            if download.get('status')=='COMPLETE':
                for name,checkpoint,updates in specifications:
                    case=state['cases'].get(name)
                    if case:
                        result=read(Path(case['job'])/'RESULT.json')
                        if result is None:break
                        if result.get('returncode')!=0:
                            state['stage']='REVIEW_REQUIRED';state['failure']=result
                            atomic_json(directory/'STATE.json',state);return
                        summary=read(directory/name/'SUMMARY.json')
                        if not summary or summary.get('status')!='COMPLETE_OFFICIAL_TRAIN_DIAGNOSTIC':
                            raise ValueError('EXIT0 without complete actual diagnostic')
                        case['status']='COMPLETE';continue
                    job=ROOT/'jobs/official_probes'/name
                    command=[str(ROOT/'upstream/venv/bin/python'),'-u','-m','tools.d46_aug_migration.official_probe',
                        '--upstream',str(ROOT/'upstream/AUGUstus'),'--checkpoint',str(ROOT/'upstream/checkpoints'/checkpoint),
                        '--data',str(ROOT/'upstream/train_probe'),'--output',str(directory/name),'--adam-updates',str(updates)]
                    state['cases'][name]=launch('training-vlawm-tsinghua-zt5',SOURCE,job,command,[7],
                        share_min_free_mib=65000 if updates else 49152,worker_source=SOURCE,
                        task_root='/mnt/project/d46-priority-rl-20261103',task='d46_aug_migration_20261104',admission_wait_seconds=1800)
                    state['stage']='RUNNING_OR_RESOURCE_ADMISSION';break
                else:
                    state['stage']='COMPLETE';atomic_json(directory/'STATE.json',state);return
            state['at']=time.time();atomic_json(directory/'STATE.json',state)
            paths=[ROOT/'upstream',directory]+[Path(v['job'])for v in state['cases'].values()]
            wait_for_events(paths,60,ignore=('STATE.json','OWNER.json','owner.log'))


if __name__=='__main__':main()
