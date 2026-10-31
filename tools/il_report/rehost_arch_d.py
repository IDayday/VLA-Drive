"""User-requested same-run D migration; reuse immutable native save/restore.

Only the metadata campaign owner is briefly adopted. Other native jobs keep
running. Destination secondary slots are held during the startup measurement.
"""
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path('/mnt/project/qwen-world-four-frameworks-20261029')
RECORD=ROOT/'resources/D_to_idle_vla_20261031'
DEST='training-vla-zt'
PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def read(path):return json.loads(Path(path).read_text())


def write(name,value):
    path=RECORD/name;tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w')as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def alive(row):
    try:
        fields=Path(f'/proc/{row["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z'and int(fields[19])==row['start_ticks']
    except FileNotFoundError:return False


def tails(run,n=64):
    rows=[]
    for rank in range(8):
        p=run/f'metrics_rank{rank}.jsonl'
        with p.open('rb')as f:
            f.seek(max(0,p.stat().st_size-512*1024));rows.append([json.loads(x)for x in f.read().splitlines()[-n:]])
    return rows


def main():
    RECORD.mkdir(parents=True,exist_ok=True)
    with ExitStack()as leases:
        guard=leases.enter_context((RECORD/'migration.lock').open('a'))
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(RECORD/'NEW_INVOCATION.json').exists():raise RuntimeError('Already dispatched: inspect existing receipt')
        manifest=read(ROOT/'RUNS.json');plan=manifest['plan'];source=Path(plan['source']);operator=Path(plan['controller_source'])
        os.environ.update(WORLD_COMPARE_ROOT=str(ROOT),WORLD_COMPARE_TASK='qwen_world_compare_20261029',PYTHONUTF8='1',LC_ALL='C.UTF-8')
        sys.path.insert(0,str(operator))
        from tools.qwen_world_compare import campaign as c
        from starVLA.rl.qwen_world_compare.resources import probe,remote,gpu_activity_issue
        from starVLA.rl.qwen_world_compare.contracts import allowed_hosts,file_sha,fingerprint
        if DEST not in allowed_hosts(admit_long_job=True):raise PermissionError('Destination no longer authorized')
        if c.expected_source_files(source)!=manifest['source_files']:raise ValueError('Scientific snapshot changed')
        run=Path(plan['architectures']['D']['IL']);identity=read(run/'IDENTITY.json')
        if identity['architecture_id']!='D'or identity['config']['il']['max_updates']!=100000:raise ValueError('Wrong D run')
        for name,digest in identity['source_files'].items():
            if file_sha(source/name)!=digest:raise ValueError('Native source identity changed: '+name)
        before=probe(DEST);uuids=[g['uuid']for g in before['gpus']]
        primary=[]
        for folder in ('runtime_gpu_leases','runtime_gpu_shared_workers','runtime_gpu_research_slots'):
            for uuid in uuids:
                stream=leases.enter_context((Path('/mnt/project')/folder/(uuid+'.lock')).open('a'))
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if folder=='runtime_gpu_leases':primary.append(stream)
        after=probe(DEST)
        if (len(uuids)!=8 or uuids!=[g['uuid']for g in after['gpus']]or
                any(s['owners']or not s['worker_python_exists']or min(g['free_MiB']for g in s['gpus'])<65000
                    or any(gpu_activity_issue(g,s['owners'])for g in s['gpus'])or
                    s['load1']>=s['CPU_affinity']*.85 for s in(before,after))):
            raise RuntimeError('Destination is not an idle eight-GPU host; learner untouched')
        environment=json.loads(remote(DEST,[PYTHON,'-c','import json,torch,transformers,deepspeed;print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,deepspeed=deepspeed.__version__,gpus=torch.cuda.device_count())))']).splitlines()[-1])
        if environment['gpus']!=8:raise ValueError('Destination environment has wrong GPU topology')
        write('PREDRAIN.json',dict(before=before,after=after,environment=environment,scientific_source=str(source),source_files_sha256=manifest['source_files_sha256']))
        owner=read(ROOT/'campaign/OWNER.json');proc=Path('/proc')/str(owner['pid'])
        argv=[x for x in(proc/'cmdline').read_bytes().decode().split('\0')if x]
        caught=int(next(x.split()[1]for x in(proc/'status').read_text().splitlines()if x.startswith('SigCgt:')),16)
        if not alive(owner)or'tools.qwen_world_compare.campaign'not in argv or not caught&(1<<(signal.SIGTERM-1)):
            raise RuntimeError('Cannot safely adopt existing metadata campaign owner')
        write('PREVIOUS_OWNER.json',dict(owner=owner,argv=argv,cwd=str(proc.joinpath('cwd').resolve())))
        write('BEFORE_WINDOW.json',dict(ranks=tails(run),note='Shared-host observations; not exclusive hardware benchmark'))
        os.kill(owner['pid'],signal.SIGTERM) # CPU scheduler only; its handler preserves every native child.
        deadline=time.monotonic()+120
        while alive(owner)and time.monotonic()<deadline:time.sleep(1)
        if alive(owner):raise RuntimeError('Metadata owner has not released ownership; no native stop requested')
        try:
            with(ROOT/'campaign/owner.lock').open('a')as owner_lock:
                fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                instance=c.Campaign(plan,manifest['source_files']);instance._poll();instance.persist()
                active=[j for j in instance.state['jobs'].values()if j.get('architecture')=='D'and j.get('kind')=='training'and j['status']=='RUNNING']
                if len(active)!=1:raise RuntimeError('Expected exactly one active D invocation')
                old=active[0];receipt=Path(old['receipt']);write('OLD_INVOCATION.json',old)
                (receipt/'STOP_REQUESTED').touch()
                write('STOP_REQUEST.json',dict(at=time.time(),native_signals=0,method='cooperative native optimizer boundary file',job=str(receipt)))
                deadline=time.monotonic()+900
                while not(receipt/'RESULT.json').exists():
                    if time.monotonic()>deadline:raise TimeoutError('No clean native result; no force kill')
                    time.sleep(5)
                result=read(receipt/'RESULT.json')
                if result.get('status')!='EXITED'or result.get('returncode')!=0:raise ValueError('D did not exit cleanly')
                code='''import json,sys;from pathlib import Path
x=json.load(open(sys.argv[1]));rows=[dict(pid=x['worker_pid'],start_ticks=x['worker_start_ticks']),dict(pid=x['child_pid'],start_ticks=x['child_start_ticks'])]+x.get('known_rank_processes',[])
live=[]
for r in rows:
 try:
  s=Path('/proc',str(r['pid']),'stat').read_text().rsplit(')',1)[1].split()
  if s[0]!='Z'and int(s[19])==r['start_ticks']:live.append(r)
 except FileNotFoundError:pass
print(json.dumps(live))'''
                for _ in range(12):
                    live=json.loads(remote(old['host'],['/usr/bin/python3','-c',code,str(receipt/'RESULT.json')]))
                    if not live:break
                    time.sleep(1)
                if live:raise RuntimeError('Old owned transport remains alive')
                progress=read(run/'progress.json');checkpoint=Path(progress['checkpoint'])
                seal=read(checkpoint/'COMPLETE');files=read(checkpoint/'files.json')
                if (progress['status']!='STOPPED'or read(checkpoint/'identity.json')!=identity or
                    file_sha(checkpoint/'files.json')!=seal['files_sha256']or seal['updates']!=progress['updates']):raise ValueError('Invalid stopped native boundary')
                import torch
                for rank in range(8):
                    client=torch.load(checkpoint/f'rank{rank:02d}_state.pt',map_location='cpu',weights_only=False)
                    if (client['identity_sha256']!=fingerprint(identity)or any(client['state'][k]!=progress[k]for k in('updates','cursor','phase'))or
                        set(client['rng'])!={'python','numpy','cpu','cuda'}or client['builder']['schema']!='four_framework_stateless_batches_v1'or
                        client['builder']['bank_hash']!=identity['candidate_bank_hash']or client['builder']['candidate_refresh']is not False):
                        raise ValueError('Incomplete per-rank state/RNG/builder identity')
                write('NATIVE_STOP_COMMITTED.json',dict(checkpoint=str(checkpoint),updates=progress['updates'],cursor=progress['cursor'],seal=seal,old_transport_and_ranks_exited=True,rank_states_validated=8,full_shard_hash_check='strict_native_restore_on_startup',optimizer_shards=sum(n.endswith('_optim_states.pt')for n in files),original_result=result))
                instance._poll();instance.persist()
                current=probe(DEST)
                if current['owners']or [g['uuid']for g in current['gpus']]!=uuids:raise RuntimeError('Destination changed before admission')
                command=list(old['command'])
                if '--resume'in command:command[command.index('--resume')+1]=str(checkpoint)
                else:command+=['--resume',str(checkpoint)]
                for stream in primary:fcntl.flock(stream,fcntl.LOCK_UN)
                entry=instance._launch('D',f'D_IL_resume{progress["updates"]:06d}',command,DEST,gpus=range(8),uuids=uuids,minimum=65000,family='D_IL',kind='training',stage='IL')
                instance.state['architectures']['D']['status']='IL_RUNNING';instance.persist()
                write('NEW_INVOCATION.json',dict(entry=entry,resume=str(checkpoint),at=time.time(),budget=100000,source_changed=False,zero2_optimizer_scheduler_rng_restored_by_native_entry=True))
        finally:
            # Resume the exact single scheduler; no other learner is recreated.
            with(RECORD/'restored_campaign.log').open('ab')as log:
                process=subprocess.Popen(argv,cwd=operator,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write('RESTORED_CAMPAIGN.json',dict(pid=process.pid,start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),command=argv,cwd=str(operator),at=time.time()))
        # Keep other admission slots reserved for an uncontended qualification
        # window; all reserved descriptors close on success or failure.
        boundary=progress['updates'];deadline=time.monotonic()+1500
        while time.monotonic()<deadline:
            native=tails(run);fresh=[[r for r in rows if r['updates']>boundary]for rows in native]
            result_path=Path(entry['receipt'])/'RESULT.json'
            if result_path.exists():raise RuntimeError('New D invocation ended: '+str(result_path))
            if min(len(r)for r in fresh)>=32:
                for rows in fresh:
                    if any(r['optimizer_global_steps']!=r['updates']or r['optimizer_skipped_steps']!=0 or not math.isfinite(r['grad_norm'])for r in rows):raise ValueError('Invalid resumed effective optimizer update')
                baseline=read(RECORD/'BEFORE_WINDOW.json')['ranks']
                old_seconds=max(statistics.median(r['seconds']for r in rows)for rows in baseline)
                new_seconds=max(statistics.median(r['seconds']for r in rows[-32:])for rows in fresh)
                write('COMPLETE.json',dict(status='PASS',at=time.time(),host=DEST,gpus=list(range(8)),native_checkpoint_boundary=boundary,actual_updates=min(r[-1]['updates']for r in fresh),fresh_effective_updates=min(len(r)for r in fresh),old_median_s=old_seconds,new_median_s=new_seconds,observed_speedup=old_seconds/new_seconds,comparison='Different wall windows, not isolated paired benchmark',other_native_jobs_stopped=0,source_budget_batch_and_seed_changed=False))
                print(json.dumps(read(RECORD/'COMPLETE.json')),flush=True);return
            time.sleep(15)
        raise TimeoutError('Training startup measurement pending; inspect existing invocation, never duplicate')


if __name__=='__main__':
    RECORD.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException:
        write('FAILURE.json',dict(at=time.time(),error=traceback.format_exc(),no_force_kill=True,no_blind_retry=True));raise
