"""User-authorized fresh100k World IL with V12 planning targets only.

Reuses the accepted trainer, atomic native recovery, asynchronous candidate
refresh controller, export/parity and full Test workers. Never starts RL.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from starVLA.rl.qwen_world_postcot.contracts import ROOT,file_sha,fingerprint
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from tools.qwen_world_postcot import campaign as c

SOURCE=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def recover_owner(path):
    """Reviewed authorization-only recovery; native scientific identity stays exact."""
    receipt=read(path);body=dict(receipt)
    if body.pop('sha256',None)!=fingerprint(body)or receipt['schema']!='world_optimized_IL_authorization_recovery_v1':
        raise ValueError('Unsealed authorization recovery')
    previous=read(ROOT/'OWNER.json')
    if previous!=receipt['previous_owner']:raise ValueError('Recovery owner ancestry changed')
    proc=Path('/proc')/str(previous['pid'])/'stat'
    if proc.exists()and int(proc.read_text().rsplit(')',1)[1].split()[19])==previous['start_ticks']:
        raise RuntimeError('Original owner still alive; no duplicate recovery')
    for rel,digest in receipt['scientific_source_files'].items():
        if file_sha(SOURCE/rel)!=digest:raise ValueError('Scientific source changed: '+rel)
    if file_sha(Path(receipt['checkpoint'])/'files.json')!=receipt['checkpoint_files_sha256']:
        raise ValueError('Recovery inventory changed')
    failure=Path(receipt['previous_failure'])
    if file_sha(failure)!=receipt['previous_failure_sha256']:raise ValueError('Failure ancestry changed')
    state=read(failure)['state'];c.poll_jobs(state)
    for name,job in state['jobs'].items():
        if job['status']in('RUNNING','WAITING_OWNED_EXIT'):raise RuntimeError('Previous worker has not exited: '+name)
        if job['status']=='FAILED':
            if name!='formal_IL_attempt001':raise ValueError('Unexpected failure requires separate review')
            error=(Path(job['receipt'])/'stdout.log').read_text(encoding='utf-8',errors='replace')
            if 'Task host lease and immutable grant disagree: temporary_hosts'not in error:
                raise ValueError('Failure differs from the reviewed authorization incident')
            job.update(status='ABORTED_AUTHORIZATION',recovery_receipt=str(Path(path).resolve()))
    state.update(stage='FULL_IL',recovery_receipt=str(Path(path).resolve()))
    return state


def command(output,config,targets,*,qualification=False,resume=None):
    args=[c.MODEL_PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',
          '-m','tools.qwen_world_postcot.train_il','--config',str(config),'--output',str(output),
          '--optimized-targets',str(targets)]
    if qualification:args+=['--qualification','--qualification-joint','--stop-after','3']
    if resume:args+=['--resume',str(resume)]
    return args


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--recover-owner');a=p.parse_args()
    if ROOT.name!='optimized_labels_IL_20261028':raise ValueError('Explicit independent WORLD_POSTCOT_ROOT required')
    ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'OWNER.json').exists()and not a.recover_owner:raise FileExistsError('Do not duplicate the optimized IL owner')
    targets=ROOT/'targets';target_id=read(targets/'MANIFEST.json');config=Path(a.config).resolve()
    state=(recover_owner(a.recover_owner)if a.recover_owner else dict(source=str(SOURCE),jobs={},resource_refusals={},stage='QUALIFICATION',
        budget_updates=100000,target_identity_sha256=target_id['sha256'],automatic_RL=False))
    atomic_json(ROOT/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source=str(SOURCE),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),at=time.time()))
    receipt=dict(minimum_free_MiB=49152,micro_batch=4,evaluation_minimum_free_MiB=32768,refresh_minimum_free_MiB=32768)
    test_process=None;tests_done=list(state.get('tests_done',[]))
    if a.recover_owner:
        old=ROOT/'FAILURE.json'
        if old.exists():os.replace(old,Path(a.recover_owner).parent/'ROOT_FAILURE_before_recovery.json')
        atomic_json(ROOT/'RECOVERY_ACTIVE.json',dict(receipt=str(Path(a.recover_owner).resolve()),owner_pid=os.getpid(),at=time.time()))
    try:
        while True:
            c.poll_jobs(state)
            qualification=ROOT/'qualification/optimized_labels'
            if state['stage']=='QUALIFICATION':
                done=[j for j in state['jobs'].values() if j['name']=='qualification' and j['status']=='COMPLETE']
                if done:
                    progress=read(qualification/'progress.json')
                    if progress['updates']!=3:raise ValueError('Real qualification did not complete three optimizer updates')
                    for rank in range(8):
                        rows=[json.loads(l) for l in (qualification/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
                        if [r['updates'] for r in rows]!=[1,2,3] or any(r['optimizer_skipped_steps'] or r['planning_supervision_identity']!=target_id['sha256'] for r in rows):
                            raise ValueError('Qualification update/target contract failed')
                        dtype=read(qualification/f'DTYPE_update000001_rank{rank}.json')
                        if any(v is None for v in dtype['incoming_gradients'].values()):raise ValueError('Missing real joint-model gradient')
                    atomic_json(ROOT/'QUALIFICATION.json',dict(status='PASS',world=8,actual_joint_updates=3,
                        planning_target_sha256=target_id['sha256'],qualification_weights_used_for_formal=False,
                        all_five_losses_present=True,original_native_recovery_reused=True))
                    state['stage']='FULL_IL'
                elif not c.active(state,name='qualification'):
                    route=c.choose_training_host(receipt['minimum_free_MiB'],excluded_uuids=c.claimed_uuids(state))
                    if route:c.dispatch(state,str(SOURCE),'qualification',command(qualification,config,targets,qualification=True),route,role='qualification')
            if state['stage']=='FULL_IL':
                progress=c.progress(c.IL_RUN)
                if not c.active(state,name='formal_IL') and (not progress or progress['updates']<100000):
                    previous=[j for j in state['jobs'].values() if j['name']=='formal_IL' and j['status']=='FAILED']
                    if previous and not previous[-1].get('result',{}).get('migration_required'):
                        raise RuntimeError('Training stopped unexpectedly; exact checkpoint review required, no reset/retry')
                    route=c.choose_training_host(receipt['minimum_free_MiB'],excluded_uuids=c.claimed_uuids(state))
                    if route:c.dispatch(state,str(SOURCE),'formal_IL',command(c.IL_RUN,config,targets,resume=progress['checkpoint'] if progress else None),route,role='training')
                # Reuse the original refresh semantics with this run's own
                # immutable checkpoint identity and its independent registry.
                c.advance_refreshes(state,str(SOURCE),receipt)
                if test_process is not None and test_process.poll() is not None:
                    directory=ROOT/'NAVTEST'/f'update{state["active_Test"]:06d}'
                    if test_process.returncode or not (directory/'COMPLETE.json').exists():raise RuntimeError('Full Test pipeline needs review: '+str(directory))
                    tests_done.append(state.pop('active_Test'));test_process=None
                for step in [20000,40000,60000,80000,100000]:
                    checkpoint=c.IL_RUN/'checkpoints'/f'update{step:06d}'
                    if step in tests_done or test_process is not None or not (checkpoint/'COMPLETE').exists():continue
                    identity=read(checkpoint/'identity.json');seal=read(checkpoint/'COMPLETE')
                    directory=ROOT/'NAVTEST'/f'update{step:06d}';directory.mkdir(parents=True,exist_ok=False);label=f'OPTIMIZED_IL_{step:06d}'
                    pin=dict(label=label,kind='IL',updates=step,checkpoint=str(checkpoint),parent_checkpoint=str(checkpoint),
                        files_sha256=seal['files_sha256'],native_identity_sha256=fingerprint(identity),source=str(SOURCE),
                        user_instruction='完整100k优化轨迹Qwen-World IL，固定单候选canonical Navtest观察')
                    atomic_json(directory/'PINNED.json',dict(schema='world_checkpoint_sweep_NAVTEST_v1',models={label:pin},
                        max_concurrent_models=1,reference_evaluation='/mnt/project/qwen-world-postcot-20261025/IL_midlate_loss_NAVTEST_20261028T102909Z/WORLD_IL_063000/evaluation',
                        seed=2101,full_population=12146))
                    log=(directory/'owner.log').open('xb')
                    test_process=subprocess.Popen([c.MODEL_PYTHON,'-u','-m','tools.qwen_world_postcot.latest_snapshot_owner','--root',str(directory),
                        '--report-dir',str(SOURCE/'reports/qwen_world_optimized_labels_IL_20261028'/f'update{step:06d}')],
                        cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();state['active_Test']=step
                if progress and progress['updates']==100000 and len(tests_done)==5:
                    atomic_json(ROOT/'COMPLETE.json',dict(status='FULL_IL100000_AND_FULL_NAVTEST_COMPLETE',tests=tests_done,automatic_RL=False));return
            fatal=[j for j in state['jobs'].values() if j['status']=='FAILED' and j['role']=='qualification']
            if fatal:raise RuntimeError('Qualification failed; preserve evidence and review before formal IL')
            state['tests_done']=tests_done;state['at']=time.time();atomic_json(ROOT/'STATE.json',state)
            time.sleep(60)
    except BaseException:
        atomic_json(ROOT/'FAILURE.json',dict(error=traceback.format_exc(),at=time.time(),state=state,automatic_retry=False));raise


if __name__=='__main__':main()
