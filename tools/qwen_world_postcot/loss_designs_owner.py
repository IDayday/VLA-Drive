"""Finite independent loss forks: TRAIN calibration, real qualification, full Test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from starVLA.rl.qwen_world_postcot.contracts import ROOT,fingerprint,file_sha,authorization,allowed_hosts
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.coordinate_objective import review_loading_adapter
from starVLA.rl.qwen_world_postcot.resources import probe,gpu_activity_issue
from tools.qwen_world_postcot import campaign as c

SOURCE=Path(__file__).resolve().parents[2]
ARMS=('TIME_RESIDUAL','DELTA','HUBER')
PARENT=ROOT.parent/'coordinate_loss_20261028/parent63000'


def read(path):return json.loads(Path(path).read_text())


def preferred_route(state,*,calibration=False):
    claimed=c.claimed_uuids(state)
    # The finite hosts were explicitly granted for this work. Use idle cards
    # there first, leaving the fixed hosts available for the fresh100k IL.
    for host in ['training-rl-zt2','training-vla-zt3']:
        if host not in allowed_hosts(admit_long_job=True):continue
        try:
            sample=probe(host);gpus=sample['gpus'][:1] if calibration else sample['gpus']
            if len(gpus)!=(1 if calibration else 8) or not sample['worker_python_exists']:continue
            if any(g['uuid']in claimed or g['free_MiB']<60000 or gpu_activity_issue(g,sample['owners'])for g in gpus):continue
            if any(o['uuid']in {g['uuid']for g in gpus}for o in sample['owners']):continue
            return dict(host=host,gpus=[g['index']for g in gpus],uuids=[g['uuid']for g in gpus],share=None,minimum_free_MiB=60000)
        except Exception:continue
    if calibration:
        routes=c.choose_eval_routes(49152,claimed,maximum=1)
        return routes[0] if routes else None
    return c.choose_training_host(49152,excluded_uuids=claimed)


def register():
    import torch
    pre=read(ROOT/'PREREGISTRATION.json');cal=read(ROOT/'calibration/CALIBRATION.json');body=dict(cal);seal=body.pop('sha256')
    if fingerprint(body)!=seal or cal['status']!='PASS' or cal['parent_model_sha256']!=pre['parent_model_sha256']:
        raise ValueError('Invalid one-time TRAIN calibration')
    client=torch.load(PARENT/'rank00_state.pt',map_location='cpu',weights_only=False)
    row=dict(schema='world_IL_loss_designs_v1',parent_checkpoint=str(PARENT.resolve()),parent_updates=63000,added_updates=2000,
        parent_model_sha256=pre['parent_model_sha256'],parent_identity_sha256=pre['parent_identity_sha256'],
        seed=42,evaluation_seed=2101,freeze_parent_candidate_bank=True,candidate_bank_hash=client['builder']['bank_hash'],
        arms=cal['arms'],calibration_sha256=seal,shared_control=pre['shared_control'],
        loading_adapter_review=review_loading_adapter('/mnt/project/qwen-world-postcot-20261025/snapshots/formal_main_v2/starVLA/rl/qwen_world_postcot/loading.py',
            SOURCE/'starVLA/rl/qwen_world_postcot/loading.py'))
    row['sha256']=fingerprint(row);atomic_json(ROOT/'REGISTRATION.json',row)


def command(arm,output,*,qualification=False,resume=None):
    result=[c.MODEL_PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        '-m','tools.qwen_world_postcot.train_il','--output',str(output),
        '--coordinate-registration',str(ROOT/'REGISTRATION.json'),'--coordinate-arm',arm]
    if qualification:result+=['--coordinate-qualification-updates','3']
    if resume:result+=['--resume',str(resume)]
    return result


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    if ROOT.name!='loss_designs_20261028':raise ValueError('Independent WORLD_POSTCOT_ROOT required')
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'OWNER.json').exists():raise FileExistsError('Do not duplicate this finite study owner')
    atomic_json(ROOT/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source=str(SOURCE),at=time.time()))
    state=dict(source=str(SOURCE),jobs={},stage='CALIBRATION',arms={name:'QUALIFICATION'for name in ('CONTROL',)+ARMS})
    try:
        while True:
            c.poll_jobs(state)
            failures=[j for j in state['jobs'].values()if j['status']=='FAILED' and not j.get('result',{}).get('migration_required')]
            if failures:raise RuntimeError('Owned invocation failed; preserve evidence for review: '+str([j['receipt']for j in failures]))
            if state['stage']=='CALIBRATION':
                completed=[j for j in state['jobs'].values()if j['name']=='calibration' and j['status']=='COMPLETE']
                if completed:
                    register();state['stage']='LEARNING'
                elif not c.active(state,name='calibration') and state.get('resource_retry_after',{}).get('calibration',0)<=time.time():
                    route=preferred_route(state,calibration=True)
                    if route:c.dispatch(state,str(SOURCE),'calibration',[c.MODEL_PYTHON,'-u','-m','tools.qwen_world_postcot.calibrate_loss_designs',
                        '--parent',str(PARENT),'--output',str(ROOT/'calibration')],route,role='calibration')
            if state['stage']=='LEARNING':
                for arm in ('CONTROL',)+ARMS:
                    stage=state['arms'][arm]
                    if stage=='COMPLETE':continue
                    name=arm+'_qualification'if stage=='QUALIFICATION'else arm+'_formal'
                    output=ROOT/('qualification'if stage=='QUALIFICATION'else'runs')/arm
                    finished=[j for j in state['jobs'].values()if j['name']==name and j['status']=='COMPLETE']
                    if finished:
                        boundary=read(output/'progress.json');endpoint=63003 if stage=='QUALIFICATION' else 65000
                        if boundary['updates']!=endpoint or boundary['status']!='STUDY_COMPLETE':
                            raise ValueError('Owned worker did not finish its registered endpoint')
                        for rank in range(8):
                            rows=[json.loads(l)for l in (output/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
                            if [r['updates']for r in rows]!=list(range(63001,endpoint+1))or any(r['optimizer_skipped_steps']for r in rows):
                                raise ValueError('Real optimizer progression/coverage failed')
                        state['arms'][arm]='COMPLETE'if stage=='FORMAL'or arm=='CONTROL'else'FORMAL'
                        continue
                    if c.active(state,name=name)or state.get('resource_retry_after',{}).get(name,0)>time.time():continue
                    # Shared original control has a real identical-value/gradient
                    # CPU check and this3-step eight-card execution qualification.
                    if arm!='CONTROL'and state['arms']['CONTROL']!='COMPLETE':continue
                    route=preferred_route(state)
                    if route:
                        previous=c.progress(output)
                        c.dispatch(state,str(SOURCE),name,command(arm,output,qualification=stage=='QUALIFICATION',
                            resume=previous['checkpoint']if previous else None),route,role='qualification'if stage=='QUALIFICATION'else'training')
                if all(v=='COMPLETE'for v in state['arms'].values()):state['stage']='WAITING_SHARED_CONTROL_TEST'
            if state['stage']=='WAITING_SHARED_CONTROL_TEST':
                control=ROOT.parent/'coordinate_loss_20261028/NAVTEST/CONTROL_065000/evaluation'
                if (ROOT.parent/'coordinate_loss_20261028/NAVTEST/COMPLETE.json').exists():
                    models={}
                    for arm in ARMS:
                        checkpoint=ROOT/'runs'/arm/'checkpoints/update065000';seal=read(checkpoint/'COMPLETE');identity=read(checkpoint/'identity.json');label=arm+'_065000'
                        models[label]=dict(label=label,kind='IL',updates=65000,checkpoint=str(checkpoint),parent_checkpoint=str(checkpoint),
                            files_sha256=seal['files_sha256'],native_identity_sha256=fingerprint(identity),source=str(SOURCE),
                            user_instruction='独立IL损失设计固定终点完整Navtest，复用匹配CONTROL')
                    evaluation=ROOT/'NAVTEST';evaluation.mkdir(exist_ok=False)
                    atomic_json(evaluation/'PINNED.json',dict(schema='world_checkpoint_sweep_NAVTEST_v1',models=models,
                        max_concurrent_models=3,reference_evaluation=str(control),seed=2101,full_population=12146,
                        loss_designs_study=dict(root=str(ROOT),shared_control=str(control))))
                    with(evaluation/'owner.log').open('xb')as f:
                        subprocess.run([c.MODEL_PYTHON,'-u','-m','tools.qwen_world_postcot.latest_snapshot_owner','--root',str(evaluation),
                            '--report-dir',str(SOURCE/'reports/qwen_world_IL_loss_designs_20261028')],cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,check=True)
                    if not(evaluation/'COMPLETE.json').exists():raise RuntimeError('Full Test did not complete')
                    atomic_json(ROOT/'COMPLETE.json',dict(status='COMPLETE',publication=read(evaluation/'PUBLICATION.json'),at=time.time()));return
            state['at']=time.time();atomic_json(ROOT/'STATE.json',state);time.sleep(60)
    except BaseException:
        atomic_json(ROOT/'FAILURE.json',dict(error=traceback.format_exc(),state=state,at=time.time(),automatic_retry=False));raise


if __name__=='__main__':main()
