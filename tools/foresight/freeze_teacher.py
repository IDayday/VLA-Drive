"""Freeze after the registered COMPLETE30epoch teacher run, using only fixed dev milestones."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256


def choose_milestone(candidates):
    if not candidates:raise ValueError('No registered teacher milestones')
    for row in candidates:
        if row['failures'] or not math.isfinite(row['ego_with_peer_ADE']) or not math.isfinite(row['vehicle_with_peer_ADE']):
            raise ValueError('Incomplete/nonfinite teacher validation')
    return min(candidates,key=lambda r:(.5*(r['ego_with_peer_ADE']+r['vehicle_with_peer_ADE']),-r['epoch']))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--teacher-run',required=True);p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.teacher_run);out=Path(a.output)
    if out.exists():raise FileExistsError('Frozen identity cannot be overwritten')
    identity=json.loads((run/'identity.json').read_text());status=json.loads((run/'status.json').read_text())
    registered=json.loads(Path(a.registration).read_text())
    if identity['config']['limit']!=0 or status['status']!='COMPLETE':raise ValueError('A completed full-data teacher is required')
    if registered['epochs']!=30 or status['epoch']!=30 or status['completed']!=registered['maximum_optimizer_updates']:
        raise ValueError('Registered full teacher length not completed')
    if identity['source_sha']!=registered['source_sha'] or identity['data_identity']!=registered['data_identity'] or identity['config']['seed']!=registered['seed']:
        raise ValueError('Teacher registration identity mismatch')
    candidates=[]
    for epoch in (1,2,4,8,16,24,30):
        path=run/f'metrics_{epoch}.json';metric=json.loads(path.read_text())
        if metric['scope']!='FIXED_LOG_DISJOINT_DEV' or metric['epoch']!=epoch or metric['scenes']!=registered['dev_scenes']:
            raise ValueError('Teacher evaluation population/scope mismatch')
        candidates.append({'epoch':epoch,'completed':metric['completed'],'failures':metric['failures'],
            'ego_with_peer_ADE':metric['groups']['ego_with_peer']['full_peer_ADE'],
            'vehicle_with_peer_ADE':metric['groups']['vehicle_with_peer']['full_peer_ADE'],
            'metrics_sha256':file_sha256(path),'query_csv_sha256':file_sha256(run/f'queries_{epoch}.csv')})
    selected=choose_milestone(candidates);checkpoint=run/f"milestone_{selected['epoch']:03d}.pt"
    report={'schema':'foresight_frozen_teacher_v1','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'teacher_run_identity':identity['identity'],'teacher_source_sha':identity['source_sha'],
            'data_identity':identity['data_identity'],'registration_sha256':file_sha256(a.registration),
            'selection':'minimum0.5*(ego_with_peer_ADE+vehicle_with_peer_ADE), equal score chooses later fixed epoch',
            'full_training_completed':status['completed'],'selected':selected,'all_milestones':candidates,
            'checkpoint':checkpoint.name,'checkpoint_sha256':file_sha256(checkpoint),
            'ego_export_policy':'fresh encoder, all ego future masked before encoding; no reuse of full-GT hidden',
            'student_target':'raw8x512ego-time latent; nonaffine LayerNorm eps1e-5 in loss','trainable_after_freeze':False}
    report['identity']=identity_hash(report);atomic_json(out,report)


if __name__=='__main__':main()
