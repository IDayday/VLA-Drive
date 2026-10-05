"""Publish compact training curves/identities for validated C/S 100k endpoints.

Reads existing records only. No checkpoints, labels, images or raw scene IDs
are copied. GPU-hours are finalized student allocation meters, not an estimate
of exclusive kernel use or the full teacher/cache/evaluation campaign cost.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from tools.full_foresight.navtest_schedule import atomic, read, sha, signature, write_csv
from tools.local_interaction_mask_v2.compare_pdms import read as read_scenes


def collect_steps(path, arm, expected_updates, expected_exposure, bin_size=1000):
    digest=hashlib.sha256();curves=[];bucket={};first=None;last=None
    total_seconds=0.;maximum_memory=0;nonfinite=0;unverified=0;observations={}
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line);r=json.loads(line)
            if r['update'] != (1 if last is None else last['update']+1):
                raise ValueError('Repeated/missing optimizer update in '+str(path))
            if first is None:first=r
            values=dict(r['raw_losses'],step_seconds=r['seconds'],grad_norm=r['grad_norm'])
            if any(not math.isfinite(x) for x in values.values()):
                nonfinite+=1;raise ValueError('Nonfinite completed training record')
            total_seconds+=r['seconds'];maximum_memory=max(maximum_memory,r['peak_memory_bytes'])
            unverified+=not r['fp32_master_update_verified']
            for k,v in values.items():bucket[k]=bucket.get(k,0.)+v
            if r.get('shared_parameter_observation'):
                observations[str(r['update'])]=r['shared_parameter_observation']
            if r['update']%bin_size==0 or r['update']==expected_updates:
                count=bin_size if r['update']%bin_size==0 else r['update']%bin_size
                curves.append(dict(arm=arm,update=r['update'],exposure=r['exposure'],
                                   averaged_updates=count,lr_end=r['lr'],
                                   **{k+'_mean':v/count for k,v in bucket.items()}))
                bucket={}
            last=r
    if last is None or last['update']!=expected_updates or last['exposure']!=expected_exposure:
        raise ValueError('Training log does not cover the completed status')
    return curves,dict(steps_sha256=digest.hexdigest(),first_step_end_unix=first['ended_unix'],
        last_step_end_unix=last['ended_unix'],optimizer_step_seconds_sum=total_seconds,
        peak_rank0_memory_bytes=maximum_memory,nonfinite_records=nonfinite,
        unverified_master_update_records=unverified,shared_gradient_update_observations=observations)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('c-root','s-root','results','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();roots={'C':Path(a.c_root),'S':Path(a.s_root)}
    report=Path(a.results);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    endpoints=list(csv.DictReader((report/'ENDPOINT_100K.csv').open()))
    if {r['arm'] for r in endpoints}!={'C0','C1','C4','S0','S1','S2','S3','S4'}:
        raise ValueError('Eight validated completed endpoints required')
    meters={key:[read(f) for f in (root/'runs').glob('*/status.json')] for key,root in roots.items()}
    students={key:{read(f)['identity']:f.parent for f in (root/'students').glob('*/identity.json')}
              for key,root in roots.items()}
    runs={};curves=[];costs=[]
    for endpoint in endpoints:
        arm=endpoint['arm'];summary=read(endpoint['score_summary']);cp=summary['export_identity']['checkpoint']
        folder=students[arm[0]][cp['run_identity']];identity=read(folder/'identity.json');status=read(folder/'status.json')
        if (status['status']!='COMPLETE' or status['completed']!=100000
                or identity['source_sha']!=cp['training_source_sha'] or status['identity']!=identity['identity']):
            raise ValueError('Completed run/endpoint identity mismatch')
        rows,evidence=collect_steps(folder/'steps.jsonl',arm,status['completed'],status['exposure'])
        curves+=rows
        attempts=[m for m in meters[arm[0]] if m.get('run_id_parent')==folder.name
                  and m.get('kind')=='foresight_student_formal']
        if not attempts or any('gpu_hours' not in m or 'end_unix' not in m for m in attempts):
            raise ValueError('Missing finalized student cost meter')
        if sum(m['real_optimizer_updates'] for m in attempts)!=status['completed']:
            raise ValueError('Attempt update accounting differs from run progress')
        checkpoints=[json.loads(line) for line in (folder/'checkpoint_costs.jsonl').read_text().splitlines()]
        init=read(folder/'initialization.json')
        controller_path=roots[arm[0]]/'formal_controllers'/folder.name/'status.json'
        controller=read(controller_path) if controller_path.exists() else {}
        costs.append(dict(arm=arm,run_id=folder.name,updates=status['completed'],exposure=status['exposure'],
            student_allocation_GPUh=sum(m['gpu_hours'] for m in attempts),
            optimizer_step_seconds_sum=evidence['optimizer_step_seconds_sum'],
            checkpoint_seconds_sum=sum(r['seconds'] for r in checkpoints),attempts=len(attempts)))
        runs[arm]=dict(run_id=folder.name,status=status,identity=identity,
            checkpoint=cp,parameters=read(folder/'parameters.json'),
            initialization_component_sha256={k:signature(v) for k,v in init.items()},
            initialization_file_sha256=sha(folder/'initialization.json'),
            auxiliary_initialization_file_sha256=sha(folder/'auxiliary_initialization.json'),
            meters=attempts,training_evidence=evidence,
            original_controller_status=controller.get('status','NOT_RECORDED'),
            original_controller_error=controller.get('error'),
            cost_scope='Finalized student allocation meters include loading/checkpoints/attempt failures; '
                       'evaluation exports may overlap training and are not added; shared teacher/cache costs excluded.')
    indexes={r['identity']['selected_index_hash'] for r in runs.values()}
    if len(indexes)!=1:raise ValueError('Training population differs across runs')
    plan=read(roots['S']/'formal_plan_seed42_v2.json')
    train=read(Path(plan['train_data'])/'index.json');dev=read(Path(plan['dev_data'])/'index.json')
    def population(rows):
        tokens={r['token'] for r in rows};logs={r['log'] for r in rows}
        if len(tokens)!=len(rows):raise ValueError('Duplicate scenes')
        return tokens,logs
    train_tokens,train_logs=population(train);dev_tokens,dev_logs=population(dev)
    if train_tokens&dev_tokens or train_logs&dev_logs:raise ValueError('Train/dev overlap')
    nav=read_scenes(Path(endpoints[0]['score_summary']).parent/'scenes.csv')
    nav_tokens=set(nav);nav_logs={r['log'] for r in nav.values()}
    if (len(nav_tokens)!=12146 or len(nav_logs)!=136 or train_tokens&nav_tokens
            or dev_tokens&nav_tokens or train_logs&nav_logs or dev_logs&nav_logs):
        raise ValueError('Navtest population/partition isolation mismatch')
    atomic(out/'DATA_POPULATION.json',dict(train_scenes=len(train),train_logs=len(train_logs),
        dev_scenes=len(dev),dev_logs=len(dev_logs),train_dev_token_overlap=0,train_dev_log_overlap=0,
        train_navtest_token_overlap=0,train_navtest_log_overlap=0,
        dev_navtest_token_overlap=0,dev_navtest_log_overlap=0,
        train_index_file_sha256=sha(Path(plan['train_data'])/'index.json'),
        dev_index_file_sha256=sha(Path(plan['dev_data'])/'index.json'),common_selected_index_hash=next(iter(indexes)),
        epoch_equivalent_each=runs['S0']['status']['exposure']/len(train),
        official_navtest_scenes=12146,official_navtest_logs=136,
        public_data='Only aggregate populations/hashes and anonymous metric rows; no original indexes, labels or sensor data.'))
    atomic(out/'COMPLETED_RUNS.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        collection_source_sha256=sha(__file__),new_optimizer_updates=0,new_model_inference=0))
    write_csv(out/'TRAINING_CURVES_1000_UPDATE_BINS.csv',curves)
    write_csv(out/'STUDENT_RECORDED_COSTS.csv',costs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,name in zip(axes.flat,('ego_fm','current_dino','future','interaction')):
        for arm in runs:
            part=[r for r in curves if r['arm']==arm]
            key=('future_dino_mean' if arm.startswith('C') else 'future_clip_mean') if name=='future' else name+'_mean'
            ax.plot([r['update']/1000 for r in part],[r[key] for r in part],label=arm)
        ax.set(xlabel='Optimizer updates (thousands)',ylabel='Raw loss, mean per1000updates',title=name)
        ax.set_yscale('log');ax.grid(alpha=.25);ax.legend(ncol=2)
    fig.suptitle('Recorded training losses; future target definitions differ between C and S')
    fig.tight_layout();fig.savefig(out/'TRAINING_LOSS_CURVES.svg');plt.close(fig)
    print(json.dumps({'runs':len(runs),'curve_bins':len(curves),'costs':costs}))


if __name__=='__main__':main()
