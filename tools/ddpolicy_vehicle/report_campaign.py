"""Read-only campaign report, optionally waiting and publishing this task's results.

Never launches/stops training. Publication uses a new detached worktree, adds
only this report and paired-score tables, and performs a normal fast-forward
push to the explicitly named task branch. Raw data/weights/images stay local.
"""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from .prepare_data import atomic_json
from .campaign import charged_gpu_hours
from .summarize_training import summarize
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def read(path):return json.loads(Path(path).read_text())


def benchmark_table(root, model, tag, split, seeds, index):
    population={r['token']:r['log'] for r in read(index)}
    metrics={'PDMS':'score','NC':'no_at_fault_collisions','DAC':'drivable_area_compliance',
             'TTC':'time_to_collision_within_bound','EP':'ego_progress','Comfort':'comfort'}
    samples=[]
    for seed in seeds:
        folder=root/'formal_evaluation'/model['run_id']/tag/split/f'scores_seed{seed}'
        summary=read(folder/'summary.json')
        with (folder/'scenes.csv').open() as stream:rows=list(csv.DictReader(stream))
        if len(rows)!=len(population) or {r['token']:r['log'] for r in rows}!=population:
            raise ValueError('Incomplete/changed benchmark population')
        failed=sum(r['status']!='ok' for r in rows)
        if not summary['valid'] or failed or summary['failed']:raise ValueError('Failed rows cannot enter a valid benchmark table')
        sample={key:float(np.mean([float(r[field]) for r in rows]))*100 for key,field in metrics.items()}
        if not all(np.isfinite(x) for x in sample.values()):raise ValueError('Nonfinite metric')
        sample.update(sampling_seed=seed,zero_scenes=sum(float(r['score'])==0 for r in rows),failed_scenes=failed,
                      scenes_csv_sha256=file_sha256(folder/'scenes.csv'))
        samples.append(sample)
    return {'arm':model['arm'],'training_seed':model['seed'],'checkpoint':tag,'scenes':len(population),
            'logs':len(set(population.values())),'scale':'0..100','mean':{k:float(np.mean([s[k] for s in samples])) for k in metrics},
            'mean_zero_scenes':float(np.mean([s['zero_scenes'] for s in samples])), 'samples':samples}


def vehicle_table(root, model, tag, split, seeds, scenes):
    samples=[]
    for seed in seeds:
        path=root/'formal_evaluation'/model['run_id']/tag/split/f'vehicles_seed{seed}'/'summary.json'
        value=read(path)
        if value['scenes']!=scenes or value['failed_scenes']:
            raise ValueError('Incomplete/failed vehicle population cannot be summarized')
        samples.append({'sampling_seed':seed,'summary_sha256':file_sha256(path),**value})
    for key in ('source_vehicle_population','supervised_targets'):
        if len({s[key] for s in samples})!=1:raise ValueError('Vehicle population changed across sampling seeds')
    return {'arm':model['arm'],'training_seed':model['seed'],'checkpoint':tag,'samples':samples,
            'aggregation':'Per-sampling-seed coverage and conditional errors retained; no averaging across changing matched populations'}


def build_report(plan, controller, output):
    root=Path(plan['campaign_root']);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    state=read(Path(controller)/'status.json');models=[]
    for item in plan['models']:
        run=root/'training'/item['run_id'];row={k:item[k] for k in ('arm','seed','run_id','config_sha256')}
        if not (run/'status.json').exists():row['status']='NOT_RUN';models.append(row);continue
        status=read(run/'status.json');attempts=[read(p) for p in run.glob('attempt_*.json')]
        row.update(status=status['status'],completed_updates=status['real_optimizer_updates'],scene_presentations=status['sample_presentations'],
            actual_update_calls=sum(a.get('attempt_optimizer_updates',0) for a in attempts),
            gpu_hours=sum(((time.time() if a['status']=='RUNNING' else a['end_unix'])-a['start_unix'])*a['gpu_count']/3600 for a in attempts),
            source_sha=status['source_sha'],identity=read(run/'identity.json')['sha256'])
        if (run/'driving_initialization.json').exists():
            name=item['run_id']+'_initialization.json';shutil.copyfile(run/'driving_initialization.json',out/name);row['initialization_manifest_sha256']=file_sha256(out/name)
        if (run/'parameters.json').exists():
            parameters=read(run/'parameters.json')
            if parameters['pretrained_driving_weights_loaded']:raise ValueError('Forbidden driving-weight initialization')
            row['parameters']={k:parameters[k] for k in ('total','trainable','policy_initialization','pretrained_driving_weights_loaded')}
            if not (out/'GENERIC_SOURCES.json').exists():atomic_json(out/'GENERIC_SOURCES.json',parameters['generic_sources'])
        if status['status'] in ('COMPLETE','PAUSED') and (run/'train_rank0.jsonl').exists():
            row['learning_and_exposure']=summarize(run)[0]
        models.append(row)
    benchmark_complete=state['status']=='COMPLETE' and (Path(controller)/'analysis/COMPLETE.json').exists()
    report={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_status':state['status'],'controller_phase':state.get('phase'),
        'primary_benchmark_complete':benchmark_complete,
        'full_experiment_complete':benchmark_complete and all(m['status']=='COMPLETE' and m.get('completed_updates')==100000 for m in models),
        'training_source_sha':plan['source_sha'],'report_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'charged_campaign_gpu_hours':charged_gpu_hours(root),'gpu_hour_cap':plan['gpu_hour_cap'],
        'models':models,'benchmarks':{},'vehicles':{},'development_selection_grid':[],
        'paired_results':{},'fixed_sampling_seeds':plan['sampling_seeds'],
        'scope':'camera-only; generic pretrained base and random driving modules; single joint ego slot0; no scorer or oracle',
        'training_data':{'train_scenes':101592,'train_logs':1176,'dev_scenes':1696,'dev_logs':16},
        'uncertainty':'Whole-log paired intervals do not represent training-seed variability. Both training seeds, when complete, are reported separately.',
        'resume_limit':'Independent continuous4 versus2+2 failed strict full-master tolerance; RNG states match. No bitwise guarantee.'}
    selection_path=Path(controller)/'selection.json'
    if report['primary_benchmark_complete']:
        selected={Path(v['run']).name:v['tag'] for v in read(selection_path)}
        for split in ('dev','navtest'):
            report['benchmarks'][split]=[benchmark_table(root,m,selected[m['run_id']],split,plan['sampling_seeds'],
                Path(plan['data']['current_'+split])/'index.json') for m in plan['models'] if m['run_id'] in selected]
            report['vehicles'][split]=[vehicle_table(root,m,selected[m['run_id']],split,plan['sampling_seeds'],
                len(read(Path(plan['data']['current_'+split])/'index.json'))) for m in plan['models'] if m['run_id'] in selected]
        report['development_selection_grid']=[benchmark_table(root,m,tag,'dev',plan['sampling_seeds'],
            Path(plan['data']['current_dev'])/'index.json') for m in plan['models'] if m['run_id'] in selected
            for tag in plan['development_tags']]
        for folder in sorted((Path(controller)/'analysis').iterdir()):
            if folder.is_dir() and (folder/'summary.json').exists() and '_vs_' in folder.name:
                report['paired_results'][folder.name]=read(folder/'summary.json')
                if folder.name.endswith('_vehicles'):
                    for metric in report['paired_results'][folder.name]['metrics'].values():
                        for group in metric.values():
                            interval=group['right_minus_left']
                            if interval and interval.get('method'):
                                interval['method']='vehicle-weighted paired log-cluster percentile bootstrap'
                if (folder/'paired_scenes.csv').exists():shutil.copyfile(folder/'paired_scenes.csv',out/(folder.name+'_paired_scenes.csv'))
        # Regenerate report graphics with this report source. Training source
        # stays frozen; arm colors must remain stable on panels omitting A.
        curves=out.with_name(out.name+'_training_curves')
        subprocess.run([sys.executable,'-m','tools.ddpolicy_vehicle.summarize_training',
            '--runs',*[str(root/'training'/m['run_id']) for m in plan['models'] if m['run_id'] in selected],
            '--output',str(curves)],check=True)
        shutil.copyfile(curves/'learning_curves.png',out/'learning_curves.png')
    atomic_json(out/'REPORT.json',report)
    lines=['# DDP vehicle-only from-scratch campaign', '',
        '**'+('COMPLETE' if report['full_experiment_complete'] else 'INCOMPLETE (see per-run status)')+'**. Snapshot: '+report['snapshot_utc'], '',
        'Training source: `'+plan['source_sha']+'`. Report source: `'+report['report_source_sha']+'`.', '',
        'Generic Qwen/Wan/PPD sources and initialization tensor hashes accompany this report. No trained driving policy initializes a formal arm. A is the original framework baseline; B adds vehicle box/motion supervision and joint DiT states; C adds the matched role auxiliary. Final ego is joint slot0.', '',
        f"Charged campaign cost: {report['charged_campaign_gpu_hours']:.2f}/{plan['gpu_hour_cap']} GPU-hours, including preparation, failed/no-op attempts, loading, diagnostics and evaluation.", '',
        '| Arm | Seed | Status | Updates | Scene presentations | GPUh |', '|---|---:|---|---:|---:|---:|']
    for m in models:lines.append(f"|{m['arm']}|{m['seed']}|{m['status']}|{m.get('completed_updates',0)}|{m.get('scene_presentations',0)}|{m.get('gpu_hours',0):.2f}|")
    for split,rows in report['benchmarks'].items():
        lines+=['', '## '+split+' — complete population, five fixed single-sample scores', '',
                '| Arm | Training seed | Selected checkpoint | PDMS | NC | DAC | TTC | EP | Comfort | Mean zero scenes |',
                '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:lines.append('|'+row['arm']+'|'+str(row['training_seed'])+'|'+row['checkpoint']+'|'+ '|'.join(f'{row["mean"][k]:.3f}' for k in ('PDMS','NC','DAC','TTC','EP','Comfort'))+f'|{row["mean_zero_scenes"]:.1f}|')
    lines+=['', '## Paired mask contribution', '', 'C−B is reported for each completed training seed. Positive PDMS differences favor MASK; no favorable seed or scenario subset is selected.', '']
    for name,value in report['paired_results'].items():
        if name.startswith('navtest_B_vs_C') and 'score' in value:
            delta=value['score']['right_minus_left'];ci=delta['ci95']
            lines.append(f"- {name}: C−B = {delta['mean']*100:.3f} points; log-cluster95% interval = {None if ci is None else [round(x*100,3) for x in ci]}.")
    if not report['primary_benchmark_complete']:lines+=['Full development/Navtest conclusions are **NOT_AVAILABLE**. Existing partial checkpoints and logs remain in the campaign artifacts. Startup/small-fit scores are not substituted.']
    elif not report['full_experiment_complete']:lines+=['The primary benchmark is complete, but the planned second training-seed pair is incomplete. Its real progress/costs are retained above; these results do not establish training-seed stability.']
    lines+=['', 'Vehicle coverage, stationary/moving results, all registered development milestones and fixed shared-target comparisons are in `REPORT.json`. Per-seed error denominators are retained; different matched target sets are not averaged as if fixed. Misses remain in coverage denominators; motion error on detected/selected vehicles is conditional on that coverage. Relative ego/vehicle error uses one joint sample. Official safety scoring keeps all object classes.', '',
            'Reaching100000updates does not automatically establish convergence. The complete development checkpoint grid and training curves are retained to assess late gains or overfitting; no post-Navtest recipe adjustment is made.', '',
            report['uncertainty'], '', report['resume_limit'], '',
            'Complete machine-readable evidence and prediction banks are local to the registered campaign artifact root; the uploaded scene tables contain derived official scores and public scene/log identifiers, without sensor data or trajectories.', '',
            'If the controller has actually stopped, resume the same frozen source and plan using:', '', '```bash',
            'cd '+plan['worktree'], plan['python']+' -m tools.ddpolicy_vehicle.campaign --plan '+str(root/'formal_campaign_v1/plan.json')+' --directory '+str(controller)+' --resume --acknowledge-stop','```']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def publish_report(repo, branch, output, publication):
    if not re.fullmatch(r'feature/ddpolicy-vehicle-joint-from-scratch-\d{8}',branch):raise ValueError('Only this task branch may be published')
    repo=Path(repo);out=Path(output);publication=Path(publication)
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=repo,text=True).strip()
    if remote not in ('git@github.com:IDayday/VLA-Drive.git','https://github.com/IDayday/VLA-Drive.git'):
        raise ValueError('Unexpected report destination')
    nonce=str(time.time_ns());ref='refs/codex-reports/'+nonce
    subprocess.run(['git','fetch','origin',f'refs/heads/{branch}:{ref}'],cwd=repo,check=True)
    base=subprocess.check_output(['git','rev-parse',ref],cwd=repo,text=True).strip()
    subprocess.run(['git','worktree','add','--detach',str(publication),base],cwd=repo,check=True)
    relative=Path('reports/ddpolicy_vehicle_from_scratch/full_campaign_results')/nonce;destination=publication/relative;destination.mkdir(parents=True)
    for source in out.iterdir():
        if not source.is_file() or source.stat().st_size>30_000_000:raise ValueError('Unexpected/oversize report artifact')
        if source.suffix not in ('.json','.md','.csv','.png') or (source.suffix=='.png' and source.name!='learning_curves.png'):
            raise ValueError('Only aggregate reports, paired scores and a training curve may be published')
        shutil.copyfile(source,destination/source.name)
    subprocess.run(['git','add',str(relative)],cwd=publication,check=True)
    subprocess.run(['git','commit','-m','Record actual DDP from-scratch campaign results and completion status'],cwd=publication,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=publication,text=True).strip()
    subprocess.run(['git','push','origin',f'HEAD:refs/heads/{branch}'],cwd=publication,check=True)
    remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=publication,text=True).split()[0]
    if remote_sha!=sha:raise RuntimeError('Remote SHA changed; retain local report commit and inspect')
    return {'commit':sha,'remote_verified':True,'branch':branch,'report_path':str(relative)}


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('plan','controller','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--publish-repo');p.add_argument('--branch')
    a=p.parse_args();plan=read(a.plan);controller=Path(a.controller)
    watch_lock=None
    if a.watch:
        watch_lock=Path(a.output).with_name(Path(a.output).name+'_WATCH.lock').open('a+')
        fcntl.flock(watch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(Path(a.output).with_name(Path(a.output).name+'_observer.json'),
            {'status':'WATCHING','pid':os.getpid(),'plan_sha256':file_sha256(a.plan),
             'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'behavior':'read only until controller terminates; no GPU allocation; report/push only to explicit task branch'})
        while read(controller/'status.json')['status']=='RUNNING':
            state=read(controller/'status.json')
            try:os.kill(state['pid'],0)
            except ProcessLookupError:break
            time.sleep(30)
    build_report(plan,controller,a.output)
    if a.publish_repo:
        if not a.branch:raise ValueError('Explicit destination branch required')
        status_path=Path(a.output).with_name(Path(a.output).name+'_publication.json')
        try:
            result=publish_report(a.publish_repo,a.branch,a.output,Path(a.output).with_name('report_publication_'+str(time.time_ns())))
        except Exception as error:
            atomic_json(status_path,{'status':'FAILED','error':repr(error),'local_report_preserved':a.output});raise
        atomic_json(status_path,{'status':'COMPLETE',**result})
    if watch_lock is not None:
        atomic_json(Path(a.output).with_name(Path(a.output).name+'_observer.json'),
            {'status':'COMPLETE','pid':os.getpid(),'report':a.output})
        watch_lock.close()


if __name__=='__main__':main()
