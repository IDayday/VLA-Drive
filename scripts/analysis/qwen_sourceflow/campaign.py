"""Finite post-IL campaign: two source recipes, fixed DEV and locked Navtest.

This only schedules existing audited entries. It never changes an actor recipe,
restarts failed training, selects from Navtest, or uses qualification weights.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from starVLA.rl.qwen_sourceflow.contracts import ROOT,require_dispatch,file_sha,fingerprint
from starVLA.rl.qwen_sourceflow.loading import inspect_il
from starVLA.rl.qwen_sourceflow.selection import validate_selection
from starVLA.rl.qwen_sourceflow.resources import MODEL_PYTHON,allowed_hosts,available,launch,operator_alive,probe
from starVLA.rl.flow_grpo.transactions import atomic_json

VARIANTS=('canonical','source')
MILESTONES=(5000,10000,20000,35000,50000)


def read(path):
    return json.loads(Path(path).read_text())


def immutable_source(path):
    path=Path(path).resolve()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True).strip():
        raise ValueError('Campaign execution source is dirty: '+str(path))
    return dict(path=str(path),sha=sha)


def formal_endpoint(run):
    """A zero exit code alone cannot establish the registered 50k endpoint."""
    rows=[read(Path(run)/f'result_rank{rank}.json') for rank in range(8)]
    for row in rows:
        expected=dict(updates=50000,collections=12500,raw_scene_exposures=100000,
            new_candidates=800000,next_inner=0,phase='COMPLETE',status='COMPLETE')
        if row.get('qualification') or any(row[k]!=v for k,v in expected.items()):
            raise ValueError('Incomplete or unmatched formal endpoint: '+str(run))
    checkpoint=Path(rows[0]['checkpoint']);seal=read(checkpoint/'COMPLETE')
    if any(row['checkpoint']!=str(checkpoint) for row in rows):raise ValueError('Rank endpoints disagree')
    if seal.get('schema')!=2 or seal['updates']!=50000 or seal['phase']!='COMPLETE' or seal['world']!=8:
        raise ValueError('No committed eight-rank 50k endpoint')
    if file_sha(checkpoint/'files.json')!=seal['files_sha256']:
        raise ValueError('Endpoint publication changed')
    files=read(checkpoint/'files.json');identity=read(Path(run)/'identity.json')
    for name in ('optimizer.pt','identity.json'):
        if file_sha(checkpoint/name)!=files[name]:raise ValueError('Changed endpoint payload: '+name)
    if read(checkpoint/'identity.json')!=identity:raise ValueError('Endpoint belongs to a different run')
    import torch
    saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    states=saved['optimizer']['state'];count=len(identity['parameter_contract']['trainable_names'])
    if len(states)!=count or any(float(value['step'])!=50000 for value in states.values()):
        raise ValueError('Actual Adam counts do not match the 50k endpoint')
    if saved['scheduler']['last_epoch']!=50000:raise ValueError('Scheduler did not complete 50k updates')
    return dict(checkpoint=str(checkpoint),files_sha256=seal['files_sha256'],
        data_stream_identity=identity['data_stream_identity'])


def scene_stream_hash(run):
    """Hash actual first-inner scene exposures; inner reuse adds no samples."""
    result=[]
    for rank in range(8):
        digest=hashlib.sha256();expected=1;fresh=0
        with (Path(run)/f'metrics_rank{rank}.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['updates']!=expected:raise ValueError('Duplicate/gapped actual updates')
                expected+=1
                if row['inner_completed']==1:
                    digest.update(f"{row['collection']}:{row['token']}\n".encode());fresh+=1
        if expected!=50001 or fresh!=12500:raise ValueError('Incomplete source data population')
        result.append(digest.hexdigest())
    return result


class Campaign:
    def __init__(self,root,control,training_source,analysis_source):
        self.root=Path(root);self.control=Path(control)
        self.training=training_source;self.analysis=analysis_source
        (self.control/'jobs').mkdir(parents=True,exist_ok=True)
        self.selection=self.root/'frozen_IL_parent.json'

    def task_path(self,name):
        return self.control/'jobs'/(name+'.json')

    def tasks(self):
        return [read(p) for p in sorted((self.control/'jobs').glob('*.json'))]

    def done(self,name):
        path=self.task_path(name)
        if not path.exists():return False
        task=read(path);done=True
        for spec in task['workers']:
            receipt=Path(spec['receipt_dir']);result=receipt/'result.json'
            if result.exists():
                if read(result)['status']!='PROCESS_COMPLETE':
                    raise RuntimeError('No automatic retry; inspect '+str(result))
            else:
                operator=receipt/'operator.json'
                if not operator.exists() or not operator_alive(spec,read(operator)):
                    raise RuntimeError('Lost or partially dispatched invocation: '+str(receipt))
                done=False
        return done

    def reserved(self):
        uuids=set()
        for task in self.tasks():
            if not self.done(task['name']):
                for spec in task['workers']:uuids.update(spec['gpu_uuids'])
        return uuids

    def submit(self,name,module,args,routes,source,*,distributed=False,cpu_host=None):
        if self.task_path(name).exists():raise FileExistsError('Never duplicate '+name)
        workers=[]
        assignments=[routes] if distributed or not routes else [[r] for r in routes]
        for rank,assignment in enumerate(assignments):
            host=assignment[0]['host'] if assignment else cpu_host
            if host not in allowed_hosts():raise PermissionError('No longer authorized: '+str(host))
            receipt=self.control/'invocations'/name/f'rank{rank:03d}'
            receipt.mkdir(parents=True,exist_ok=False)
            command=[MODEL_PYTHON,'-u','-m']
            if distributed:
                if len(assignment)!=8 or len({r['host'] for r in assignment})!=1:
                    raise ValueError('Formal/qualification training is single-node eight-GPU')
                command+=['torch.distributed.run','--standalone','--nproc_per_node=8','-m']
            command+=[module,*args]
            if module.endswith('evaluate_policy'):
                command+=['--rank',str(rank),'--world',str(len(assignments))]
            spec=dict(source=source['path'],source_sha=source['sha'],host=host,
                gpus=[r['gpu'] for r in assignment],gpu_uuids=[r['uuid'] for r in assignment],
                min_free_MiB=60000 if distributed else 32000 if assignment else 0,
                module=module,command=command,receipt_dir=str(receipt))
            atomic_json(receipt/'spec.json',spec);workers.append(spec)
        # Publish the complete dispatch intent before any child can start.
        atomic_json(self.task_path(name),dict(name=name,workers=workers,status='DISPATCHING',time_unix=time.time()))
        for spec in workers:
            path=Path(spec['receipt_dir'])
            atomic_json(path/'operator.json',launch(path/'spec.json',source['path']))
        atomic_json(self.task_path(name),dict(name=name,workers=workers,status='RUNNING',time_unix=time.time()))

    def run_analysis(self,module,args):
        subprocess.run([MODEL_PYTHON,'-m',module,*args],cwd=self.analysis['path'],check=True)

    def training_args(self,variant,parent,output,*,qualification):
        args=['--config',f'configs/qwen_sourceflow/rl_{variant}.yaml',
            '--manifest','/mnt/project/simwam-source-process-20261018/source_manifest.json',
            '--il-checkpoint',parent['path'],'--output',str(output)]
        if qualification:args+=['--qualification','--stop-after-update','16','--save-every','4']
        else:args+=['--parent-selection',str(self.selection)]
        return args

    def ensure_training(self,name,variant,parent,output,*,qualification):
        if self.task_path(name).exists():return self.done(name)
        routes,inventory=available(self.reserved(),min_free_MiB=60000)
        atomic_json(self.control/'last_training_admission.json',dict(time_unix=time.time(),inventory=inventory))
        groups=defaultdict(list)
        for route in routes:groups[route['host']].append(route)
        full=[r for r in groups.values() if len(r)==8]
        if not full:return False
        chosen=max(full,key=lambda rows:sum(r['free_MiB'] for r in rows))
        self.submit(name,'scripts.analysis.qwen_sourceflow.train_rl',
            self.training_args(variant,parent,output,qualification=qualification),chosen,self.training,distributed=True)
        return False

    def ensure_export(self,variant,update,run):
        name=f'{variant}_update{update:06d}_export';output=self.root/'exports'/f'{variant}_update{update:06d}'
        if self.task_path(name).exists():
            if self.done(name) and not (output/'COMPLETE').exists():raise ValueError('Export exited without a sealed bundle')
            return output if self.done(name) else None
        checkpoint=Path(run)/'checkpoints'/f'update{update:06d}_inner0'
        if not (checkpoint/'COMPLETE').exists():return None
        # One CPU export at a time bounds complete-model reconstruction on the
        # shared filesystem. It occupies no GPU and changes no inference math.
        if any(t['workers'][0]['module'].endswith('export_policy') and not self.done(t['name']) for t in self.tasks()):return None
        def capacity(host):
            try:return host,probe(host)
            except Exception:return host,None
        with ThreadPoolExecutor(max_workers=max(1,len(allowed_hosts()))) as pool:
            capacity_rows=list(pool.map(capacity,allowed_hosts()))
        suitable=[(host,row) for host,row in capacity_rows if row and
            row['available_memory_bytes']>=48*1024**3 and row['CPU_affinity']>=8 and row['load1']<row['CPU_affinity']]
        if not suitable:return None
        host=min(suitable,key=lambda item:item[1]['load1']/item[1]['CPU_affinity'])[0]
        self.submit(name,'scripts.analysis.qwen_sourceflow.export_policy',
            ['--checkpoint',str(checkpoint),'--output',str(output)],[],self.analysis,cpu_host=host)
        return None

    def ensure_evaluation(self,name,parent,cohort,*,export=None):
        output=self.root/'evaluations'/name;summary=self.root/'analysis'/(name+'.json')
        if summary.exists():
            value=read(summary)
            if value['status']!='COMPLETE' or value['qualification'] or not value['full_population']:
                raise ValueError('Invalid complete evaluation summary')
            expected=export is not None
            if (value.get('rl_export') is not None)!=expected:raise ValueError('IL/RL evaluation identity mismatch')
            return summary
        if self.task_path(name).exists():
            if self.done(name):
                self.run_analysis('scripts.analysis.qwen_sourceflow.summarize_results',
                    ['--evaluation',str(output),'--output',str(summary)])
                return summary
            return None
        routes,inventory=available(self.reserved(),min_free_MiB=32000)
        atomic_json(self.control/'last_evaluation_admission.json',dict(time_unix=time.time(),inventory=inventory))
        if not routes:return None
        data=self.root/'data'/('DEV_v2' if cohort=='DEV' else 'NAVTEST')
        if not (data/'manifest.json').exists():
            # The concrete data directory is recorded before queue launch;
            # no alternate population or silent fallback is admitted here.
            data=Path(read(self.control/'identity.json')['data'][cohort])
        args=['--checkpoint',parent['path'],'--output',str(output),'--cohort',cohort,'--data',str(data)]
        if cohort=='NAVTEST':args+=['--parent-selection',str(self.selection)]
        if export is not None:args+=['--rl-export',str(export)]
        self.submit(name,'scripts.analysis.qwen_sourceflow.evaluate_policy',args,routes,self.training)
        return None

    def qualification(self,variant):
        run=self.root/'qualification'/f'selected_IL_{variant}_profile16';rows=[]
        for rank in range(8):
            row=read(run/f'result_rank{rank}.json')
            if (not row['qualification'] or row['updates']!=16 or row['collections']!=4 or
                row['raw_scene_exposures']!=32 or row['new_candidates']!=256 or
                row['frozen_SHA_before']!=row['frozen_SHA_after'] or
                row['frozen_observation_before']!=row['frozen_observation_after']):
                raise ValueError('Selected-parent qualification failed')
            for collection in range(1,5):
                import torch
                batch=torch.load(run/'behavior'/f'collection{collection:06d}'/f'rank{rank}.pt',map_location='cpu',weights_only=False)
                if batch['qualification_sampling_check']['status']!='PASS':raise ValueError('Actual sampling/recompute differs')
            rows.append(row)
        phases=('rollout','scoring_wait','old_probability','update')
        per_collection=max(sum(r['seconds_'+p] for p in phases) for r in rows)/4
        return dict(qualification=True,updates=16,fresh_global_rollouts=4,candidates=256,
            slowest_rank_phase_seconds_per_collection=per_collection,
            projected_formal_seconds=per_collection*12500,
            estimate_scope='Four fresh rollouts, excludes formal loading/save/evaluation and future contention; not a promised speedup',
            peak_GPU_bytes=max(r['peak_GPU_bytes'] for r in rows))

    def tick(self):
        require_dispatch()
        if not self.selection.exists():return dict(status='WAITING_FOR_FULL_IL_AND_FIVE_DEV')
        selection=read(self.selection);parent=inspect_il(selection['selected']['checkpoint'])
        validate_selection(self.selection,parent)
        registration=self.control/'registration.json'
        body=dict(schema='native_two_RL_formal_campaign_v1',IL_parent=parent['path'],IL_model_sha256=parent['files']['model.pt'],
            IL_selection_sha256=file_sha(self.selection),training=self.training,analysis=self.analysis,
            training_seed=42,inference_seed=2101,group_size=8,denoising_steps=10,
            variants={v:dict(optimizer_updates=50000,fresh_global_rollouts=12500,scene_exposures=100000,candidates=800000,
                config_sha256=file_sha(Path(self.training['path'])/f'configs/qwen_sourceflow/rl_{v}.yaml')) for v in VARIANTS},
            DEV_updates=[0,*MILESTONES],RL_selection_rule='highest DEV canonical PDMS among 5k/10k/20k/35k/50k; exact tie later',
            primary_comparison='fixed 50k endpoints; DEV-selected trained checkpoints separately reported',
            final_NAVTEST='Only after both 50k endpoints and all registered DEV results are locked; seed2101 single shift5 ODE',
            qualification_weights_excluded=True)
        if registration.exists() and read(registration)!=body:raise ValueError('Registered campaign identity changed')
        if not registration.exists():atomic_json(registration,body)
        qualified=True
        for variant in VARIANTS:
            name=variant+'_selected_IL_profile16';run=self.root/'qualification'/f'selected_IL_{variant}_profile16'
            if not self.ensure_training(name,variant,parent,run,qualification=True):qualified=False
        if not qualified:return dict(status='SELECTED_PARENT_QUALIFICATION',IL_parent=parent['path'])
        if not (self.control/'qualified_cost.json').exists():
            atomic_json(self.control/'qualified_cost.json',{v:self.qualification(v) for v in VARIANTS})
        complete={};dev={};runs={}
        for variant in VARIANTS:
            run=self.root/'runs'/f'RL_{variant.upper()}_seed42';runs[variant]=run
            if self.ensure_training(variant+'_formal50k',variant,parent,run,qualification=False):
                complete[variant]=formal_endpoint(run)
            dev[variant]={}
            for update in MILESTONES:
                export=self.ensure_export(variant,update,run)
                if export is not None:
                    summary=self.ensure_evaluation(f'RL_{variant}_update{update:06d}_DEV',parent,'DEV',export=export)
                    if summary is not None:dev[variant][update]=dict(summary=str(summary),summary_sha256=file_sha(summary),PDMS=read(summary)['PDMS'],export=str(export))
        # The IL model was already fixed using DEV. This independent baseline
        # never influences the two pre-registered RL configs or their selection.
        baseline=self.ensure_evaluation('selected_IL_NAVTEST',parent,'NAVTEST')
        if len(complete)!=2 or any(len(dev[v])!=len(MILESTONES) for v in VARIANTS):
            return dict(status='FORMAL_RL_AND_DEV',complete_variants=list(complete),complete_DEV={v:list(dev[v]) for v in VARIANTS})
        locked=self.control/'RL_ENDPOINTS_AND_SELECTION_LOCKED.json'
        choices={v:max(dev[v],key=lambda u:(dev[v][u]['PDMS'],u)) for v in VARIANTS}
        lock_body=dict(schema='native_RL_fixed_end_and_DEV_selection_v1',IL_selection_sha256=file_sha(self.selection),
            endpoints=complete,DEV=dev,selected_updates=choices,selection_rule=body['RL_selection_rule'],Navtest_used=False)
        if not locked.exists():
            streams={v:scene_stream_hash(runs[v]) for v in VARIANTS}
            if streams['canonical']!=streams['source']:raise ValueError('Actual scene exposure order differs between matched runs')
            lock_body['matched_actual_scene_stream_hashes']=streams['canonical']
            atomic_json(locked,lock_body)
        else:
            existing=read(locked)
            if existing['endpoints']!=complete or existing['DEV']!={v:{str(k):value for k,value in dev[v].items()} for v in VARIANTS}:
                raise ValueError('Locked endpoints/DEV changed')
        results={};pending=False
        for variant in VARIANTS:
            results[variant]={}
            for update in sorted({50000,choices[variant]}):
                value=self.ensure_evaluation(f'RL_{variant}_update{update:06d}_NAVTEST',parent,'NAVTEST',export=dev[variant][update]['export'])
                if value is None or baseline is None:pending=True;continue
                paired=self.root/'analysis'/f'RL_{variant}_update{update:06d}_NAVTEST_paired.json'
                if not paired.exists():
                    self.run_analysis('scripts.analysis.qwen_sourceflow.analyze_paired',
                        ['--evaluation',str(self.root/'evaluations'/f'RL_{variant}_update{update:06d}_NAVTEST'),
                         '--reference',str(self.root/'evaluations/selected_IL_NAVTEST'),'--output',str(paired)])
                results[variant][update]=dict(summary=str(value),paired=str(paired))
        if pending:return dict(status='LOCKED_FINAL_NAVTEST',selected_updates=choices)
        for variant,run in runs.items():
            target=self.root/'analysis'/f'RL_{variant}_training50k.json'
            if not target.exists():self.run_analysis('scripts.analysis.simwam_recipe.summarize_training',['--run',str(run),'--output',str(target)])
        final=dict(schema='native_Qwen_flow_campaign_completion_v1',status='COMPLETE',IL_navtest_summary=str(baseline),
            results=results,fixed_endpoint_updates=50000,DEV_selected_updates=choices,
            training_seeds=[42],inference_seeds=[2101],registration_sha256=file_sha(registration),
            endpoint_lock_sha256=file_sha(locked),reached92={v:{str(u):read(x['paired'])['reached92'] for u,x in rows.items()} for v,rows in results.items()})
        atomic_json(self.control/'COMPLETE.json',final)
        return final


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default=str(ROOT));p.add_argument('--control-dir',required=True)
    p.add_argument('--training-source',required=True);p.add_argument('--analysis-source',required=True)
    p.add_argument('--once',action='store_true');a=p.parse_args();require_dispatch()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Campaign owner must be CPU-only')
    control=Path(a.control_dir);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    train=immutable_source(a.training_source);analysis=immutable_source(a.analysis_source);root=Path(a.root)
    nav=[d for d in (root/'data').iterdir() if d.is_dir() and 'NAVTEST' in d.name.upper()]
    if len(nav)!=1:raise ValueError('Require one concrete locked Navtest data directory')
    identity=dict(schema='native_finite_campaign_owner_v1',training=train,analysis=analysis,
        root=str(root.resolve()),data=dict(DEV=str(root/'data/DEV_v2'),NAVTEST=str(nav[0])),
        qualification_updates=16,formal_updates=50000,variants=list(VARIANTS),milestones=list(MILESTONES),
        best_DEV_rule='highest canonical PDMS among trained milestones; exact tie later update',
        no_automatic_retry=True,no_recipe_changes=True)
    if (control/'identity.json').exists() and read(control/'identity.json')!=identity:raise ValueError('Campaign owner identity changed')
    atomic_json(control/'identity.json',identity);atomic_json(control/'owner.json',dict(pid=os.getpid(),start_unix=time.time()))
    queue=Campaign(root,control,train,analysis)
    try:
        while True:
            value=queue.tick();atomic_json(control/'status.json',dict(observed_unix=time.time(),**value))
            if a.once or value['status']=='COMPLETE' or (control/'STOP_AFTER_TICK').exists():break
            time.sleep(300)
    except BaseException:
        atomic_json(control/'failure.json',dict(time_unix=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
