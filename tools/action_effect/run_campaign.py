"""Bounded thin dispatcher for real physical-effect research stages.

This runner calls existing collection/simulation/training/evaluation entries.
Missing inputs are BLOCKED; NOT_RUN/dry runs never become experimental PASS.
It does not schedule servers, preempt jobs, run navtest, or update ACTIVE.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from .common import artifact,digest,load_config,lock,read_json,verify_artifact,verify_identity,write_json

PHASES=('audit','g0','g1','route-a','route-b','evaluation','all')
G1_ARMS=('absolute_score','terminal_pdms','absolute_feedback','delta_feedback','trajectory_only','shuffled_context','privileged_upper_bound')
G1_MATRIX=G1_ARMS[:-1]
ROUTE_ARMS=('A0','A1','ScoreRefiner','B0','B1-trajectory','B1-effect')


class Blocked(RuntimeError):pass


def evaluation_arms(args):
    if args.arm=='S0':return ['S0']
    if args.arm in ROUTE_ARMS:return ['S0',args.arm]
    if args.phase=='all' and args.arm is None:
        return ['S0','A0','B0','B1-trajectory']+(['ScoreRefiner'] if args.include_score_refiner else [])+['A1','B1-effect']
    return ['S0','A1','B1-effect']


def task_plan(args):
    """Explicit arms stay explicit; independent controls precede dependencies."""
    def task(phase,arm=None):return {'phase':phase,'arm':arm}
    if args.phase!='all':
        if args.phase=='g1' and args.g1_matrix:return [task('g1',arm) for arm in G1_MATRIX]
        if args.phase=='evaluation':return [task('evaluation',arm) for arm in evaluation_arms(args)]
        return [task(args.phase,args.arm)]
    tasks=[task('audit'),task('g0')]
    if args.arm in G1_ARMS:return tasks+[task('g1',args.arm)]
    if args.arm=='S0':return tasks+[task('evaluation','S0')]
    if args.arm in ROUTE_ARMS:
        if args.arm in ('A1','B1-effect') and not args.effect_checkpoint:tasks.append(task('g1','delta_feedback'))
        if args.arm=='ScoreRefiner' and not args.effect_checkpoint:tasks.append(task('g1','absolute_score'))
        tasks.append(task('route-a' if args.arm in ROUTE_ARMS[:3] else 'route-b',args.arm))
    else:
        tasks += [task('g1',arm) for arm in (G1_MATRIX if args.g1_matrix else ('delta_feedback',))]
        tasks += [task('route-a','A0'),task('route-b','B0'),task('route-b','B1-trajectory')]
        if args.include_score_refiner:tasks.append(task('route-a','ScoreRefiner'))
        tasks += [task('route-a','A1'),task('route-b','B1-effect')]
    return tasks+[task('evaluation',arm) for arm in evaluation_arms(args)]


def require(path):
    path=Path(path)
    if not path.is_file():raise Blocked('WAITING_INPUTS: '+str(path))
    return path


def latest_complete(run):
    """Only the named arm/run, never scan arbitrary project checkpoints."""
    import torch
    run=Path(run);identity=read_json(require(run/'IDENTITY.json'));identity_hash=digest(identity)
    valid=[]
    for p in sorted(run.glob('step_*.COMPLETE.json')):
        try:
            r=read_json(p);cp=verify_artifact(r['checkpoint'])
            state=torch.load(cp,map_location='cpu',weights_only=False)
            if r['run_identity']!=identity_hash or state['run_identity']!=identity_hash or state['optimizer_step']!=r['optimizer_step']:
                continue
            if state['optimizer_step']>0:valid.append((state['optimizer_step'],str(cp),cp))
        except Exception:continue  # A corrupt/partial receipt never outranks a valid complete one.
    if not valid:raise Blocked('WAITING_INPUTS: no complete trained checkpoint in '+str(run))
    return max(valid)[2]


def used_calls(paths):
    return sum(read_json(p/'calls'/'COUNTER.json')['reserved_calls'] for p in paths if (p/'calls'/'COUNTER.json').is_file())


def audit(cfg,replay_evidence):
    from .collect_base_trajectories import prepare
    binding=prepare(cfg)
    evidence=read_json(require(replay_evidence));receipt=read_json(verify_artifact(evidence['receipt']))
    if (evidence.get('status')!='PASS' or evidence.get('fixed_historical_raw_physical_hashes_equal') is not True
            or evidence.get('label_independence') is not True or evidence.get('strict_FP32_master_load') is not True
            or receipt.get('strict_load') is not True or receipt.get('optimizer_updates')!=0
            or receipt['parameter_buffer_sha256_after']!=receipt['parameter_buffer_sha256_before']):
        raise ValueError('Existing pinned original S0 replay evidence is incomplete/incompatible')
    return {'status':'PASS','scope':'source/binding and existing actual S0 replay evidence; no new model execution',
        'campaign_binding':binding['identity'],'replay_evidence':artifact(replay_evidence),
        'real_replay_receipt':evidence['receipt'],'scientific_gain':'UNTESTED'}


class Dispatcher:
    def __init__(self,cfg,args,folder):
        self.cfg=cfg;self.a=args;self.folder=Path(folder);self.root=Path(cfg['artifact_root'])
        self.context=Path(args.context_manifest or self.root/'contexts'/f'MANIFEST_{args.max_samples:06d}.json')
        self.g0=self.root/'g0'/f'population_{args.max_samples:06d}'
        self.pairs=self.g0/'PAIR_DATASET.json';self.repairs=self.g0/'REPAIR_DATASET.json'
        self.seed=args.seed;self.counter=0;self.invocation=time.time_ns()
        self.eval_arms=evaluation_arms(args)
        eval_count=args.evaluation_max_samples or args.max_samples
        self.evaluations=[self.root/'evaluation'/arm/f'seed_{self.seed}'/args.evaluation_role/
            f'population_{eval_count:06d}'/'score' for arm in self.eval_arms]
    def call(self,module,options,*,cpu=False):
        cmd=[sys.executable,'-m',module,'--config',self.a.config]+[str(x) for x in options]
        self.counter+=1;stem=self.folder/f'{self.invocation}_{self.counter:02d}_{module.rsplit(".",1)[-1]}'
        env=dict(os.environ)
        if cpu:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with stem.with_suffix('.stdout.log').open('w') as out,stem.with_suffix('.stderr.log').open('w') as err:
            completed=subprocess.run(cmd,stdout=out,stderr=err,env=env)
        command={'command':cmd,'returncode':completed.returncode,'stdout':artifact(stem.with_suffix('.stdout.log')),
                 'stderr':artifact(stem.with_suffix('.stderr.log'))}
        write_json(stem.with_suffix('.COMMAND.json'),command)
        if completed.returncode:raise RuntimeError('Real child stage failed; see '+str(stem)+'.stdout.log/.stderr.log')
        return command
    def resume(self,output,*,training=False):
        if not self.a.resume:return []
        if training and not list(Path(output).glob('step_*.COMPLETE.json')):return []
        return ['--resume']
    def budget(self,output):
        spent=used_calls(self.call_roots())
        current=used_calls([Path(output)])
        remaining=self.a.max_simulation_budget-spent
        if remaining<=0:raise Blocked('BLOCKED_SIMULATION_BUDGET: campaign call allowance exhausted')
        registration=Path(output)/'EVALUATION_REGISTRATION.json'
        if registration.is_file():
            original=verify_identity(read_json(registration))['max_calls']
            if original-current>remaining:raise Blocked('BLOCKED_SIMULATION_BUDGET: existing evaluation ceiling exceeds remaining total allowance')
            return original
        return remaining
    def call_roots(self):
        paths={self.g0,*self.evaluations}
        if hasattr(self,'root'):
            # Prefix populations and other explicit arms consume real calls too.
            for directory in ('g0','evaluation'):
                paths.update(p.parent.parent for p in (self.root/directory).glob('**/calls/COUNTER.json'))
        return sorted(paths)
    def teacher(self,mode='delta_feedback'):
        return Path(self.a.effect_checkpoint) if self.a.effect_checkpoint else latest_complete(self.model_run('g1',mode))
    def model_run(self,family,arm):
        return self.root/'models'/f'population_{self.a.max_samples:06d}'/family/arm/f'seed_{self.seed}'
    def reserve_updates(self,output,arm):
        progress=0
        if list(Path(output).glob('step_*.pt')):
            if not self.a.resume:raise Blocked('WAITING_EXPLICIT_RESUME: existing named training run '+str(output))
            import torch
            progress=torch.load(latest_complete(output),map_location='cpu',weights_only=False)['optimizer_step']
        if progress>self.a.max_updates:raise Blocked('BLOCKED_UPDATE_ROLLBACK: existing run exceeds requested horizon')
        allowance=min(self.a.max_updates-progress,self.a.stop_after or self.a.max_updates)
        path=self.folder/'UPDATE_BUDGET.json'
        budget=read_json(path) if path.exists() else {'ceiling':self.a.max_total_updates,'reserved_new_updates':0,'reservations':[]}
        if budget['ceiling']!=self.a.max_total_updates:raise ValueError('Immutable dispatch update ceiling changed')
        if budget['reserved_new_updates']+allowance>budget['ceiling']:
            raise Blocked('BLOCKED_OPTIMIZER_BUDGET: no implicit multiplication across arms')
        budget['reserved_new_updates']+=allowance
        budget['reservations'].append({'output':str(output),'arm':arm,'starting_step':progress,'new_update_ceiling':allowance})
        write_json(path,budget,immutable=False)
        return allowance
    def train_arm(self,arm):
        require(self.context);require(self.repairs)
        family='route_a' if arm in ('A0','A1','ScoreRefiner') else 'route_b'
        output=self.model_run(family,arm)
        common=['--context-manifest',self.context,'--repair-dataset',self.repairs,'--arm',arm,
            '--device',self.a.device,'--seed',self.seed,'--max-updates',self.a.max_updates,'--output',output]
        if self.a.stop_after:common+=['--stop-after',self.a.stop_after]
        if arm in ('A1','ScoreRefiner','B1-effect'):
            teacher=self.teacher('absolute_score' if arm=='ScoreRefiner' else 'delta_feedback')
            common+=['--effect-checkpoint',require(teacher)]
        reserved=self.reserve_updates(output,arm)
        command=self.call('tools.action_effect.train_refiner' if family=='route_a' else 'tools.action_effect.train_effect_flow',
            common+self.resume(output,training=True))
        result=read_json(require(output/'RESULT.json'))
        return {'status':'PASS' if result['status']=='COMPLETE' else 'INCONCLUSIVE',
            'scope':'bounded optimizer execution only; no heldout/scientific gain claim','training_result':artifact(output/'RESULT.json'),
            'optimizer_step':result['optimizer_step'],'reserved_new_update_ceiling':reserved,'command':command}
    def run(self,phase,arm=None):
        a=self.a
        if phase=='audit':return audit(self.cfg,a.replay_evidence)
        if phase=='g0':
            require(self.context)
            if used_calls(self.call_roots())>a.max_simulation_budget:
                raise Blocked('BLOCKED_SIMULATION_BUDGET: existing counted campaign calls exceed this ceiling')
            commands=[]
            if not (self.g0/'COMPLETE.json').exists():
                current=used_calls([self.g0]);other=used_calls(self.call_roots())-current
                budget=a.max_simulation_budget-other
                if budget<current or budget<a.max_samples:raise Blocked('BLOCKED_SIMULATION_BUDGET')
                existing=self.g0/'REGISTRATION.json'
                if existing.exists() and read_json(existing)['max_calls']!=budget:
                    raise Blocked('BLOCKED_BUDGET_REALLOCATION: immutable G0 ceiling differs; resume it explicitly')
                commands.append(self.call('tools.action_effect.simulate_interventions',
                    ['--max-samples',a.max_samples,'--max-simulation-budget',budget,'--workers',self.cfg['g0']['workers'],
                     '--context-manifest',self.context,'--max-seconds',a.max_seconds]+self.resume(self.g0),cpu=True))
            # Both tools revalidate real completion/feedback chains; a marker alone is not success.
            for module in ('build_effect_pairs','analyze_repair_opportunities'):
                commands.append(self.call('tools.action_effect.'+module,['--max-samples',a.max_samples,'--g0-root',self.g0],cpu=True))
            return {'status':'PASS','scope':'real TRAIN intervention data and opportunity analysis, not generalization',
                'pair_dataset':artifact(require(self.pairs)),'repair_dataset':artifact(require(self.repairs)),
                'analysis':artifact(require(self.g0/'G0_RESULTS.json')),'calls_so_far':used_calls([self.g0]),'commands':commands}
        if phase=='g1':
            require(self.context);require(self.pairs);arm=arm or (a.arm if a.arm in G1_ARMS else 'delta_feedback')
            if arm=='privileged_upper_bound':raise Blocked('BLOCKED_PRIVILEGED_FEATURES: no privileged artifact registered')
            output=self.model_run('g1',arm)
            options=['--context-manifest',self.context,'--pair-dataset',self.pairs,'--arm',arm,'--device',a.device,
                '--seed',self.seed,'--max-updates',a.max_updates,'--output',output]+self.resume(output,training=True)
            if a.stop_after:options+=['--stop-after',a.stop_after]
            reserved=self.reserve_updates(output,arm)
            command=self.call('tools.action_effect.train_effect_model',options)
            result=read_json(require(output/'RESULT.json'))
            return {'status':'PASS' if result['status']=='COMPLETE' else 'INCONCLUSIVE','training':artifact(output/'RESULT.json'),
                'scope':'bounded training and available validation metrics; scientific comparison not inferred',
                'reserved_new_update_ceiling':reserved,'command':command}
        if phase=='route-a':return self.train_arm(arm or (a.arm if a.arm in ROUTE_ARMS[:3] else 'A1'))
        if phase=='route-b':return self.train_arm(arm or (a.arm if a.arm in ROUTE_ARMS[3:] else 'B1-effect'))
        if phase=='evaluation':
            requested_arm=arm
            if requested_arm is not None and requested_arm not in self.eval_arms:raise ValueError('Evaluation arm absent from fixed task plan')
            results=[]
            context=(require(a.evaluation_context_manifest) if a.evaluation_role=='dev_report' else require(self.context))
            evaluation_count=a.evaluation_max_samples or a.max_samples
            for arm,score_output in zip(self.eval_arms,self.evaluations):
                if requested_arm is not None and arm!=requested_arm:continue
                family='route_a' if arm.startswith('A') or arm=='ScoreRefiner' else 'route_b'
                checkpoint=(None if arm=='S0' else Path(a.checkpoint) if a.checkpoint else latest_complete(self.model_run(family,arm)))
                module='tools.action_effect.evaluate_refiner' if family=='route_a' else 'tools.action_effect.evaluate_effect_flow'
                output=score_output.parent/'prediction'
                options=['--phase','predict',
                    '--context-manifest',context,'--output',output,'--role',a.evaluation_role,
                    '--max-samples',evaluation_count,'--device',a.device]
                options+=['--baseline-only'] if arm=='S0' else ['--checkpoint',require(checkpoint)]
                if a.evaluation_role!='dev_report':options+=['--repair-dataset',require(self.repairs)]
                if a.evaluation_binding:options+=['--campaign-binding',a.evaluation_binding]
                self.call(module,options+self.resume(output))
                self.call(module,['--phase','score','--predictions',require(output/'PREDICTIONS.json'),'--output',score_output,
                    '--max-simulation-budget',self.budget(score_output),'--max-seconds',a.max_seconds]+self.resume(score_output),cpu=True)
                result=read_json(require(score_output/'RESULT.json'))
                results.append({'arm':arm,'evaluation':artifact(score_output/'RESULT.json'),'status':result['status'],
                    'summary':result['summary']})
            return {'status':'PASS' if all(x['status']=='COMPLETE' for x in results) else 'INCONCLUSIVE',
                'scope':'fixed '+a.evaluation_role+' population; never NAVTEST','results':results,
                'scientific_conclusion':'requires paired same-budget controls and independent log statistics'}
        raise ValueError('Unknown phase')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--phase',required=True,choices=PHASES)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--max-updates',type=int);p.add_argument('--stop-after',type=int)
    p.add_argument('--max-simulation-budget',type=int);p.add_argument('--max-seconds',type=float,default=3600.)
    p.add_argument('--artifact-root');p.add_argument('--device',default='cpu');p.add_argument('--arm')
    p.add_argument('--max-samples',type=int);p.add_argument('--context-manifest');p.add_argument('--effect-checkpoint');p.add_argument('--checkpoint')
    p.add_argument('--evaluation-role',choices=('validation','dev_report'),default='validation')
    p.add_argument('--evaluation-context-manifest');p.add_argument('--evaluation-binding');p.add_argument('--evaluation-max-samples',type=int)
    p.add_argument('--replay-evidence',default=str(Path(__file__).resolve().parents[2]/'reports/effect_repair/S0_FRESH_REPLAY.json'))
    p.add_argument('--execute-ready-stages',action='store_true');p.add_argument('--continue-independent-on-failure',action='store_true')
    p.add_argument('--g1-matrix',action='store_true',help='Explicit six own G1 controls; no privileged/fake feature mode')
    p.add_argument('--include-score-refiner',action='store_true',help='Include ScoreRefiner only with its own absolute-score G1 checkpoint; missing teacher stays BLOCKED')
    p.add_argument('--max-total-updates',type=int,help='Explicit aggregate NEW optimizer-update dispatch ceiling; required for all/matrix execution')
    a=p.parse_args(argv)
    try:
        cfg=load_config(a.config)
        if a.artifact_root and Path(a.artifact_root).resolve()!=Path(cfg['artifact_root']).resolve():
            raise ValueError('Changed artifact-root requires a new resolved configuration/campaign identity')
        a.seed=cfg['seed'] if a.seed is None else a.seed
        a.max_updates=cfg['training']['max_updates'] if a.max_updates is None else a.max_updates
        a.max_samples=cfg['pilot_scenes'] if a.max_samples is None else a.max_samples
        a.max_simulation_budget=cfg['g0']['max_simulation_budget'] if a.max_simulation_budget is None else a.max_simulation_budget
        if (a.seed not in cfg['training']['training_seeds']+cfg['training']['finalist_seeds'] or
                not 1<=a.max_updates<=cfg['training']['max_updates'] or not 1<=a.max_samples<=cfg['cohort_size'] or
                not 1<=a.max_simulation_budget<=cfg['g0']['max_simulation_budget'] or
                not 0<a.max_seconds<=14400):raise ValueError('Unregistered seed or out-of-range bounded resource budget')
        if a.arm is not None and a.arm not in G1_ARMS+('S0','A0','A1','ScoreRefiner','B0','B1-trajectory','B1-effect'):
            raise ValueError('Unknown research arm')
        if a.g1_matrix and (a.phase not in ('all','g1') or a.arm is not None):raise ValueError('--g1-matrix is an explicit matrix alternative to a single --arm')
        if a.include_score_refiner and (a.phase!='all' or a.arm is not None):raise ValueError('--include-score-refiner is only for the unfiltered all plan')
        permitted={'g1':G1_ARMS,'route-a':ROUTE_ARMS[:3],'route-b':ROUTE_ARMS[3:],'evaluation':('S0',)+ROUTE_ARMS}
        if a.arm is not None and a.phase in permitted and a.arm not in permitted[a.phase]:raise ValueError('Explicit arm does not belong to requested phase')
        if a.stop_after is not None and not 1<=a.stop_after<=a.max_updates:raise ValueError('Bounded stop-after must preserve the registered training horizon')
        if a.checkpoint and a.arm not in ('A0','A1','ScoreRefiner','B0','B1-trajectory','B1-effect'):
            raise ValueError('An explicit evaluation checkpoint also requires its explicit route arm')
        if a.evaluation_role=='dev_report' and not a.evaluation_context_manifest:
            raise ValueError('DEV evaluation needs its separate complete registered context manifest')
        if a.evaluation_max_samples is not None and not 1<=a.evaluation_max_samples<=100000:
            raise ValueError('Explicit bounded evaluation sample count required')
        if a.phase=='all' and not a.execute_ready_stages and not a.dry_run:
            raise ValueError('--phase all requires explicit --execute-ready-stages')
        tasks=task_plan(a)
        needs_total=(a.phase=='all' or a.g1_matrix) and any(t['phase'] in ('g1','route-a','route-b') for t in tasks)
        if needs_total and a.max_total_updates is None and not a.dry_run:
            raise ValueError('Expanded all/matrix execution requires explicit --max-total-updates; per-arm --max-updates is not a multiplied budget')
        if a.max_total_updates is not None and not 1<=a.max_total_updates<=12*cfg['training']['max_updates']:
            raise ValueError('Explicit aggregate optimizer budget out of range')
        if a.max_total_updates is None and a.phase!='all' and not a.g1_matrix:a.max_total_updates=a.max_updates
        phases=list(PHASES[:-1]) if a.phase=='all' else [a.phase]
        if a.dry_run:
            print(json.dumps({'status':'NOT_RUN','phases':phases,'arm':a.arm,'max_samples':a.max_samples,
                'max_updates':a.max_updates,'aggregate_simulation_ceiling':a.max_simulation_budget,
                'tasks':tasks,'aggregate_new_update_ceiling':a.max_total_updates,
                'requires_explicit_update_ceiling_for_execute':needs_total and a.max_total_updates is None,
                'uninitialized_training_plan_upper_bound':sum(t['phase'] in ('g1','route-a','route-b') for t in tasks)*(a.stop_after or a.max_updates),
                'resume_requested':a.resume,'no_jobs_launched':True}));return 0
        profile=a.arm or ('g1_matrix' if a.g1_matrix else 'default')
        if a.include_score_refiner:profile+='_score_refiner'
        folder=Path(cfg['artifact_root'])/'runner'/a.phase/profile/f'seed_{a.seed}'/f'population_{a.max_samples:06d}'/a.evaluation_role
        folder.mkdir(parents=True,exist_ok=True)
        with lock(folder/'RUN.lock'):
            identity={'schema':'action_effect_campaign_dispatch_v1','config':artifact(a.config),'phase':a.phase,'arm':a.arm,
                'seed':a.seed,'max_samples':a.max_samples,'max_updates':a.max_updates,
                'tasks':tasks,'max_total_new_updates':a.max_total_updates,
                'evaluation_role':a.evaluation_role,'evaluation_context_manifest':a.evaluation_context_manifest,
                'evaluation_max_samples':a.evaluation_max_samples,
                'max_simulation_budget':a.max_simulation_budget,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
            write_json(folder/'IDENTITY.json',identity)
            dispatcher=Dispatcher(cfg,a,folder);rows=[]
            for task in tasks:
                try:result=dispatcher.run(task['phase'],arm=task['arm'])
                except Blocked as exc:result={'status':'BLOCKED','error':str(exc)}
                except Exception as exc:result={'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}
                rows.append({**task,**result})
                write_json(folder/'STATUS.json',{'status':'RUNNING','stages':rows},immutable=False)
                if result['status'] in ('FAIL','BLOCKED') and not a.continue_independent_on_failure:break
            status='PASS' if len(rows)==len(tasks) and all(r['status']=='PASS' for r in rows) else 'INCONCLUSIVE'
            receipt={'schema':'action_effect_campaign_dispatch_result_v1','status':status,'stages':rows,
                'task_plan':tasks,'max_total_new_updates':a.max_total_updates,
                'optimizer_dispatch_budget':read_json(folder/'UPDATE_BUDGET.json') if (folder/'UPDATE_BUDGET.json').exists() else None,
                'scientific_conclusion':'UNTESTED unless established by actual paired evaluations; completion is not positive gain',
                'new_navtest_started':False,'ACTIVE_updated':False}
            write_json(folder/'STATUS.json',receipt,immutable=False)
            write_json(folder/f'INVOCATION_{time.time_ns()}.json',receipt)
        print(json.dumps(receipt));return 0 if status=='PASS' else 1
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())
