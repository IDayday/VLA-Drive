"""Independent single-arm B64/G16 campaign; never adopts F/G/H owners."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT_PATH = Path('/mnt/project/qwen-world-safety-exploration-20261031/integrated_training_20261102')
CONTROL = ROOT_PATH.parent/'standard_training_20261101'
PROTECTED = {'training-vla-zt', 'training-vlawm-tsinghua-zt'}
IMMUTABLE = ('id', 'root', 'source', 'source_commit', 'config', 'config_sha256',
             'parent_checkpoint', 'owner_state', 'group_size', 'trainer_module',
             'execution_source', 'execution_migration_receipt')
USER_10K_POLICY = 'world_integrated64_user_minimum10k_eval2k5k10k_v1'
USER_EVALUATION_GATES = [2000, 5000, 10000]


def validate_runtime_budget(run):
    """Runtime authorization is independent of the frozen scientific LR clock."""
    from starVLA.rl.qwen_world_safety_exploration.integrated64_recipe import EVALUATE_UPDATES
    if run.get('runtime_budget_policy') == USER_10K_POLICY:
        retained = run.get('retained_native_handoff_gates')
        if (run['target_updates'] != 10000 or retained != [250,625]
                or run.get('evaluation_gates') != USER_EVALUATION_GATES
                or run.get('historical_evaluation_gates') != [250]
                or run.get('minimum_committed_updates') != 10000
                or run.get('score_based_stopping_before_minimum') is not False
                or run.get('post10k_long_training_review_required') is not True
                or run['gates'] != sorted(retained + USER_EVALUATION_GATES)):
            raise ValueError('User explicitly requires actual10k before review and evaluations2k/5k/10k')
    elif (run.get('runtime_budget_policy') is not None
            or run['target_updates'] not in (6250,12500)
            or run['gates'] != [u for u in EVALUATE_UPDATES if u <= run['target_updates']]):
        raise ValueError('Unknown integrated runtime budget/cadence')
    return run


def evaluation_gates(run, *, include_history=True):
    targets = list(run.get('evaluation_gates',run['gates']))
    if include_history:
        targets = sorted(set(targets + run.get('historical_evaluation_gates',[])))
    return targets


def evaluation_registry(value):
    """Keep native handoffs out of the GPU evaluation queue, retaining old results."""
    view = copy.deepcopy(value)
    for run in view['runs']:
        run['gates'] = evaluation_gates(run)
    return view


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    loaded = sys.modules.get('starVLA.rl.qwen_world_safety_exploration.contracts')
    if loaded is not None and loaded.ROOT.resolve() != ROOT_PATH:
        raise RuntimeError('Use a fresh CPU interpreter for the independent I campaign')
    os.environ.update(WORLD_COMPARE_ROOT=str(ROOT_PATH),
                      WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031',
                      CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1', LC_ALL='C.UTF-8')


def validate_registry(value):
    import yaml
    from starVLA.rl.qwen_world_safety_exploration.contracts import file_sha
    from starVLA.rl.qwen_world_safety_exploration.integrated64_recipe import (
        ARM, validate_integrated_recipe)
    if (value.get('schema') != 'world_integrated64_campaign_registry_v1'
            or value.get('task') != 'qwen_world_safety_exploration_20261031'
            or value.get('root') != str(ROOT_PATH)
            or value.get('training_population') != 103288
            or len(value.get('runs', [])) != 1):
        raise ValueError('Only the explicitly registered independent integrated arm')
    r = value['runs'][0]
    if (r['id'] != ARM or r['root'] != str(ROOT_PATH/'runs'/ARM)
            or r['owner_state'] != str(ROOT_PATH/'owners/INTEGRATED_NATIVE/STATE.json')
            or r['group_size'] != 16
            or r['trainer_module'] != 'tools.qwen_world_safety_exploration.train'
            or file_sha(r['config']) != r['config_sha256']
            or not (Path(r['source'])/'tools/qwen_world_safety_exploration/train.py').is_file()
            or not r['deployment_candidates']
            or any(p['host'] in PROTECTED for p in r['deployment_candidates'])):
        raise ValueError('Integrated source/config/group/cadence/native binding changed')
    validate_runtime_budget(r)
    validate_integrated_recipe(yaml.safe_load(Path(r['config']).read_text()))
    if r.get('execution_source') is not None:
        if (not r.get('execution_migration_receipt')
                or not Path(r['execution_migration_receipt']).is_file()
                or not (Path(r['execution_source'])/'tools/qwen_world_safety_exploration/train.py').is_file()):
            raise ValueError('Execution migration requires its explicit frozen source and receipt')
    f = read(CONTROL/'RUNS.json'); parent = next(x for x in f['runs'] if x['id']=='SAFETY_G8_CONTROL')
    if (value['manifest'] != f['manifest'] or value['observation_data'] != f['observation_data']
            or r['parent_checkpoint'] != parent['parent_checkpoint']):
        raise ValueError('Same full TRAIN population and fresh IL70 parent required')
    return value


def checkpoint_at_gate(run, updates):
    from starVLA.rl.qwen_world_safety_exploration.contracts import file_sha
    from starVLA.rl.qwen_world_safety_exploration.integrated64_recipe import budget
    native = read(run['owner_state']).get('runs', {}).get(run['id'], {})
    stage = native.get('stages', {}).get(str(updates), {})
    if not stage.get('complete'):
        return None, native
    p = Path(stage['checkpoint']).resolve()
    if not p.is_relative_to(Path(run['root']).resolve()/'checkpoints'):
        raise ValueError('Foreign integrated checkpoint')
    seal, identity = read(p/'COMPLETE'), read(p/'identity.json')
    if (seal.get('updates') != updates or seal.get('next_inner') != updates % 4
            or seal.get('world') != 8 or seal.get('files_sha256') != file_sha(p/'files.json')
            or identity.get('purpose') != 'formal' or identity.get('world') != 8
            or identity['recipe']['research']['arm'] != run['id']
            or any(not (p/f'rank{i}.pt').is_file() for i in range(8))):
        raise ValueError('Complete formal integrated eight-rank boundary required')
    return dict(path=str(p), updates=updates, next_inner=updates % 4,
                files_sha256=seal['files_sha256'], identity_sha256=file_sha(p/'identity.json'),
                actual_collection_budget=budget(updates)), native


def commit_completed_stage(owner, run, target, node, checkpoint):
    """Persist the completed gate before its evaluator reads the owner journal."""
    arm=run['id'];node.update(complete=True,checkpoint=str(checkpoint),end_unix=time.time())
    owner.update(arm,status='STAGE_COMPLETE',updates=target,checkpoint=str(checkpoint))
    material,_=checkpoint_at_gate(run,target)
    if material is None:
        raise RuntimeError('Completed integrated gate was not persisted')
    owner.update(arm,actual_collection_budget=material['actual_collection_budget'])
    return str(checkpoint)


def native_class():
    from .owner import Owner, StopBeforeDispatch
    from .dispatch import launch, job_state
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts
    class IntegratedOwner(Owner):
        def __init__(self):
            super().__init__(owner_name='INTEGRATED_NATIVE')
            validate_registry(self.registry)
            self.binding = {k:self.runs[0].get(k) for k in IMMUTABLE}

        def current(self):
            self.registry = validate_registry(read(ROOT_PATH/'RUNS.json'))
            run = self.registry['runs'][0]
            if {k:run.get(k) for k in IMMUTABLE} != self.binding:
                raise ValueError('Integrated scientific/native lineage cannot change while running')
            return run

        def candidate(self, run):
            return super().candidate(dict(run, deployment_candidates=[p for p in run['deployment_candidates']
                if p['host'] in allowed_hosts(admit_long_job=True) and p['host'] not in PROTECTED]))

        def run_stage(self, initial, target, resume):
            arm = initial['id']; node = self.stage(arm,target)
            if node.get('complete'):
                return node['checkpoint']
            while True:
                run = self.current()
                if not node.get('job'):
                    if (ROOT_PATH/'STOP_REQUESTED').exists():
                        raise StopBeforeDispatch('Integrated user stop before dispatch')
                    self.update(arm,status='WAITING_AUTHORIZED_RESOURCE',stage_target_updates=target)
                    chosen = self.candidate(run)
                    if chosen is None:
                        time.sleep(10); continue
                    route,sample = chosen
                    latest = Path(run['root'])/'latest.json'
                    if latest.exists():
                        saved=read(latest);resume=saved['checkpoint']
                        if saved['updates'] >= target:
                            raise ValueError('Existing gate requires explicit adoption; never start another lineage')
                    attempt=int(node.get('attempt',0))+1;name=f'{arm}/formal_to{target:06}_attempt{attempt:03}'
                    command=[MODEL_PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1',
                        '--nproc_per_node=8','-m',run['trainer_module'],'--config',run['config'],
                        '--manifest',self.registry['manifest'],'--il-checkpoint',run['parent_checkpoint'],
                        '--data',self.registry['observation_data'],'--parent-selection',self.registry['parent_selection'],
                        '--output',run['root'],'--stop-after-update',str(target),'--save-every','125']
                    if resume:
                        command+=['--resume',resume]
                        if run.get('execution_migration_receipt'):
                            command+=['--migration-receipt',run['execution_migration_receipt']]
                    try:
                        job=launch(route['host'],run.get('execution_source',run['source']),command,name,gpus='0,1,2,3,4,5,6,7',
                            expected_uuids=[g['uuid']for g in sample['gpus']],
                            share_minimum=route['minimum_free_MiB'],lease_slot=route['lease_slot'])
                    except Exception:
                        node.update(attempt=attempt,job=str(ROOT_PATH/'jobs'/name),dispatch_error=traceback.format_exc())
                        self.update(arm,status='DISPATCH_TRANSPORT_REVIEW',job=node['job']);raise
                    node.update(attempt=attempt,job=str(job),host=route['host'],resume=resume)
                if (ROOT_PATH/'STOP_REQUESTED').exists():
                    (Path(node['job'])/'STOP_REQUESTED').touch(exist_ok=True)
                receipt=job_state(node['job']);latest=Path(run['root'])/'latest.json'
                updates=read(latest)['updates']if latest.exists()else 0
                self.update(arm,status='TRAINING'if receipt.get('status')=='RUNNING'else receipt.get('status','STARTING'),
                            updates=updates,job=node['job'],host=node['host'],stage_target_updates=target)
                if receipt.get('status')=='REFUSED_RESOURCE':
                    if receipt.get('returncode')!=75 or receipt.get('child_not_started')is not True:
                        raise RuntimeError('Ambiguous refusal cannot be automatically retried')
                    node.setdefault('refused_jobs',[]).append(node.pop('job'));time.sleep(10);continue
                if 'returncode'not in receipt:
                    time.sleep(5);continue
                if receipt['returncode']!=0:
                    raise RuntimeError('Scientific/transport failure retained: '+node['job'])
                saved=read(latest);p=Path(saved['checkpoint']);seal=read(p/'COMPLETE')
                if (seal['world']!=8 or seal['updates']!=saved['updates']
                        or any(read(Path(run['root'])/f'result_rank{i}.json')['checkpoint']!=str(p) for i in range(8))):
                    raise ValueError('Missing actual eight-rank native completion')
                if saved['updates']!=target:
                    if (ROOT_PATH/'STOP_REQUESTED').exists():
                        self.update(arm,status='STOPPED_AT_COOPERATIVE_BOUNDARY',checkpoint=str(p));raise StopBeforeDispatch()
                    if node['host']not in allowed_hosts(admit_long_job=True) or receipt.get('migration_required')is True:
                        node.setdefault('cooperative_returns',[]).append(dict(job=node.pop('job'),checkpoint=str(p)))
                        resume=str(p);continue
                    raise ValueError('Native completion outside registered integrated gate')
                return commit_completed_stage(self,run,target,node,p)
    return IntegratedOwner


def evaluation_class():
    from . import eval_owner
    # This new I-only interpreter; no existing F/G/H process or source is edited.
    eval_owner.checkpoint_at_gate=checkpoint_at_gate
    class IntegratedEvaluation(eval_owner.EvaluationOwner):
        def registry_now(self):
            return evaluation_registry(validate_registry(super().registry_now()))
        def samples(self):
            return [(host,s)for host,s in super().samples()if host not in PROTECTED]
    return IntegratedEvaluation


def review_loop():
    from .review import Review, BASELINES
    from .notify import refresh
    from tools.qwen_world_research.evaluate import read_sealed
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.rl.qwen_world_safety_exploration.integrated64_recipe import budget
    review=Review()
    while True:
        registry=validate_registry(read(ROOT_PATH/'RUNS.json'));run=registry['runs'][0];published=[];pending=[]
        targets=evaluation_gates(run)
        for u in targets:
            ev=ROOT_PATH/f"evaluations/{run['id']}_u{u:06}/NAVTEST";summary=ev/'SUMMARY.json'
            if not summary.exists():pending.append(u);continue
            point=review.out/f"{run['id']}_u{u:06}.json"
            value=read_sealed(summary)
            if value.get('status')!='COMPLETE'or value.get('scenes')!=12146 or value.get('invalid_scenes')!=0:
                raise ValueError('Full canonical NAVTEST only')
            if not point.exists():
                comparisons={name:review.compare(ev,p,label=name)for name,p in BASELINES.items()}
                peak=CONTROL/'evaluations/SAFETY_G8_CONTROL_u010000/NAVTEST'
                comparisons['FIXED_GLOBAL_BEST']=review.compare(ev,peak,label='FIXED_GLOBAL_BEST')
                payload=dict(schema='world_integrated64_full_population_review_v1',status='COMPLETE',
                    arm=run['id'],updates=u,PDMS=value['PDMS'],evaluation=str(ev),
                    actual_collection_budget=budget(u),comparisons=comparisons,
                    budget_warning='scene8x and candidate16x clocks differ; pending surplus and extraB1 cost are not equal-compute',
                    NAVTEST_never_enters_training_bank=True)
                atomic_json(point,payload);published.append(str(point))
            for name,ref in [('OPTIMIZER_UPDATES',u),('SCENE_CLOCK',8*u),('SOURCE_PG_CANDIDATE_CLOCK',16*u)]:
                p=CONTROL/f'evaluations/SAFETY_G8_CONTROL_u{ref:06}/NAVTEST'
                axis=('optimizer'if name=='OPTIMIZER_UPDATES'else'scene'if name=='SCENE_CLOCK'else'candidate')
                paired=review.out/f"{run['id']}_{u}_{axis}_vs_SAFETY_G8_CONTROL_{ref}.json"
                if not paired.exists()and(p/'SUMMARY.json').exists():
                    comparison=review.compare(ev,p,label=name)
                    atomic_json(paired,dict(schema='world_integrated64_external_control_comparison_v1',
                        status='COMPLETE',arm=run['id'],updates=u,reference_arm='SAFETY_G8_CONTROL',
                        reference_updates=ref,comparison_axis=axis,comparison=comparison,
                        actual_collection_budget=budget(u),
                        budget_contract='nominal '+axis+' clock only; pending surplus and additional B1 cost explicit'))
                    published.append(str(paired))
        atomic_json(review.out/'STATE.json',dict(status='WAITING_COMPLETE_NAVTEST'if pending else'COMPLETE_AT_REGISTERED_GATES',
                    ready=len(targets)-len(pending),pending=pending,published=published,heartbeat_unix=time.time()))
        if published:refresh(review.out/'STATE.json','Integrated64 full paired evidence published')
        if not pending or (review.out/'STOP_OWNER').exists():return
        time.sleep(120)


def run():
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    folder=ROOT_PATH/'owners/INTEGRATED_CAMPAIGN';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=native_class()();source=Path(__file__).resolve().parents[2];helpers={}
    for mode in ('evaluate','review'):
        own=(ROOT_PATH/'owners/RESEARCH_EVALUATION/OWNER.json'if mode=='evaluate'
             else ROOT_PATH/'analysis/PILOT_COMPARISONS/OWNER.json')
        if own.exists():
            previous=read(own)
            try:
                proc=Path('/proc')/str(previous['pid']);stat=(proc/'stat').read_text().split()
                env=(proc/'environ').read_bytes().split(b'\0')
                command=(proc/'cmdline').read_bytes().decode().split('\0')
                if (stat[2]not in('Z','X')and int(stat[21])==previous['start_ticks']
                        and ('WORLD_COMPARE_ROOT='+str(ROOT_PATH)).encode()in env
                        and 'tools.qwen_world_safety_exploration.integrated_campaign'in command
                        and mode in command):
                    helpers[mode]=dict(previous,adopted_existing=True);continue
            except FileNotFoundError:
                pass
        with (folder/f'{mode}.log').open('ab',buffering=0)as log:
            p=subprocess.Popen([MODEL_PYTHON,'-u','-m','tools.qwen_world_safety_exploration.integrated_campaign',mode],
                cwd=source,env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=''),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        helpers[mode]=dict(pid=p.pid,start_ticks=int(Path('/proc',str(p.pid),'stat').read_text().split()[21]))
    receipt=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
                 host=socket.gethostname(),source=str(source),helpers=helpers,started_unix=time.time())
    atomic_json(folder/'OWNER.json',receipt);atomic_json(folder/'LAUNCH.json',receipt)
    owner.run()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('check','run','evaluate','review'));a=p.parse_args()
    configure();registry=validate_registry(read(ROOT_PATH/'RUNS.json'))
    if a.mode=='check':
        print(json.dumps(dict(status='REGISTRY_VALID_NO_DISPATCH',arm=registry['runs'][0]['id'])));return
    if a.mode=='run':return run()
    if a.mode=='evaluate':return evaluation_class()(ROOT_PATH/'RUNS.json',max_shards=32,max_workers=64).run(interval=30)
    review_loop()


if __name__=='__main__':
    main()
