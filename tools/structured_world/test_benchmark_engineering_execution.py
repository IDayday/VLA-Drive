"""CPU-only reproducible driver checks. Fresh output; no GPU/SSH dispatch."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('benchmark', args.driver)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
root = args.output.resolve()
records = []

def check(name, fn):
    try: fn(); records.append({'name': name, 'passed': True})
    except Exception as exc: records.append({'name': name, 'passed': False, 'error': repr(exc)})

def rejects(fn):
    try: fn()
    except (ValueError, BlockingIOError): return
    raise AssertionError('Unsafe input unexpectedly accepted')

def store(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); b.write(path, data)

config = {'structured_world': {'formal_recipe_assets': {'label_cache_identity': 'label', 'current_DINO_identity': 'dino', 'shared_geometry_identity': 'geo'}},
          'framework': {'action_model': {'num_inference_timesteps': 10}}}
contract = dict(identity='original', scope='formal', dataset='navsim', config=config, training_source_sha='0'*40,
    training_source_tree_sha256={'train': 'sha'}, cache_identity='label', DINO_identity='dino', shared_geometry_identity='geo',
    current_and_GT_identity='current', training_seed=42, world_size=8, effective_global_batch=32, FM_repeat=8,
    microbatch=4, scenes_per_epoch=101592, source_population_scenes=101592, engineering_training_selection=None,
    updates_per_epoch=3175, schedule_horizon=100000, schedule_warmup=5000, determinism={'same': True}, precision='FP32', initialization_assets={'same': True})

def native(name, completed=25000, change=None):
    run=root/name; parent=copy.deepcopy(contract)
    if change: parent.update(change)
    store(run/'identity.json', parent); store(run/'optimizer_groups.json', {'lr': 'original'})
    tag='native_'+str(completed); folder=run/'checkpoints'/tag
    folder.mkdir(parents=True)
    names={'mp_rank_00_model_states.pt'} | {f'rng_rank{r}.pt' for r in range(8)} | {f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(8)}
    for name in names: (folder/name).write_bytes(b'CPU MOCK ONLY')
    complete=dict(identity='original', ranks=8, tag=tag, completed=completed, file_sha256={name:b.digest(folder/name) for name in names})
    store(folder/'COMPLETE.json', complete)
    case=dict(name=run.name, pilot_origin_run=str(run), pilot_origin_tag=tag, origin_identity='original', origin_complete_sha256=b.digest(folder/'COMPLETE.json'),
              train_args=sum(([flag, '4' if flag=='--micro-batch' else '/asset/'+flag[2:]] for flag in sorted(b.BASE_FLAGS)), []))
    return case

base_case=native('mid'); early=native('early',0,{'scope':'profile'})
manifest=dict(source_root=str(root/'source'), source_sha='1'*40, python=sys.executable, host='training-vlawm-zt4', gpu_uuids=['GPU-'+str(i) for i in range(8)],
              artifact_root=str(root/'locks'), minimum_free_MiB=60000, qualification={'path':'/qualification','sha256':'0'*64},
              budget=dict(GPU_hours_limit=120, already_spent=5, reserved_elsewhere=30, per_job_reserve_GPU_hours=3),
              cases=[dict(base_case,name='G1_mid'),dict(early,name='G1_early'),dict(base_case,name='G3_mid'),dict(early,name='G3_early')])

def plan():
    p=b.build_plan(manifest,root/'planned',driver=args.driver.resolve())
    assert len(p['jobs'])==24 and p['maximum_total_updates']==5280
    for i,j in enumerate(p['jobs']):
        assert j['mode']==b.MODES[i%2]
        assert '--benchmark-no-end-save' in j['train_args'] and '--resume' not in j['train_args']
        assert j['train_args'][j['train_args'].index('--debug-updates')+1]=='220'
        assert j['command'][j['command'].index('--')+1]==str(args.driver.resolve())
        assert j['original_command'][j['original_command'].index('--')+1].endswith('/train_vla.py')
    store(root/'MOCK_MANIFEST.json',manifest); store(root/'MOCK_PLAN.json',p)
check('24_jobs_5280_updates_alternating_native_no_save_separator',plan)

def candidates():
    for mode in b.CANDIDATES:
        m=copy.deepcopy(manifest);m['candidate_mode']=mode
        p=b.build_plan(m,root/('plan_'+mode),driver=args.driver.resolve())
        assert len(p['jobs'])==24 and p['mode_pair']==[b.MODES[0],mode]
        assert [j['mode'] for j in p['jobs']]==[b.MODES[0],mode]*12
        store(root/('MOCK_PLAN_'+mode+'.json'),p)
    m=copy.deepcopy(manifest);m['candidate_mode']='unsupported'
    rejects(lambda:b.build_plan(m,root/'bad_mode',driver=args.driver.resolve()))
check('single_candidate_fast_timevec_precision_24_windows_no_cartesian',candidates)
check('formal_mid_origin_accepted', lambda: b.origin(base_case))
check('profile_checkpoint0_fullpopulation_formal_schedule_accepted', lambda: b.origin(early))
check('short_profile_schedule_rejected', lambda: rejects(lambda: b.origin(native('short',0,{'scope':'profile','schedule_horizon':220}))))
check('debug_population_rejected', lambda: rejects(lambda: b.origin(native('subset',0,{'scope':'profile','scenes_per_epoch':64}))))
check('mismatched_config_asset_identity_rejected', lambda: rejects(lambda: b.origin(native('foreign',0,{'cache_identity':'foreign'}))))

def invalid_plan(name, mutate):
    m=copy.deepcopy(manifest); mutate(m); rejects(lambda:b.build_plan(m,root/name,driver=args.driver.resolve()))
check('over24_jobs_rejected', lambda: invalid_plan('too_many',lambda m:m['cases'].append(dict(base_case,name='fifth'))))
check('over120_GPUh_rejected', lambda: invalid_plan('budget_cap',lambda m:m['budget'].update(GPU_hours_limit=121)))
check('whole_matrix_budget_reservation_rejected', lambda: invalid_plan('matrix_cap',lambda m:m['budget'].update(reserved_elsewhere=80)))
check('duplicate_UUID_rejected', lambda: invalid_plan('uuid',lambda m:m['gpu_uuids'].__setitem__(7,'GPU-0')))
check('scientific_argument_override_rejected', lambda: invalid_plan('override',lambda m:m['cases'][0]['train_args'].extend(['--debug-scenes','32'])))

def timer_counts():
    for micros in (1,2,4):
        events=[]; clock=iter([10.,610.])
        timer=b.BoundaryTimer(lambda:events.append('sync'),lambda x:events.append(('MAX',x)) or x,lambda:next(clock))
        for _ in range(220):
            for i in range(micros): timer.after_step(i==micros-1)
        assert timer.count==220 and timer.local_seconds==timer.max_rank_seconds==600
        assert events==['sync','sync',('MAX',600.)]
check('timer_micro1_2_4_only_two_boundary_syncs_one_MAX',timer_counts)

def worker_counts():
    from torch.distributed.run import get_args_parser
    p=b.build_plan(manifest,root/'worker_plan',driver=args.driver.resolve())
    parsed=get_args_parser().parse_args(p['jobs'][0]['command'][3:])
    assert parsed.training_script==str(args.driver.resolve()) and '--run' in parsed.training_script_args
    for micros in (1,2,4):
        run=root/f'worker_{micros}';run.mkdir(); events=[]
        class Engine:
            boundary=False
            def is_gradient_accumulation_boundary(self): return self.boundary
            def step(self): events.append(('step',self.boundary)); return 'original_return'
        cuda=types.SimpleNamespace(synchronize=lambda device:events.append('sync'), is_initialized=lambda:True,
            max_memory_allocated=lambda dev:100,max_memory_reserved=lambda dev:200)
        tensor=lambda value,**kw:types.SimpleNamespace(item=lambda:value)
        torch=types.ModuleType('torch');torch.cuda=cuda;torch.float64='float64';torch.device=lambda *x:x;torch.tensor=tensor
        dist=types.ModuleType('torch.distributed');dist.ReduceOp=types.SimpleNamespace(MAX='MAX');dist.all_reduce=lambda *a,**kw:events.append('MAX')
        torch.distributed=dist
        ds=types.ModuleType('deepspeed.runtime.engine');ds.DeepSpeedEngine=Engine
        modules={'torch':torch,'torch.distributed':dist,'deepspeed':types.ModuleType('deepspeed'),
            'deepspeed.runtime':types.ModuleType('deepspeed.runtime'),'deepspeed.runtime.engine':ds}
        def fake_train(*a,**kw):
            engine=Engine()
            for _ in range(220):
                for i in range(micros):
                    engine.boundary=i==micros-1; assert engine.step()=='original_return'
        argv=['driver','--worker','--trainer',str(root/'source/tools/structured_world/train_vla.py'),'--run',str(run),'--','--scope','profile']
        with patch.dict(sys.modules,modules),patch.dict(os.environ,RANK='0',LOCAL_RANK='0',CUDA_VISIBLE_DEVICES=','.join(manifest['gpu_uuids'])),patch.object(sys,'argv',argv),patch.object(b.runpy,'run_path',fake_train): b.worker()
        record=b.read(run/'worker_rank0.json')
        assert record['passed'] and record['optimizer_boundaries']==220
        assert events.count('sync')==2 and events.count('MAX')==1 and len([x for x in events if isinstance(x,tuple)])==220*micros
check('real_torchrun_argparse_and_worker_step_wrapper_micro1_2_4',worker_counts)

def busy_gates():
    m=copy.deepcopy(manifest)
    rows='\n'.join(f'{uuid}, 0, 70000, 0' for uuid in m['gpu_uuids'])
    def output(cmd,*a,**kw):return rows if '--query-gpu=uuid,memory.used,memory.free,utilization.gpu' in cmd else ''
    with patch.object(b,'authorize',lambda host: {'mock':True}),patch.object(b.socket,'gethostname',lambda:m['host']+'-worker-0'),patch.object(b,'checked',output): assert len(b.readiness(m)['gpus'])==8
    def occupied(cmd,*a,**kw):return rows if '--query-gpu=uuid,memory.used,memory.free,utilization.gpu' in cmd else 'GPU-0, 999, REAL_TASK'
    with patch.object(b,'authorize',lambda host:{'mock':True}),patch.object(b.socket,'gethostname',lambda:m['host']),patch.object(b,'checked',occupied):rejects(lambda:b.readiness(m))
check('readiness_idle_accepted_real_compute_app_rejected_no_signal',busy_gates)

def locks():
    from contextlib import ExitStack
    with ExitStack() as stack:
        b.acquire(stack,root/'owned.lock')
        with ExitStack() as other: rejects(lambda:b.acquire(other,root/'owned.lock'))
    with ExitStack() as stack:b.acquire(stack,root/'owned.lock')
check('nonblocking_own_lock_exclusion_and_release',locks)

def fixture(name,mode):
    run=root/name;case=base_case;native_record=b.origin(case);start=native_record['complete']['completed']
    identity=copy.deepcopy(contract);identity.update(identity=name,benchmark_no_end_checkpoint=True,execution_mode=mode,
        engineering_continue_origin={'identity':'original','tag':case['pilot_origin_tag'],'complete_sha256':case['origin_complete_sha256']})
    store(run/'identity.json',identity);store(run/'TRAINING_COMPLETE.json',{'identity':name,'updates':start+220})
    store(run/'RESUME_BOUNDARY_VERIFIED.json',{'identity':name,'passed':True,'completed':start,'all_task_RNG':'exact all ranks'})
    store(run/'optimizer_groups.json',{'lr':'original'})
    with (run/'metrics.jsonl').open('x') as f:
        for i in range(1,221):f.write(json.dumps({'updates':start+i,'scene_exposure':i*32,'seconds':2.9})+'\n')
    with (run/'stdout_metric_arrivals.jsonl').open('x') as f:
        for i in (range(1,221) if mode==b.MODES[0] else range(20,221,20)):
            f.write(json.dumps({'updates':start+i,'monotonic_ns':int((100+i*3)*1e9),'since_process_start_seconds':i*3+50})+'\n')
    for rank in range(8):store(run/f'worker_rank{rank}.json',dict(rank=rank,passed=True,GPU_UUID=manifest['gpu_uuids'][rank],optimizer_boundaries=220,
        stable_200_local_wall_seconds=600-rank,stable_200_max_rank_wall_seconds=600,peak_allocated_bytes=100+rank,peak_reserved_bytes=200+rank))
    if mode!=b.MODES[0]:
        with (run/'performance_windows.jsonl').open('x') as f:
            for i in range(11):f.write(json.dumps({'start_update':start+20*i+1,'end_update':start+20*(i+1),'updates':20,
                'wall_seconds_max_rank_including_log_flush':61,'update_current_stream_seconds_max_rank':[2.8]*20,'scene_exposure':640})+'\n')
    job={'key':name,'case':'G1_mid','mode':mode,'run':str(run),'gpu_uuids':manifest['gpu_uuids']}
    return job,native_record,{'wall_seconds':800,'native_boundary_observed_seconds':40}

def analyze(mode):
    job,n,p=fixture('analysis_'+mode,mode); result=b.analyze_job(job,n,p)
    assert result['stable_200_window_max_rank_wall_seconds']==600
    assert result['stable_scene_exposure']==6400 and result['stable_updates']==200
    assert result['stable_200_window_rank0_stdout_wall_seconds']==600
    assert result['stable_wall_seconds']==600
    assert result['trainer_window_wall_seconds']==(None if mode==b.MODES[0] else 610)
    assert result['checkpoint_save_seconds']==0 and len(result['all_rank_peak_allocated_reserved_bytes'])==8
    store(root/('ANALYSIS_'+mode+'.json'),result)
check('IO_exact_rank_wall_plus_reference_stdout_200_population',lambda:analyze(b.MODES[0]))
check('fast_true_log_window_wall_and_rank_boundary_wall_200_population',lambda:analyze(b.MODES[1]))
check('timevec_same_primary_rank_wall_secondary_trainer_window',lambda:analyze('engineering_timevec'))
check('precision_same_primary_rank_wall_secondary_trainer_window',lambda:analyze('precision_fast'))

def dynamic_aggregate():
    for mode in b.CANDIDATES:
        reference=b.read(root/('ANALYSIS_'+b.MODES[0]+'.json'))
        candidate=b.read(root/('ANALYSIS_'+mode+'.json'))
        row=b.aggregate([reference,candidate]*3,mode)['G1_mid']
        assert row['mode_pair']==[b.MODES[0],mode] and row['candidate_to_IO_throughput_ratio']==1
        assert row[mode]['finished_repetitions']==3 and row[b.MODES[0]]['finished_repetitions']==3
check('dynamic_mode_pair_aggregation_identical_PRIMARY_wall_fairness',dynamic_aggregate)

def missing_rank():
    job,n,p=fixture('missing_rank',b.MODES[0]);(Path(job['run'])/'worker_rank7.json').unlink()
    try:b.analyze_job(job,n,p)
    except FileNotFoundError:return
    raise AssertionError('Missing rank accepted')
check('missing_rank_receipt_rejected',missing_rank)

def invalid_receipt():
    job,n,p=fixture('nan_receipt',b.MODES[0]);record=Path(job['run'])/'worker_rank7.json'
    data=b.read(record);data['stable_200_local_wall_seconds']=float('nan');record.write_text(json.dumps(data))
    rejects(lambda:b.analyze_job(job,n,p))
check('nonfinite_rank_boundary_receipt_rejected',invalid_receipt)

def cpu_process():
    trainer=root/'source/tools/structured_world/train_vla.py';trainer.parent.mkdir(parents=True,exist_ok=True);trainer.write_text('# CPU MOCK ONLY\n')
    code="import json,pathlib,time;pathlib.Path('"+str(root/'owned_cpu')+"/RESUME_BOUNDARY_VERIFIED.json').write_text('{}');time.sleep(.1);print(json.dumps({'updates':1,'seconds':3}),flush=True)"
    job={'run':str(root/'owned_cpu'),'trainer':str(trainer),'gpu_uuids':manifest['gpu_uuids'],'command':[sys.executable,'-B','-c',code]}
    with b.ExitStack() as stack:
        lock=b.acquire(stack,root/'cpu_only.lock'); result=b.run_job(job,[lock])
    assert result['returncode']==0 and result['native_boundary_observed_seconds'] is not None
    assert len(b.json_lines(Path(job['run'])/'stdout_metric_arrivals.jsonl'))==1
    job.update(run=str(root/'owned_cpu_failed'),command=[sys.executable,'-B','-c',"print('owned CPU failure');raise SystemExit(7)"])
    try:b.run_job(job,[])
    except ValueError:pass
    else:raise AssertionError('Failure silently accepted')
    assert b.read(Path(job['run'])/'PROCESS_EXIT.json')['returncode']==7
    assert (Path(job['run'])/'stdout.log').read_text().strip()=='owned CPU failure'
check('owned_CPU_stdout_timestamp_startup_failure_preserved_no_retry',cpu_process)

def no_dispatch():
    source=root/'source';trainer=source/'tools/structured_world/train_vla.py'
    trainer.write_text("# '--benchmark-no-end-save' 'engineering_fast' CPU MOCK ONLY\n")
    m=copy.deepcopy(manifest);m['source_root']=str(source);input=root/'INPUT.json';store(input,m)
    with (patch.object(b,'immutable_source',lambda *a:None),patch.object(b,'readiness',side_effect=AssertionError('No GPU reads in planning')),
          patch.object(b,'runtime_snapshot',side_effect=AssertionError('No runtime dispatch in planning')),
          patch.object(sys,'argv',['driver','--manifest',str(input),'--output',str(root/'plan_only')])):assert b.main()==0
    assert (root/'plan_only/MANIFEST.json').is_file() and not (root/'plan_only/REGISTRATION.json').exists()
    with patch.object(sys,'argv',['driver','--manifest',str(input),'--output',str(root/'plan_only')]):rejects(b.main)
check('plan_only_no_workers_no_GPU_queries_fresh_output_reject',no_dispatch)

result=dict(passed=all(r['passed'] for r in records), cases=len(records), tests=records, driver=str(args.driver.resolve()),
            driver_sha256=b.digest(args.driver), validation_script_sha256=b.digest(__file__), GPU_jobs_launched=0,
            qualification='CPU mock only; no hardware/throughput qualification claimed', fixture_root=str(root))
store(root/'VALIDATION.json',result);print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
