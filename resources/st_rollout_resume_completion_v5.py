"""Lead-only finish the remaining B2+2; reuse completed v4 A4 read-only.

Original runtime751 and v4 observer math are imported unchanged. Fresh metadata
admission/leases are mandatory; neither old leases nor the old failed result are
rewritten. This coordinator never launches A or permits more than two B starts.
"""
from __future__ import annotations
import argparse,fcntl,importlib.util,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path

OLD_FILE=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v4-20261018/resources/st_rollout_resume_smoke_v1.py')
OLD_SHA='468fe8f1ed20ef3abb2b2407413f92fb078f6b1d05f6cbfb2ed4e4b6e69f9692'
CAMPAIGN=Path('/mnt/project/st-world-policy-20261016')
OLD_ROOT=CAMPAIGN/'smoke/rollout_scale_native_resume_v4'
ROOT=CAMPAIGN/'smoke/rollout_scale_native_resume_completion_v5'
IMPLEMENTER='/root/road_failure_direction_runner'
_LEGACY=None


def legacy():
    global _LEGACY
    if _LEGACY is None:
        import hashlib
        if hashlib.sha256(OLD_FILE.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('Original v4 observer changed')
        spec=importlib.util.spec_from_file_location('unchanged_native_resume_v4',OLD_FILE)
        _LEGACY=importlib.util.module_from_spec(spec);spec.loader.exec_module(_LEGACY)
        _LEGACY.ROOT=ROOT # Explicit output/control adapter, never a math hook.
    return _LEGACY


def references():
    """Only small controls/native receipt+stat, never GB model reads or writes."""
    v=legacy();history=[]
    for root in (v.OLDER_ROOT,v.PRIOR_ROOT,OLD_ROOT):
        v.prior_exited(root/'control')
        for admission in sorted(root.glob('control/invocations/*/*/ADMISSION.json')):
            history.append({name:v.pin(admission.with_name(name+'.json')) for name in ('ADMISSION','LAUNCH','EXIT')})
    if len(history)!=4:raise ValueError('Exactly four actual prior launches must remain charged')
    result_path=OLD_ROOT/'continuous/runs/C1_rollout_std_native_continuous_v1/RESULT.json'
    result=v.read(result_path);run=result_path.parent
    if (result['completed']!=4 or result['new_optimizer_updates']!=4 or result['status']!='PAUSED'
            or not result['frozen_encoder_unchanged'] or result['counts']['actual_rollouts']!=512):raise ValueError('No immutable actual A4 reference')
    natives=[]
    for step in (0,2,4):
        folder,record,_=v.selected_native(run,step,require_latest=False)
        if record['state']['identity']!=result['identity']:raise ValueError('A native identity differs')
        natives.append(v.pin(folder/'COMPLETE.json'))
    observer=[]
    for rank in range(4):
        path=OLD_ROOT/'continuous/observer'/f'INITIAL_OBSERVER_COMPLETE_rank{rank}.json';row=v.read(path)
        if row['observer_sha256']!=OLD_SHA or row['completed']!=4 or row['runtime_sha']!=v.SOURCE:raise ValueError('A observer completion missing')
        observer.append(v.pin(path))
    b=OLD_ROOT/'interrupted/runs/C1_rollout_std_native_interrupted_v1'
    if sorted(p.name for p in b.iterdir())!=['RUN.lock','identity.json','resolved_config.json']:raise ValueError('Old B is not the preserved zero-progress metadata-only start')
    calls=sum(v.read(p)['calls'] for p in OLD_ROOT.glob('*/runs/*/official_feedback/*/attempt_*/ADMITTED.json'))
    if calls!=512:raise ValueError('Prior official-call budget is not exactly512')
    start=v.read(OLD_ROOT/'control/START.json');failed=OLD_ROOT/'control/initial_RESULT.json'
    elapsed=math.ceil(failed.stat().st_mtime-start['started_unix'])
    if not 0<elapsed<=360:raise ValueError('Actual prior controller exceeds the conservative360s charge')
    return {'original_observer':v.pin(OLD_FILE),'old_binding':v.pin(OLD_ROOT/'BINDING.json'),
        'A_registration':v.pin(OLD_ROOT/'continuous/REGISTRATION.json'),'A_result':v.pin(result_path),
        'A_native_receipts':natives,'A_observer_completions':observer,
        'old_B_registration':v.pin(OLD_ROOT/'interrupted/REGISTRATION.json'),
        'old_B_metadata':[v.pin(p) for p in sorted(b.iterdir())],
        'history':history,'last_result':v.pin(failed),'prior_stats':v.read(failed)['stats'],
        'old_expired_START':v.pin(OLD_ROOT/'control/START.json'),'old_controller_seconds_observed_ceil':elapsed,'old_controller_seconds_charged':360,
        'prior_reserved_seconds':660,'remaining_training_seconds':2940,
        'completed_updates':4,'remaining_updates':4,'consumed_calls':512,'remaining_calls':512,
        'prior_invocations':4,'remaining_invocations':2,'campaign_invocation_cap':6,
        'calendar_gap_scope':'Waiting/calendar gap is disclosed in fresh admission; not retroactively extending the expired old lease'}


def draft(output,layout_ref,valid_seconds):
    v=legacy();ep=v.helper();refs=references();layout=v.pinned(layout_ref);host=socket.gethostname()
    if not 3900<=valid_seconds<=7200 or layout['hostname']!=host or layout['host']!='training-vlawm-tsinghua-zt6':raise ValueError('Actual authorized fresh zt6 lease required')
    group=layout['groups']['B_pause2_resume2']
    if group['GPU_indices']!=[4,5,6,7] or len(set(group['GPU_UUIDs']))!=4:raise ValueError('Only original B four GPUs')
    output=Path(output)
    if not output.is_absolute() or not output.is_relative_to(CAMPAIGN) or output.exists() or output==ROOT or ROOT in output.parents:raise ValueError('Fresh separate admission output required')
    if str(v.RUNTIME) not in sys.path:sys.path.insert(0,str(v.RUNTIME))
    from tools.st_world_policy.training_config import environment_record
    environment=environment_record();old_reg=v.pinned(refs['old_B_registration'])
    if environment!=v.pinned(old_reg['artifacts']['environment']):raise ValueError('Numeric/software environment changed')
    output.mkdir(parents=True,exist_ok=False);v.write(output/'ENVIRONMENT.json',environment);now=time.time()
    lease={'schema':'st_training_resource_lease_v1','hostnames':[host],'world_size':4,
        'cuda_visible_devices_by_host':{host:','.join(group['GPU_UUIDs'])},'gpu_uuids_by_host':{host:group['GPU_UUIDs']},
        'valid_from_unix':now-1,'valid_until_unix':now+valid_seconds,'max_invocation_seconds':1800,
        'authorization':str(ep.AUTHORIZATION),'purpose':'Only remaining B2+2, A4 read-only reference; never formal initialization'}
    v.write(output/'B_RESOURCE_LEASE.json',lease)
    binding={'schema':'st_native_resume_completion_binding_v5','coordinator_sha256':v.sha(__file__),'root':str(ROOT),
        'observer_sha256':OLD_SHA,'source_sha':v.SOURCE,'references':refs,'resource_layout':layout_ref,
        'dispatch_policy':v.pin(ep.DISPATCH_POLICY),'authorization':v.pin(ep.AUTHORIZATION),
        'environment':v.pin(output/'ENVIRONMENT.json'),'lease':v.pin(output/'B_RESOURCE_LEASE.json'),
        'route':{'host':layout['host'],'hostname':host,'gpus':[4,5,6,7],'gpu_uuids':group['GPU_UUIDs'],
            'valid_from_unix':now-1,'valid_until_unix':now+valid_seconds},
        'calendar_gap_seconds_at_admission':now-Path(refs['last_result']['path']).stat().st_mtime}
    path=output/'BINDING_DRAFT.json';v.write(path,{'binding':binding,'binding_sha256':v.digest(binding),
        'status':'WAITING_TWO_INDEPENDENT_REVIEWS','metadata_only':True})
    return {'status':'DRAFT_ONLY_NOT_DISPATCHED','artifact':v.pin(path),'binding_sha256':v.digest(binding)}


def admission(path,checksum):
    v=legacy()
    if v.sha(path)!=checksum:raise ValueError('Admission checksum changed')
    receipt=v.read(path);binding=receipt['binding']
    if (receipt['status']!='APPROVED' or receipt['binding_sha256']!=v.digest(binding)
            or binding['coordinator_sha256']!=v.sha(__file__) or binding['observer_sha256']!=OLD_SHA
            or binding['source_sha']!=v.SOURCE or binding['root']!=str(ROOT) or binding['references']!=references()):raise ValueError('Stale completion binding')
    names=[]
    for kind in ('correctness','resources'):
        row=v.pinned(receipt['reviews'][kind]);name=row['reviewer']
        if row['status']!='PASS' or row['review_kind']!=kind or row['binding_sha256']!=v.digest(binding) or name in (IMPLEMENTER,receipt['issuer']):raise ValueError('Independent completion review missing')
        if not row['evidence_artifacts']:raise ValueError('Review evidence missing')
        for ref in row['evidence_artifacts']:
            if v.sha(ref['path'])!=ref['sha256']:raise ValueError('Review evidence changed')
        names.append(name)
    if len(set(names))!=2:raise ValueError('Two distinct completion reviewers required')
    return receipt,binding



def worker_binding(binding):
    v=legacy()
    return {'controller_sha256':OLD_SHA,'coordinator_sha256':v.sha(__file__),
        'completion_binding_sha256':v.digest(binding),'original_reference':binding['references']}


def prepare(binding):
    v=legacy()
    if ROOT.exists():raise ValueError('Completion namespace exists; never overwrite')
    if str(v.RUNTIME) not in sys.path:sys.path.insert(0,str(v.RUNTIME))
    from tools.st_world_policy.flow_training_config import validate_config,target_contract
    from tools.st_world_policy.check_formal_approval import verify_registration
    ROOT.mkdir();(ROOT/'continuous').symlink_to(OLD_ROOT/'continuous',target_is_directory=True)
    v.write(ROOT/'COORDINATOR_BINDING.json',binding)
    v.write(ROOT/'BINDING.json',worker_binding(binding))
    folder=ROOT/'interrupted';folder.mkdir();reg=v.pinned(binding['references']['old_B_registration']);cfg=v.pinned(reg['artifacts']['config'])
    cfg['runtime']['output_root']=str(folder/'runs');validate_config(cfg)
    v.write(folder/'CONFIG.json',cfg);v.write(folder/'TARGET_CONTRACT.json',target_contract(cfg))
    lease=v.pinned(binding['lease']);host=binding['route']['hostname'];uuids=binding['route']['gpu_uuids']
    if (lease['world_size']!=4 or lease['hostnames']!=[host] or lease['gpu_uuids_by_host']!={host:uuids}
            or lease['cuda_visible_devices_by_host']!={host:','.join(uuids)}):raise ValueError('Fresh B lease topology mismatch')
    reg['run_id']='C1_rollout_std_native_interrupted_v5';reg['artifacts'].update(config=v.pin(folder/'CONFIG.json'),
        target_contract=v.pin(folder/'TARGET_CONTRACT.json'),environment=binding['environment'],resource_lease=binding['lease'])
    v.write(folder/'REGISTRATION.json',reg);verify_registration(folder/'REGISTRATION.json',repo=v.RUNTIME)
    return {'status':'PREPARED_B_ONLY','A_reference_read_only':str(OLD_ROOT/'continuous'),'root':str(ROOT)}


def route_checks(binding):
    v=legacy();ep=v.helper()
    if str(v.RUNTIME) not in sys.path:sys.path.insert(0,str(v.RUNTIME))
    helpers=ep.mature_helpers();host=socket.gethostname();route=binding['route']
    if binding['dispatch_policy']['path']!=str(ep.DISPATCH_POLICY) or binding['authorization']['path']!=str(ep.AUTHORIZATION):raise ValueError('Actual live policy/auth required')
    policy=v.pinned(binding['dispatch_policy']);auth=v.pinned(binding['authorization'])
    if host!=route['hostname'] or host not in policy['allowed_aliases'] or route['host'] not in auth['allowed_hosts']:raise ValueError('Host unauthorized')
    for name,row in policy['global_dispatch_prohibitions'].items():
        if host==name or route['host']==name or host in row.get('aliases',[]):raise ValueError('Withdrawn host')
    helpers['check_route_time_v3'](route,time.time(),binding['references']['remaining_training_seconds'])
    helpers['fresh_environment_v3'](binding['environment']);helpers['shared_lock_mount_v3']()
    capacity=helpers['cpu_capacity_v3']()
    if capacity['quota_us']>=0 and capacity['quota_us']/capacity['period_us']<16:raise ValueError('Insufficient CPU quota')
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0])*1024<v.LIMITS['group_RSS_bytes']:raise ValueError('Fresh group RAM insufficient')
    import shutil
    if shutil.disk_usage(CAMPAIGN).free<v.LIMITS['runtime_shared_free_bytes']:raise ValueError('Shared reserve insufficient')
    if route['gpus']!=[4,5,6,7] or len(set(route['gpu_uuids']))!=4 or not set(range(16,32)).issubset(os.sched_getaffinity(0)):raise ValueError('Only four B GPUs/CPU16–31')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=v.RUNTIME,text=True,timeout=15).strip()!=v.SOURCE or subprocess.check_output(['git','status','--porcelain'],cwd=v.RUNTIME,timeout=15).strip():raise ValueError('Runtime751 changed')
    return route,helpers


def drain_finished_job(job):
    """A normal root exit may leave proven CPU compiler children briefly alive."""
    v=legacy()
    if job['child'].poll()!=0:raise ValueError('Launcher did not exit successfully')
    v.stop_smoke_owned([job])
    if not job.get('cleanup_complete'):raise ValueError('Owned descendants not clean after successful launcher drain')



def finish_successful_job(job,phase):
    v=legacy();step=4 if phase=='resume' else 2
    if job['child'].poll()!=0:raise ValueError('Launcher did not exit successfully')
    outcome=v.validate_result('interrupted',step,2)
    for rank in range(4):
        path=ROOT/'interrupted/observer'/f"{'RESUMED' if phase=='resume' else 'INITIAL'}_OBSERVER_COMPLETE_rank{rank}.json"
        row=v.read(path)
        if (row['observer_sha256']!=OLD_SHA or row['runtime_sha']!=v.SOURCE or row['rank']!=rank
                or row['completed']!=step or row['registration']!=v.pin(ROOT/'interrupted/REGISTRATION.json')):
            raise ValueError('Four-rank original observer completion evidence missing')
    drain_finished_job(job)
    return outcome


def execute(phase,receipt,binding):
    v=legacy();ep=v.helper();route,helpers=route_checks(binding);control=ROOT/'control_v5';control.mkdir(exist_ok=True)
    if v.read(ROOT/'COORDINATOR_BINDING.json')!=binding:raise ValueError('Prepared identity changed')
    start=control/'START.json'
    if phase=='initial':v.write(start,{'started_unix':time.time(),'deadline_unix':time.time()+binding['references']['remaining_training_seconds']})
    deadline=v.read(start)['deadline_unix'];jobs=[];locks=[];stop=[];reason=None;outcome=None
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def requested(signum,frame):
        if not stop:stop.append(signum)
    def check():
        if stop:raise RuntimeError('OPERATOR_SIGNAL_'+str(stop[0]))
        if time.time()>=deadline:raise RuntimeError('COMPLETION_ABSOLUTE_DEADLINE')
    prior=binding['references']['prior_stats'] if phase=='initial' else v.read(control/'initial_RESULT.json')['stats']
    seen={};retired={'rchar':0,'read_bytes':0};stats={'sampled_rchar':prior['sampled_rchar'],'sampled_read_bytes':prior['sampled_read_bytes'],'peak_RSS_bytes':0};last_io=last_gpu=0
    try:
        for path in [control/'RUN.lock']+[Path('/tmp/foresight_gpu_'+str(i)+'.lock') for i in route['gpus']]:
            lock=path.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        for sig in old:signal.signal(sig,requested)
        check();v.prior_exited(control)
        if phase=='resume':
            if v.read(control/'initial_RESULT.json')['status']!='B2_COMPLETE_CLEAN':raise ValueError('B2 not complete')
            paused=v.validate_result('interrupted',2,2);v.archive_owned_pause(v.run_path('interrupted'),Path(paused['native']))
        elif v.run_path('interrupted').exists():raise ValueError('Fresh B output already exists')
        claimed=[p for p in control.glob('invocations/*/*') if p.is_dir()]
        if len(claimed)>=2:raise ValueError('Total campaign6 starts exhausted (prior4+new2)')
        unit='B_resume2' if phase=='resume' else 'B_initial2'
        argv=['timeout','--signal=TERM','--kill-after=60s',str(max(1,int(deadline-time.time())))+'s','taskset','-c','16-31',v.PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',str(Path(__file__).resolve()),'worker','--registration',str(ROOT/'interrupted/REGISTRATION.json'),'--audit-output',str(ROOT/'interrupted/observer')]
        if phase=='resume':argv+=['--resume']
        attempt=ep.claim_invocation(control,unit,resume=False,binding_sha=v.digest(binding),argv=argv)
        for gpu,uuid in zip(route['gpus'],route['gpu_uuids']):helpers['fresh_gpu_v3']({'gpu':gpu,'gpu_uuid':uuid})
        check();log=None
        try:
            log=(attempt/'OUTPUT.log').open('xb');child=subprocess.Popen(argv,cwd=v.RUNTIME,env={**os.environ,**ep.THREAD_ENV,'CUDA_VISIBLE_DEVICES':','.join(route['gpu_uuids'])},stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=tuple(p.fileno() for p in locks))
            job={'child':child,'attempt':attempt,'owned':{},'start_ticks':None};jobs.append(job)
        except OSError as error:
            v.write(attempt/'EXIT.json',{'status':'KNOWN_NOT_STARTED','owned_cleanup_complete':True,'error':repr(error)});raise
        finally:
            if log is not None:log.close()
        v.owned_processes(job);v.write(attempt/'LAUNCH.json',{'pid':child.pid,'argv':argv,'coordinator_sha256':v.sha(__file__),'observer_sha256':OLD_SHA,'binding_sha256':v.digest(binding)})
        while True:
            check();tree=v.owned_processes(job)
            for pid,row in tree.items():
                previous=seen.get(pid,{})
                if previous.get('start_ticks')==row['start_ticks']:
                    for key in retired:
                        if key in previous:row[key]=previous[key]
                elif previous:
                    for key in retired:retired[key]+=previous.get(key,0)
                seen[pid]=row
            rss=sum(row['rss'] for row in tree.values());stats['peak_RSS_bytes']=max(stats['peak_RSS_bytes'],rss)
            if rss>v.LIMITS['group_RSS_bytes']:raise ValueError('Group RSS cap')
            if time.monotonic()-last_gpu>=5:
                ep.LIMITS['GPU_driver_bytes']=v.LIMITS['GPU_driver_bytes'];v.monitor_smoke_gpu(ep,route,[job],tree);last_gpu=time.monotonic()
            if time.monotonic()-last_io>=10:
                import shutil
                roots=(v.OLDER_ROOT,v.PRIOR_ROOT,OLD_ROOT,ROOT)
                written=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
                dirs=[p for root in roots for p in root.glob('*/runs/*/checkpoints/step_*') if p.is_dir() and not (root==ROOT and p.is_relative_to(ROOT/'continuous'))]
                if written>v.LIMITS['output_bytes'] or len(dirs)>8:raise ValueError('Combined preserved output/native cap')
                if any(sum(p.stat().st_size for p in d.iterdir() if p.is_file())>v.LIMITS['native_directory_bytes'] for d in dirs):raise ValueError('Native single-dir cap')
                if shutil.disk_usage(CAMPAIGN).free<v.LIMITS['runtime_shared_free_bytes']:raise ValueError('Shared reserve')
                calls=512+sum(v.read(p)['calls'] for p in (ROOT/'interrupted').glob('runs/*/official_feedback/*/attempt_*/ADMITTED.json'))
                if calls>1024:raise ValueError('Campaign1024 official calls cap')
                for pid,row in seen.items():
                    try:
                        if v.process_record(pid)['start_ticks']!=row['start_ticks']:continue
                        io=dict(line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
                        for key in retired:row[key]=int(io[key])
                    except (FileNotFoundError,ProcessLookupError):pass
                for key in retired:stats['sampled_'+key]=prior['sampled_'+key]+retired[key]+sum(row.get(key,0) for row in seen.values())
                if stats['sampled_rchar']>v.LIMITS['monitor_read_bytes']:raise ValueError('Cumulative sampled rchar cap')
                last_io=time.monotonic()
            if child.poll() is not None:
                outcome=finish_successful_job(job,phase);break
            time.sleep(.1)
    except BaseException as error:reason=repr(error)
    finally:
        try:v.stop_smoke_owned(jobs)
        except BaseException as error:reason='CLEANUP_UNCONFIRMED:'+repr(error)
        if any(not job.get('cleanup_complete') for job in jobs):reason=reason or 'OWNED_CLEANUP_UNCONFIRMED'
        for job in jobs:v.write(job['attempt']/'EXIT.json',{'returncode':job['child'].returncode,'owned_cleanup_complete':job.get('cleanup_complete',False),'owned_IDs':{str(pid):row['start_ticks'] for pid,row in job['owned'].items()},'stop_reason':reason})
        for sig,value in old.items():signal.signal(sig,value)
        for lock in reversed(locks):lock.close()
    result={'status':('B4_COMPLETE_CLEAN' if phase=='resume' else 'B2_COMPLETE_CLEAN') if reason is None else 'PARTIAL_OR_FAILED','stop_reason':reason,'outcome':outcome,'stats':stats,'new_invocations_started':len(jobs),'A_reference_retrained':False,'weights_usable_for_formal_init':False}
    v.write(control/(phase+'_RESULT.json'),result);return result


def worker(args):
    v=legacy();reg=Path(args.registration).resolve();expected=(ROOT/'interrupted/REGISTRATION.json').resolve()
    if reg!=expected or Path(args.audit_output).resolve()!=(ROOT/'interrupted/observer').resolve():raise ValueError('Completion worker permits only new B')
    binding=v.read(ROOT/'COORDINATOR_BINDING.json')
    if binding['coordinator_sha256']!=v.sha(__file__) or v.read(ROOT/'BINDING.json')!=worker_binding(binding):raise ValueError('Completion worker source/binding changed')
    v.write(ROOT/'interrupted/observer'/f'COMPLETION_WRAPPER_rank{os.environ["RANK"]}.json',{'coordinator_sha256':v.sha(__file__),'original_observer_sha256':OLD_SHA,'source_sha':v.SOURCE,'B_only':True})
    return v.worker(args)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('status');subs.add_parser('compare')
    sub=subs.add_parser('draft-binding');sub.add_argument('--output',type=Path,required=True);sub.add_argument('--resource-layout',type=Path,required=True);sub.add_argument('--resource-layout-sha256',required=True);sub.add_argument('--valid-for-seconds',type=int,default=7200)
    for name in ('prepare','initial','resume'):
        sub=subs.add_parser(name);sub.add_argument('--admission',type=Path,required=True);sub.add_argument('--admission-sha256',required=True)
    sub=subs.add_parser('worker');sub.add_argument('--registration',required=True);sub.add_argument('--audit-output',required=True);sub.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='status':result={'status':'WAITING_FRESH_V5_BINDING','A_retrained':False,'remaining_updates':4,'remaining_calls':512,'remaining_invocations':2}
        elif args.command=='draft-binding':result=draft(args.output,{'path':str(args.resource_layout),'sha256':args.resource_layout_sha256},args.valid_for_seconds)
        elif args.command=='worker':worker(args);return 0
        elif args.command=='compare':result=legacy().compare()
        else:
            receipt,binding=admission(args.admission,args.admission_sha256)
            if args.command=='prepare':result=prepare(binding)
            else:result=execute(args.command,receipt,binding)
        print(json.dumps(result));return 2 if result.get('status')=='PARTIAL_OR_FAILED' or result.get('verdict')=='REPLAY_DIVERGENCE' else 0
    except (ValueError,KeyError,OSError,RuntimeError,subprocess.SubprocessError) as error:print(json.dumps({'status':'BLOCKED','error':repr(error)}));return 2

if __name__=='__main__':raise SystemExit(main())
