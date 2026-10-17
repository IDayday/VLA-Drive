"""Read only small immutable receipts; no checkpoint tensors, GPU or launch."""
import datetime,hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
sys.path.insert(0,str(REPO))
os.sched_setaffinity(0,{10,11})
started=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('review wall limit')))
signal.alarm(60)
PINS=[]; cached={}

def read(path, expected=None, jsonl=False):
    path=Path(path).resolve()
    assert path.suffix in {'.json','.jsonl','.py','.md'},path
    assert path.stat().st_size<2*1024**2,path
    if str(path) not in cached:
        b=path.read_bytes();cached[str(path)]=b
        PINS.append({'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    b=cached[str(path)];sha=hashlib.sha256(b).hexdigest()
    if expected:assert sha==expected,(path,sha,expected)
    if path.suffix in {'.py','.md'}:return b.decode()
    return [json.loads(x) for x in b.decode().splitlines()] if jsonl else json.loads(b)

def pin(p,**kw):return read(p['path'],p['sha256'],**kw)
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

try:
    summary=read(REPO/'reports/st_world_policy/20261017/T1_REAL_NATIVE2_TO4_RESULT_v1/RESULT.json')
    root=Path('/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1')
    runtime=Path('/mnt/project/VLA-Drive-st-temporal-readiness-inventory-20261017')
    sampler=read(runtime/'tools/ddpolicy_vehicle/training_state.py')
    assert sampler==subprocess.check_output(['git','show',summary['source_sha']+':tools/ddpolicy_vehicle/training_state.py'],cwd=REPO,text=True)
    assert sampler==read(REPO/'tools/ddpolicy_vehicle/training_state.py')
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    from tools.foresight.student_state import learning_rate
    from starVLA.model.modules.vehicle_joint.initialization import identity_hash
    manifest=pin(summary['data_manifest']);assert len(manifest['scenes'])==64 and manifest['split']=='train'
    all_steps=[]; phases={}
    for phase in summary['phases']:
        step=phase['phase']; adapter=pin(phase['adapter_result']);native=pin(phase['native_result'])
        receipt=pin(phase['native_receipt']);rows=pin(phase['steps'],jsonl=True)
        registration=pin(adapter['registration']);cfg=pin(registration['artifacts']['config'])
        run=root/'runs'/registration['run_id']; run_id=read(run/'identity.json')
        assert identity({k:v for k,v in run_id.items() if k!='identity'})==run_id['identity']
        assert identity({k:v for k,v in receipt.items() if k!='receipt_identity'})==receipt['receipt_identity']
        assert run_id['registration_sha256']==adapter['registration']['sha256']
        assert cfg==run_id['config'] and registration['source_sha']==summary['source_sha']==native['source_sha']
        assert adapter['status']=='COMPLETE' and adapter['new_optimizer_updates']==native['new_optimizer_updates']==2
        assert native['status']=='PAUSED' and native['pause_reason']=='BUDGET_PAUSED'
        assert native['completed']==receipt['state']['completed']==step
        assert native['identity']==run_id['identity']==receipt['state']['identity']
        assert receipt['state']['offset']==step and receipt['state']['exposure']==4*step and receipt['world_size']==2
        assert cfg['execution']['max_updates']==20000 and cfg['optimizer']['warmup']==1000
        assert cfg['execution']['scope']=='SMOKE' and not registration['weights_usable_for_formal_init']
        assert cfg['execution']['global_batch']==4 and cfg['execution']['micro_batch']==1 and cfg['execution']['world_size']==2
        assert [x['step'] for x in rows]==([1,2] if step==2 else [3,4])
        all_steps.extend(rows);phases[step]=(adapter,native,receipt,registration,cfg,run_id)
    before=phases[2];after=phases[4]
    continuation=after[4]['continuation']
    assert continuation['parent_receipt_sha256']==summary['phases'][0]['native_receipt']['sha256']
    assert continuation['parent_registration']==before[0]['registration']
    assert continuation['parent_tag']==before[2]['state']['tag']
    comparable=json.loads(json.dumps(after[4]));del comparable['continuation'];comparable['execution']['stop_after']=2
    assert comparable==before[4]
    resolution=read(root/'runs/T1_official_real64_phase4_v1/RESUME_RESOLUTION.json')
    assert resolution['request']==continuation and resolution['selected']['receipt']==before[2]
    assert resolution['parent_identity']==before[5]['identity']==after[1]['budget_extension_parent']
    assert resolution['parent_identity_sha256']==hashlib.sha256(cached[str((root/'runs/T1_official_real64_phase2_v1/identity.json').resolve())]).hexdigest()
    batches=epoch_batches(64,4,42,0);sample_checks=[]
    for row in all_steps:
        step=row['step'];expected=[[manifest['scenes'][idx]['token'] for idx in batches[step-1][rank::2]] for rank in range(2)]
        assert row['sample_ids_by_rank']==expected
        assert row['epoch']==0 and row['offset']==step and row['exposure']==step*4
        assert set(row['losses'])=={'ego_fm','current'}
        assert all(math.isfinite(v) for v in row['losses'].values())
        assert math.isfinite(row['gradient_norm']) and row['gradient_norm']>0
        assert row['fp32_master_update_verified'] is True and row['sampled_changed_elements']>0
        assert math.isfinite(row['maximum_sampled_change']) and row['maximum_sampled_change']>0
        assert row['counts']=={'current':4*3*96*384,'ego_scenes':4}
        assert row['lr']==learning_rate(step-1,1e-5,5e-7,1000,20000)
        assert row==summary['steps'][step-1]
        sample_checks.append({'step':step,'expected_ids_by_rank':expected,'actual_equal':True,
          'finite_logged_losses_and_gradient_norm':True,'sampled_changed_elements':row['sampled_changed_elements'],
          'maximum_sampled_change':row['maximum_sampled_change'],'lr':row['lr']})
    ranks=[]
    for pinrow in summary['live_restore_audits']:
        audit=pin(pinrow['receipt']);rank=pinrow['rank']
        assert audit['rank']==rank and audit['world_size']==2 and audit['expected_step']==2
        assert audit['checkpoint']==str(Path(summary['phases'][0]['native_receipt']['path']).parent)
        assert audit['receipt_sha256']==summary['phases'][0]['native_receipt']['sha256']
        assert audit['native_client_state']==before[2]['state'] and audit['receipt_identity']==before[2]['receipt_identity']
        assert audit['status']=='LIVE_MODEL_OPTIMIZER_AND_RNG_MATCH' and audit['optimizer_updates']==0
        assert audit['formal_launch_allowed'] is False
        leaves=0
        for name,section in audit['sections'].items():
            assert section['differences']==[] and section['nonfinite_paths']==[]
            for item in section['tensor_hashes'].values():assert item['live']==item['native']
            leaves+=len(section['tensor_hashes'])
            assert {k:v for k,v in section.items() if k!='tensor_hashes'}==pinrow['sections'][name]
        assert audit['nonpersistent_buffers']==pinrow['nonpersistent_buffers']
        ranks.append({'rank':rank,'matched_tensor_leaves':leaves,'matched_sections':list(audit['sections']),
          'actual_parent_receipt_sha256':audit['receipt_sha256'],'before_update':3,
          'nonpersistent_buffers_compared':False,'zero_update_exit_test':audit['zero_update_exit_audit']})
    observer=read('/mnt/project/st-world-policy-20261016/audits/correctness/audit_live_official_native_load.py',
        '5fce51c4c3e682ff4d3c33c958d7cacad49d05d372a7c46a9d39123a8dea5b5d')
    adapter_source=read('/mnt/project/VLA-Drive-st-t1-native4-execution-20261017/reports/st_world_policy/20261017/official_T1_native4_execution_v1/run_once.py',
        'fe7b418dadb683c143f61d51508df32dae0c9d92f2726a4aade1950f9d112f93')
    resource_failure=read(REPO/'reports/st_world_policy/20261017/t1_descendant_scope_review_v1/REVIEW.json')
    assert resource_failure['status']==summary['resource_qualification']=='FAIL_LIVE_DESCENDANT_SCOPE'
    assert summary['continuous_vs_interrupted_training_comparison']=='NOT_RUN'
    assert summary['last_native_checkpoint_reload']=='NOT_RUN_OUTSIDE_TWO_INVOCATION_BUDGET'
    result={'schema':'st_T1_actual_functional_review_v1','status':'PASS_RECORDED_FUNCTIONAL_SCOPE',
      'reviewer':'/root/st_calibration_independent_correctness_v2','source_sha':summary['source_sha'],
      'adapter_source_sha':summary['adapter_source_sha'],'review_base_sha':'fa75ea66ad342f90a477499f7d19fbc45d7054ae',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':sample_checks,'live_restore_ranks':ranks,
      'actual_total_updates':4,'actual_total_exposure':16,'resource_qualification':'FAIL_LIVE_DESCENDANT_SCOPE',
      'continuous_vs_interrupted_per_step_bitwise':'NOT_RUN','native_step4_reload':'NOT_RUN',
      'new_training_invocations':0,'GPU_used':False,'checkpoint_tensor_bytes_read':0,'model_image_teacher_payload_bytes_read':0,
      'formal_launch_allowed':False,'scientific_result':'NOT_EVALUATED','small_artifact_pins':PINS}
    (OUT/'REVIEW.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS: two actual phases, steps1-4, original sampler and LR, both restore-parent receipts and tensor hash comparisons agree.')
    print('Resource qualification remains FAIL_LIVE_DESCENDANT_SCOPE. No uninterrupted stepwise or final reload claim.')
finally:
    io=dict(line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())
    cost={'elapsed_seconds':time.monotonic()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      'logical_read_bytes':int(io['rchar']),'physical_read_bytes':int(io['read_bytes']),
      'CPU_affinity':sorted(os.sched_getaffinity(0)),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'unique_small_artifact_bytes':sum(len(v) for v in cached.values()),'unique_small_artifacts':len(cached)}
    (OUT/'COST.json').write_text(json.dumps(cost,indent=2)+'\n')
    print(json.dumps(cost));assert cost['elapsed_seconds']<60 and cost['max_rss_bytes']<4*1024**3 and cost['logical_read_bytes']<256*1024**2
