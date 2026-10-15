import datetime, fcntl, hashlib, json, math, os, pathlib, statistics, time

root=pathlib.Path(__file__).resolve().parent
directory=root/'G2_EVENT_UNIFORM'
def read(p): return json.loads(pathlib.Path(p).read_text())
def write(p,value):
    p=pathlib.Path(p);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)
m=read(directory/'REGISTRATION.json');run=pathlib.Path(m['run'])
parent=read(root/'PARENT_CLIENT_STATE.json')
launch=read(directory/'TRAINING_LAUNCH.json')
deadline=time.monotonic()+50
while True:
    boundary=read(run/'RESUME_BOUNDARY_VERIFIED.json')
    with (run/'metrics.jsonl').open('rb') as f:
        f.seek(launch['metrics_prefix_bytes']); data=f.read()
    lines=data.splitlines()
    if data and not data.endswith(b'\n'):lines=lines[:-1]
    metrics=[json.loads(x) for x in lines]
    if boundary.get('completed')==parent['completed'] and boundary.get('passed') and len(metrics)>=8: break
    state=read(directory/'STATE.json')
    assert state['status']!='FAILED',state
    if time.monotonic()>=deadline:
        print(json.dumps({'pending':True,'boundary_completed':boundary.get('completed'),'new_updates':len(metrics),'last':metrics[-1]['updates'] if metrics else None}));raise SystemExit(75)
    time.sleep(5)
assert all(boundary[k]==parent[k] for k in ('identity','completed','epoch','offset','exposure'))
assert read(root/'SOURCE_NATIVE_BOUNDARY_VERIFIED.json')['passed']
old=pathlib.Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/G2_vla_zt2_migration_20261014T0700Z/G2_EVENT_UNIFORM')
assert read(old/'STATE.json')=={'status':'PAUSED','completed':89574,'tag':'latest_0089574'}
assert read(run/'identity.json')==read(root/'SOURCE_PREVIOUS_identity.json')
assert read(run/'optimizer_groups.json')==read(root/'SOURCE_PREVIOUS_optimizer_groups.json')
assert [x['updates'] for x in metrics]==list(range(parent['completed']+1,parent['completed']+len(metrics)+1))
for i,row in enumerate(metrics,1):
    assert row['epoch']==parent['epoch'] and row['offset']==parent['offset']+i
    assert row['scene_exposure']==parent['exposure']+32*i
    assert row['fp32_master_update_verified'] and row['sampled_changed_elements']>0
    assert math.isfinite(row['global_gradient_norm'])
    assert all(math.isfinite(x) for x in row['weighted_losses'].values())
    assert all(math.isfinite(x) and x>0 for x in row['actual_parameter_group_lrs'])
with (run/'HOST_MIGRATION_RUNTIME.lock').open('a') as lock:
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: held=True
    else: held=False;fcntl.flock(lock,fcntl.LOCK_UN)
assert held,'Native controller must still own runtime lock'
pin=read(root/'ROLLBACK_CHECKPOINT_PIN.json')
for row in pin['files']:
    a=(pathlib.Path(pin['pin'])/row['name']).stat()
    assert (a.st_dev,a.st_ino,a.st_size)==(row['device'],row['inode'],row['bytes'])
times=[x['seconds'] for x in metrics]
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'host':m['destination_host'],
        'parent_completed':parent['completed'],'confirmed_update':metrics[-1]['updates'],'continuous_optimizer_updates':len(metrics),
        'training_source_sha':m['training_source_sha'],'execution_mode':m['execution_mode'],'identity':m['identity'],
        'exact_native_recovery':boundary,'source_controller_stays_PAUSED':True,'runtime_lock_owned':True,
        'no_replayed_or_lost_updates':True,'optimizer_inventory_unchanged':True,'all_new_losses_and_gradients_finite':True,
        'FP32_master_updates_observed':True,'data_epoch_offset_and_exposure_continuous':True,
        'initial_resume_steps_seconds':times,'all_observed_steps_median_seconds':statistics.median(times),
        'old_host_physical_gpu_occupancy':'UNKNOWN; source host was not contacted',
        'source_exit_evidence':'native PAUSED89574 COMPLETE + old controller child.wait outcome + exclusive shared runtime lock acquired before destination launch',
        'no_new_scientific_or_execution_configuration':True,'rollback_checkpoint':pin['pin']}
write(root/'EXACT_RESUME_BOUNDARY_VERIFIED.json',boundary)
write(root/'CONTINUATION_ACCEPTED.json',record)
write(root/'CONTINUATION_METRICS.json',metrics)
route=read(run/'CURRENT_RESOURCE_ROUTE.json')
assert route['migration_receipt']==str(root)
route.update(status='RUNNING',utc=record['utc'],current_updates=metrics[-1]['updates'],exact_resume_boundary_passed=True,
             continuation_evidence=str(root/'CONTINUATION_ACCEPTED.json'))
write(run/'CURRENT_RESOURCE_ROUTE.json',route)
print(json.dumps({k:record[k] for k in ('passed','host','parent_completed','confirmed_update','continuous_optimizer_updates','all_observed_steps_median_seconds')}))
