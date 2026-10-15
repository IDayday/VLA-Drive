import datetime, fcntl, hashlib, importlib.util, json, os, pathlib, socket, subprocess

root = pathlib.Path(__file__).resolve().parent
directory = root/'G2_EVENT_UNIFORM'
spec = importlib.util.spec_from_file_location('g2_native_controller',root/'continue_registered_navsim.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
host=socket.gethostname().removesuffix('-worker-0')
c.authorize(host)
m=c.read(directory/'REGISTRATION.json')
assert host == m['destination_host'] == 'training-vlawm-tsinghua-zt4'
run=pathlib.Path(m['run'])
with (root/'MIGRATION_LAUNCH.lock').open('a') as launch_lock:
    fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (directory/'CONTROLLER_LAUNCH.json').exists()
    assert not (directory/'TRAINING_LAUNCH.json').exists()
    assert c.read(root/'SOURCE_NATIVE_BOUNDARY_VERIFIED.json')['passed']
    assert c.read(root/'ENVIRONMENT_AND_DATA_ACCEPTED.json')['passed']
    assert c.read(root/'PARENT_CLIENT_STATE.json')['completed'] == 89574
    assert c.read(root/'ROLLBACK_CHECKPOINT_PIN.json')['same_inode_all_files']
    assert c.checkpoint(m)['completed'] == 89574
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    rows=[x.split(',') for x in raw.splitlines() if x.strip()]
    assert len(rows)==8 and all(int(x[2])>=m['minimum_free_MiB_per_GPU'] for x in rows)
    with (run/'HOST_MIGRATION_RUNTIME.lock').open('a') as runtime_lock:
        fcntl.flock(runtime_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=run/'STOP_AFTER_CHECKPOINT'
        value=c.read(stop)
        assert value['reason']=='Latest user withdraws vla-zt2; native checkpoint-safe relocation'
        assert not (root/'NATIVE_STOP_MARKER.json').exists()
        stop.rename(root/'NATIVE_STOP_MARKER.json')
    command=['/usr/bin/python3','-B','-u',str(root/'continue_registered_navsim.py'),'--group','G2_EVENT_UNIFORM']
    with (directory/'controller.log').open('ab') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),
            **c.process(child.pid),'command':command,'resource_sharing':True,
            'no_existing_process_signaled':True,'prelaunch_memory':raw}
    c.write(directory/'CONTROLLER_LAUNCH.json',record)
    print(json.dumps(record))
