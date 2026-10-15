import datetime, fcntl, hashlib, json, pathlib, time

root = pathlib.Path(__file__).resolve().parent
run = pathlib.Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/formal_navsim_local_G3_G2_v3_io/G2_EVENT_UNIFORM')
old = pathlib.Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/G2_vla_zt2_migration_20261014T0700Z/G2_EVENT_UNIFORM')
state = json.loads((old/'STATE.json').read_text())
tag = (run/'checkpoints/latest').read_text().strip()
assert state == {'status':'PAUSED', 'completed':89574, 'tag':'latest_0089574'} and tag == state['tag']
check = run/'checkpoints'/tag
receipt = json.loads((check/'COMPLETE.json').read_text())
assert receipt['completed'] == state['completed'] and receipt['status'] == 'PAUSED' and receipt['ranks'] == 8
assert receipt['identity'] == json.loads((run/'identity.json').read_text())['identity']
assert len(receipt['file_sha256']) == 17
with (run/'HOST_MIGRATION_RUNTIME.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    prefix = (run/'metrics.jsonl').stat()
    rows = []
    for name, expected in receipt['file_sha256'].items():
        h=hashlib.sha256()
        with (check/name).open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
        assert h.hexdigest() == expected, name
        rows.append({'file': name, 'sha256': expected, 'bytes': (check/name).stat().st_size})
    now = (run/'metrics.jsonl').stat()
    assert now.st_size == prefix.st_size and now.st_mtime_ns == prefix.st_mtime_ns
    assert json.loads((old/'STATE.json').read_text()) == state
    record = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed':True, 'completed':89574,
              'source_controller_state':state,'runtime_lock_exclusive_acquired':True,
              'source_training_has_exited_by_controller_child_wait_and_runtime_lock':True,
              'no_source_host_SSH_or_machine_wide_observation':True,
              'old_host_physical_GPU_usage':'unknown; no remote connection allowed',
              'checkpoint':str(check), 'complete_receipt':receipt, 'files':rows,
              'metrics_prefix_bytes':prefix.st_size, 'metrics_unchanged_during_validation':True,
              'wall_seconds':time.monotonic()-start}
    (root/'SOURCE_NATIVE_BOUNDARY_VERIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':True,'files':len(rows),'completed':89574,'wall_seconds':record['wall_seconds']}))
