"""Read-only SP0 control check. No weights, images, ledger or cache writes."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import spatial_w_storage as w

run = Path('/mnt/project/st-world-policy-20261016/formal/spatial/runs/SP0_fulltrain_seed42_official_5k_v1')
registration = Path('/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP0_fulltrain_seed42_official_5k_v1/REGISTRATION.json')
receipt_path = run/'checkpoints/step_005000_1792217229597178910/COMPLETE.json'
opened = []
read_bytes = 0

def small(path, expected=None):
    global read_bytes
    path = b._path(str(path))
    with path.open('rb') as f:
        before=os.fstat(f.fileno()); raw=f.read((1<<20)+1); after=os.fstat(f.fileno())
    assert len(raw) <= 1<<20 and m._version(before)==m._version(after)==m._version(path.stat())
    sha=hashlib.sha256(raw).hexdigest()
    if expected is not None: assert sha==expected
    opened.append({'path':str(path),'bytes':len(raw),'sha256':sha})
    read_bytes += len(raw)
    return m._json(raw),sha

started=time.perf_counter()
reg,reg_sha=small(registration)
cfg,cfg_sha=small(reg['artifacts']['config']['path'],reg['artifacts']['config']['sha256'])
base,base_sha=small(reg['artifacts']['base_binding']['path'],reg['artifacts']['base_binding']['sha256'])
w._sealed(base,'binding_sha256')
receipt,receipt_sha=small(receipt_path)
w._sealed(receipt,'receipt_identity')
original=os.open
run_opens=[]
def checked_open(path,*a,**k):
    assert Path(path)==run/'identity.json'
    run_opens.append(str(path))
    return original(path,*a,**k)
w.os.open=checked_open
cold_start=time.perf_counter()
projection=w._run_identity_projection(run/'identity.json')
cold_seconds=time.perf_counter()-cold_start
assert projection['bytes']==19105073
assert projection['identity']==receipt['state']['identity']
assert receipt['state']['completed']==5000 and receipt['world_size']==cfg['execution']['world_size']
assert projection['config_sha256']==b._digest(cfg)
assert projection['source_sha']==reg['source_sha']
assert projection['registration_sha256']==reg_sha
assert projection['base_binding']==base['binding_sha256']
assert cfg['execution']['scope']=='FORMAL' and cfg['space_time']['cell']=='S-P0'
assert projection['weights_usable_for_formal_init'] is True
warm_start=time.perf_counter()
for _ in range(100): assert w._run_identity_projection(run/'identity.json') is projection
warm_seconds=time.perf_counter()-warm_start
assert len(run_opens)==1 and len(w._RUN_IDENTITIES)==1
read_bytes+=projection['bytes']
assert read_bytes <=32<<20
report=dict(schema='st_frozen_w_large_run_identity_check_v1',status='PASS_CONTROL_ONLY',
    source_start_commit='cf5805016b5e23bdab9541c77b562ed5fda83dc3',
    projection=dict(projection), projection_json_bytes=len(json.dumps(dict(projection))),
    controls=opened, run_identity_open_count=len(run_opens), warm_calls=100,
    cold_seconds=cold_seconds,warm_total_seconds=warm_seconds,
    wall_seconds=time.perf_counter()-started,metadata_bytes_read=read_bytes,
    rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    cpu_affinity=sorted(os.sched_getaffinity(0)),max_address_space_bytes=2<<30,
    control_limit_unchanged=m.CONTROL_LIMIT,run_identity_limit=w.RUN_IDENTITY_LIMIT,
    model_image_tensor_opens=0,real_ledger_mutations=0,native_payload_reverified=False,
    formal_admission_granted=False)
print(json.dumps(report,sort_keys=True,indent=2))
