"""Parse the two real builder commands and native resource guards; never build."""
import hashlib,json,os,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).absolute().parent
START=time.monotonic()
with open('/proc/self/io') as f:BEFORE={k:int(v) for k,v in (s.split(':') for s in f)}
plan=json.loads((HERE/'PREPARATION_PLAN.json').read_text())
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only preparation check')
sys.path.insert(0,plan['runtime']['path'])
from tools.st_world_policy import prepare_causal_history as writer,metadata_storage as storage
checks=[]
for role in ('train','dev'):
    row=plan['roles'][role];p=row['storage_registration'];raw=Path(p['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=p['sha256']:raise ValueError('Draft pin changed')
    reg=json.loads(raw);args=writer.parser().parse_args(reg['command'][3:]);storage._resources(reg)
    if args.max_scenes!=row['population']['scenes'] or args.split!=role or args.target_times!=[0.] or args.resume:
        raise ValueError('Full original role/current-only target mismatch')
    for pin in reg['source_pins']:
        if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Existing source changed')
    with patch.object(storage,'_ACTIVE_GUARD',SimpleNamespace(registration=reg,limits=reg['limits'])):
        writer.validate(args)  # Actual native entry guard, no actual build/guard/ledger admission.
    checks.append(dict(role=role,scenes=args.max_scenes,record_layout=args.record_layout,
        native_parser=True,native_resource_guard=True,native_packed_population_guard=True,build_called=False))
with open('/proc/self/io') as f:AFTER={k:int(v) for k,v in (s.split(':') for s in f)}
print(json.dumps(dict(status='STATIC_COMMAND_GUARDS_PASS_NOT_BUILD_APPROVAL',checks=checks,elapsed_seconds=time.monotonic()-START,
    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,charged_rchar=AFTER['rchar']-BEFORE['rchar'],
    population_read=False,images_or_logs_read=False,ledger_accessed=False,gpu_used=False),indent=2))
