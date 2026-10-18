"""Record real CPU migration open paths; no temporal preparation or dispatch."""
import collections, contextlib, hashlib, io, json, os, resource, runpy, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNTIME=Path('/mnt/project/VLA-Drive-st-temporal-packed-migration-fix-20261018')
start=time.monotonic()
def io_read():
    return int(next(r.split(':')[1] for r in Path('/proc/self/io').read_text().splitlines() if r.startswith('rchar:')))
before=io_read(); opened=collections.Counter(); blocked=[]; opaque=[]
def audit(event,args):
    if event!='open':return
    if not isinstance(args[0],(str,bytes)):
        opaque.append(str(args[0]));return
    path=os.fsdecode(args[0]);opened[path]+=1
    if Path(path).suffix.lower() in {'.sqlite','.db','.safetensors','.pt','.pth','.pkl','.jpg','.jpeg','.png'}:
        blocked.append(path);raise AssertionError('Payload/DB forbidden in metadata source migration: '+path)
sys.addaudithook(audit)
sys.path.insert(0,str(RUNTIME)); sys.dont_write_bytecode=True
output=HERE/'SOURCE_MIGRATION_OPEN_AUDIT.json'
sys.argv=['qualify_indexed_provider','prepare-profile-migration','--parent-descriptor','/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json','--parent-runtime','/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017','--output',str(output)]
stream=io.StringIO()
with contextlib.redirect_stdout(stream):
    try:runpy.run_module('tools.st_world_policy.qualify_indexed_provider',run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None,0):raise
from tools.st_world_policy import qualify_indexed_provider as q
migration=json.loads(output.read_text());q.validate_control_migration(q.tp.pinned(migration['parent_plan']),migration)
assert migration['mathematics_compatibility']==json.loads((HERE/'MATHEMATICS_COMPATIBILITY.json').read_text())
read=io_read()-before
assert read<256<<20 and not blocked
report={'actual_native_migration_and_validator':'PASS','actual_temporal_preparation':False,'GPU':False,'ledger_written':False,'independent_approval':False,'elapsed_seconds':time.monotonic()-start,'self_rchar_bytes':read,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'opened_paths':dict(sorted(opened.items())),'blocked_payload_paths':blocked,'integer_fd_open_events':opaque,'audit_scope':'Python audit open events only; imported native library/internal libc and git child reads are not path-traced; self rchar includes parent reads','migration':{'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},'runtime_source_sha':'cf516259c8efc24503727f14b9e330ed32231293'}
(HERE/'OPENED_PATH_AUDIT.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
(HERE/'NATIVE_CLI.log').write_text(stream.getvalue())
print(json.dumps({k:v for k,v in report.items() if k not in ('opened_paths','integer_fd_open_events')}))
