"""Actual cold-Python fresh_environment import; all other route inputs fixtures."""
import argparse,hashlib,importlib,importlib.util,json,os,pathlib,socket,subprocess,sys,tempfile,types
p=argparse.ArgumentParser();p.add_argument('--source',choices=('old','new'),required=True);a=p.parse_args()
repo=pathlib.Path(__file__).resolve().parents[5]
controller=repo/'resources/st_rollout_resume_smoke_v1.py'
runtime=pathlib.Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
# Build a CPU-only actual package-version reference, then truly discard every
# tools module/path so the tested route starts with no accidental prior seed.
sys.path.insert(0,str(runtime))
from tools.st_world_policy.training_config import environment_record
environment=environment_record()
sys.path=[value for value in sys.path if value!=str(runtime) and value!=str(repo)]
for key in list(sys.modules):
    if key=='tools' or key.startswith('tools.'):del sys.modules[key]
try:importlib.import_module('tools')
except ModuleNotFoundError:pass
else:raise AssertionError('Probe was not cold: tools already importable')
raw=controller.read_bytes() if a.source=='new' else subprocess.check_output(['git','show','19a5a244cca2a07600339ba9be7c3a150ef61285:resources/st_rollout_resume_smoke_v1.py'],cwd=repo)
m=types.ModuleType('cold_resume_source');m.__file__=str(controller);exec(compile(raw,str(controller),'exec'),m.__dict__)
ep=m.helper();helpers=ep.mature_helpers();helpers.update(check_route_time_v3=lambda *a:None,shared_lock_mount_v3=lambda:None,cpu_capacity_v3=lambda:{'quota_us':-1,'period_us':100000})
ep.mature_helpers=lambda:helpers;m.helper=lambda:ep
host=socket.gethostname()
with tempfile.TemporaryDirectory(prefix='native-cold-route-') as temp:
    temp=pathlib.Path(temp)
    def put(path,value):
        path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    ep.DISPATCH_POLICY=temp/'policy.json';ep.AUTHORIZATION=temp/'authorization.json'
    binding={'dispatch_policy':put(ep.DISPATCH_POLICY,{'allowed_hosts':{'training-vlawm-tsinghua-zt6':{'aliases':[host]}},'allowed_aliases':[host],'global_dispatch_prohibitions':{}}),
        'authorization':put(ep.AUTHORIZATION,{'allowed_hosts':['training-vlawm-tsinghua-zt6']}),
        'route':{'hostname':host,'host':'training-vlawm-tsinghua-zt6','gpus':list(range(8)),'gpu_uuids':['fixture'+str(i) for i in range(8)]},
        'environment':put(temp/'environment.json',environment)}
    # No GPU query, mount operation, git traversal, affinity change or disk quota
    # claim: only the real frozen fresh_environment_v3 import is under test.
    m.subprocess.check_output=lambda argv,**kw:m.SOURCE+'\n' if argv==['git','rev-parse','HEAD'] else b''
    os.sched_getaffinity=lambda pid:set(range(32))
    import shutil
    shutil.disk_usage=lambda path:types.SimpleNamespace(free=2**50)
    try:answer=m.route_checks(binding,initial=False)
    except ModuleNotFoundError as error:
        if a.source!='old' or error.name!='tools':raise
        print(json.dumps({'source':'old19a5','outcome':'REPRODUCED_ModuleNotFoundError_tools','model_GPU_official_calls':0}));sys.exit(0)
    if a.source=='old':raise AssertionError('Old source unexpectedly did not reproduce missing tools')
    actual=importlib.import_module('tools.st_world_policy.training_config')
    assert pathlib.Path(actual.__file__).resolve().is_relative_to(runtime)
    assert str(runtime) in sys.path and answer[0]==binding['route']
    print(json.dumps({'source':'new_v2','outcome':'REAL_FROZEN_ENVIRONMENT_IMPORT_OK','actual_module':str(actual.__file__),'model_GPU_official_calls':0}))
