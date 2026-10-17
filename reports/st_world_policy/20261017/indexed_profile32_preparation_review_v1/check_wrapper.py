"""Bounded metadata/source review only; does not call factory or wrapper main."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
SCRIPT=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/indexed_profile32_preparation_v1/prepare_bounded.py')
spec=importlib.util.spec_from_file_location('reviewed_wrapper',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert 'torch' not in sys.modules
pin=lambda p:{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
tree=ast.parse(SCRIPT.read_text()); commands=m.commands()
assert len(commands)==3
assert commands[0][1]=='prepare-profile-migration'
assert commands[0][-1]==str(m.ROOT/'SOURCE_MIGRATION.json')
assert commands[1][commands[1].index('--output')+1]==str(m.ROOT/'PROVIDER.json')
assert commands[1][commands[1].index('--configured-output')+1]==str(m.ROOT/'configured')
assert commands[2][commands[2].index('--factory-root')+1]==str(m.ROOT)
for n in (1,2):
    source=m.RUNTIME/'tools/st_world_policy/teacher_provider.py' if n==1 else Path(commands[2][0])
    parsed=ast.parse(source.read_text())
    flags={arg.value for call in ast.walk(parsed) if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument' for arg in call.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)}
    assert all(x in flags for x in commands[n][1:] if x.startswith('--'))
node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='locked_counter_read')
events=[]
@contextmanager
def lock(directory):
    events.append('lock');yield;events.append('unlock')
def read(path,*args,**kwargs):
    events.append('read');return {'value':1},{}
env={'Path':Path,'ATTEMPT':m.ATTEMPT,'control_lock':lock,'original_read':read}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact-wrapper-reader>','exec'),env)
env['locked_counter_read'](m.ATTEMPT/'COUNTERS.json',limit=65536)
assert events==['lock','read','unlock'];events.clear()
env['locked_counter_read'](m.ATTEMPT/'ADMITTED.json',expected='fixed')
assert events==['read'];events.clear()
@contextmanager
def timeout(directory):
    raise TimeoutError('Bounded control write lock wait exceeded');yield
env['control_lock']=timeout
try:env['locked_counter_read'](m.ATTEMPT/'COUNTERS.json')
except TimeoutError:pass
else:raise AssertionError('Lock timeout swallowed')
assert events==[]
assert not m.ROOT.exists()
assert not Path(commands[2][-1]).exists()
paths=[m.PARENT,*m.PARENT.rglob('*')]
assert len(paths)<256 and not any(p.is_symlink() for p in paths)
used=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in paths)
assert used+(8<<20)<32<<20
handoff=json.loads((m.HANDOFF/'HANDOFF.json').read_text())
inputs=[]
for key in ('input_config','data_manifest','base_binding','parent_descriptor','reference_config','target_contract'):
    p=Path(handoff[key]['path']);assert p.stat().st_size<1<<20
    actual=pin(p);assert actual['sha256']==handoff[key]['sha256'];inputs.append({**actual,'bytes':p.stat().st_size})
assert handoff['metadata_output_root']==str(m.ROOT) and handoff['source_sha']==m.SOURCE
helperpins={name:pin(m.HELPERS/name)['sha256'] for name in ('supervise_once.py','prepare_bindings.py','observed_cli.py')}
now=time.time()
admission={'schema':'st_independent_profile_metadata_admission_v1','status':'PASS_ONCE_METADATA_ONLY','reviewer':'/root/st_metadata_resource_review','issued_at_unix':now,'valid_until_unix':now+600,'latest_start_unix':now+380,'script':pin(SCRIPT),'source_commit':m.SOURCE,'runtime':str(m.RUNTIME),'handoff':pin(m.HANDOFF/'HANDOFF.json'),'relocate':pin(m.HANDOFF/'relocate_config.py'),'helpers':helperpins,'output_root':str(m.ROOT),'existing_parent':str(m.PARENT),'existing_parent_charged_bytes':used,'parent_inclusive_cap_bytes':32<<20,'additional_metadata_reservation_bytes':0,'effective_campaign_metadata_reserved_mib':1264,'cpu_affinity':[0,1],'rss_bytes':2<<30,'wall_seconds':180,'cleanup_seconds':40,'logical_read_bytes':4<<30,'child_read_cap_bytes':m.READ_CAP,'known_pca_bytes':1590864,'parent_read_allowance_bytes':16<<20,'gpu':False,'new_teacher_tensor_bytes':0,'training_approved':False,'checkpoint_namespace_creation_approved':False,'automatic_retry':False,'checks':['three_exact_cli_routes_match_source','counter_only_same_writer_lock','immutable_reader_unmodified','lock_timeout_fail_closed','runtime_clean_checked_separately','existing_parent_budget_and_inputs_verified'],'inputs':inputs}
(HERE/'ADMISSION.json').write_text(json.dumps(admission,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':admission['checks'],'script':admission['script'],'admission':pin(HERE/'ADMISSION.json'),'existing_parent_bytes':used,'existing_path_count':len(paths),'valid_until_unix':admission['valid_until_unix'],'latest_start_unix':admission['latest_start_unix']},sort_keys=True,indent=2))
