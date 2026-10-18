"""Read-only frozen frontend review; does not bind or execute analysis."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
REL = Path('reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1')
OWNER = Path('/mnt/project/VLA-Drive-st-spatial20k-evidence-20261018') / REL
LEAD = Path('/mnt/project/VLA-Drive-st-world-policy-20261016') / REL
def pin(path):
    path = Path(path)
    assert path.stat().st_size < 1 << 20
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
old, actual = OWNER/'run_common_once.py', LEAD/'run_common_once.py'
assert pin(old)['sha256'] == '13ddeb7e51b40297c9349eac9cc7580b35d641a7559df3647976dde225ea3736'
assert pin(actual)['sha256'] == '3cccc594f984cb293f7bd68b74696e0db30379370d81b439baf9cf6610feff1c'
old_ast = ast.parse(old.read_text()); new_ast = ast.parse(actual.read_text())
calls = [n for n in ast.walk(new_ast) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Attribute) and n.func.attr == 'run_pair']
assert len(calls) == 1 and isinstance(calls[0].args[6], ast.Name) and calls[0].args[6].id == 'serial'
calls[0].args[6].id = 'lock'
assert ast.dump(new_ast, include_attributes=False) == ast.dump(old_ast, include_attributes=False)
fixed_path = LEAD/'FIXED_COMMANDS.json'; fixed = json.loads(fixed_path.read_text())
assert pin(fixed_path)['sha256'] == '4da48038c29d6473fe2b5a09f3e7629293a577e0aeafb19ae10449c1a5c34673'
assert fixed_path.read_bytes() == (OWNER/'FIXED_COMMANDS.json').read_bytes()
assert fixed['endpoint'] == 4096 and fixed['stage'] == 'COMMON_ENDPOINT'
assert fixed['per_pair_limits'] == {'cleanup_seconds':40,'output_bytes_including_controls':32<<20,
    'read_bytes':1<<30,'rss_bytes':2<<30,'wall_seconds':1800}
assert fixed['max_invocations_per_arm'] == 1
analyzer = fixed['analysis_source']; assert pin(analyzer['path']) == analyzer
runtime = str(Path(analyzer['path']).parents[2])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip() == fixed['source_sha']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=runtime,text=True).strip()
monitor = Path('/mnt/project/VLA-Drive-st-c4-cpu6-execution-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py')
assert pin(monitor)['sha256'] == '8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644'
assert 'pass_fds=(fd,)' in monitor.read_text()
limits = {'actual_analysis_run':False,'payloads_read':False,'large_population_read':False,
          'CSV_or_bootstrap_executed':False,'models_loaded':False,'bind_called':False,'dispatch':False}
presence={}
base = Path('/mnt/project/st-world-policy-20261016')
for arm in ['C0','C1','C2','C3','C4']:
    root=base/'formal/evaluation/jobs'/f'{arm}_step004096_dev1696_seed42_v1'
    presence[arm]={'official_score_COMPLETE_present':(root/'official_score/COMPLETE.json').exists(),
                   'analysis_output_exists':(base/'formal/evaluation/analyses'/f'{arm}_minus_C0_COMMON_ENDPOINT_dev1696_v1').exists() if arm!='C0' else None}
result = {'schema':'st_common4096_frontend_independent_correctness_v1',
    'reviewer':'/root/spatial20k_correctness','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
    'status':'PASS_CORRECTED_FRONTEND_ONLY_WAITING_ACTUAL_INPUTS',
    'rejected_original':pin(old),'approved_corrected_source':pin(actual),
    'fixed_commands':pin(fixed_path),'retained_monitor':pin(monitor),'analyzer_unchanged':analyzer,
    'runtime_source':fixed['source_sha'],
    'finding':{'id':'SERIAL_LOCK_NOT_INHERITED','original_impact':'Parent loss could release cross-arm SERIAL while its child retains only the per-arm lock.',
       'fix':'Pass serial to original run_pair; retain both parent locks; original monitor forwards this FD to the child.',
       'delta':'Exactly one run_pair argument lock -> serial, plus explanatory comments; original monitor/analyzer unchanged.'},
    'tests':{'actual_corrected_frontend_metadata_fixtures':9,'actual_FD_boundary_tests':2,
       'failures':0,'skips':0,'metadata_log':pin(OUT/'ACTUAL_FRONTEND_TEST_RESULTS.txt'),
       'fd_log':pin(OUT/'LOCK_TEST_RESULTS.txt'),'scope':'CPU source/metadata and temporary file-descriptor counterexample only'},
    'checks':{'all_four_original_argv_env_cwd_limits_preserved':True,'only4096_complete_inputs':True,
      'old1024_and_partial_inputs_rejected':True,'unchanged_native_completion_payload_verifier':True,
      'one_GiB_read_including_binding':True,'two_GiB_aggregate_RSS':True,'thirtytwo_MiB_total_output':True,
      'wall1800_cleanup40':True,'no_automatic_retry':True,'per_arm_and_shared_lock':True,
      'shared_lock_retained_in_child':True,'new_COMMON4096_control_namespace':True},
    'current_presence_only_not_scientific_verification':presence,'execution_boundary':limits,
    'remaining':'Lead must bind actual complete score/export identities and check live resources. This is not a completed COMMON4096 analysis or launch admission.'}
with (OUT/'REVIEW.json').open('x') as stream:
    stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(pin(OUT/'REVIEW.json')))
