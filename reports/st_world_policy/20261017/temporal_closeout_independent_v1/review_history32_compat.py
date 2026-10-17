"""Narrow compatibility check: no builder, no live ledger, no payloads or GPU."""
import importlib.util,json,hashlib,os,sys,time,resource,unittest
from pathlib import Path
from unittest.mock import patch
os.sched_setaffinity(0,{8,9});START=time.monotonic()
def io():
 with open('/proc/self/io') as f:return {k:int(v) for k,v in (x.split(':') for x in f)}
BEFORE=io();HERE=Path(__file__).resolve().parent
T=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261017/temporal_history32_preparation_v1/test_history32_binding.py')
assert hashlib.sha256(T.read_bytes()).hexdigest()=='d1ffe7d0996bcfd05a4e12383c00ce34094ce3e1c2f4190b742af2c07c0e97ee'
spec=importlib.util.spec_from_file_location('new_bridge_tests',T);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
suite=unittest.TestSuite([m.BindingTests('test_compatibility_pins_required_in_admission'),m.BindingTests('test_parent_and_child_install_before_original_main')]);suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(m.CompatibilityTests));result=unittest.TextTestRunner(verbosity=2).run(suite)
# After class teardown, install once in a fresh canonical import state. Parent and
# child execution use this same path; no original controller main is invoked.
proof=m.m.install_current_w_compatibility(m.m.W_COMPAT)
w=sys.modules['tools.st_world_policy.spatial_w_storage'];metadata=__import__('tools.st_world_policy.metadata_storage',fromlist=['x'])
policy=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261017/temporal_history32_preparation_v1/actual_preparation/PROPOSED_POLICY.json');raw=policy.read_bytes();assert hashlib.sha256(raw).hexdigest()=='c77dca4739f3026940b8a56821cfd4bd222c643cdf9e4e07e5b9efba21a82163';p=json.loads(raw)
# Original real common/package validator on actual admitted small signatures.
# The already-admitted deep native-run validation is explicitly not repeated.
with patch.object(w,'validate_signature',side_effect=lambda sig:(sig['binding'],w._requirements(len(sig['rows']),sig['binding']['memory_tokens'],sig['binding']['hidden_dim']))):
 total=w.validate_package({k:v for k,v in p.items() if k!='claims'},{r['root']:r for r in p['claims']})
assert total==138426288 and 'torch' not in sys.modules
metrics=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),actual_mixed_package_bytes=total,actual_policy_sha256=hashlib.sha256(raw).hexdigest(),deep_signature_validation='INHERITED_ADMITTED_NOT_REEXECUTED',source_compatibility=proof,metadata_module=str(metadata.__file__),budget_module=str(w.b.__file__),wall_seconds=time.monotonic()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,logical_read_bytes=io()['rchar']-BEFORE['rchar'],gpu=False,ssh=False,ledger_accessed=False,builder_executed=False)
assert metrics['peak_rss_bytes']<1<<30 and metrics['logical_read_bytes']<128<<20 and metrics['wall_seconds']<90
(HERE/'HISTORY32_COMPAT_TEST.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics));raise SystemExit(not result.wasSuccessful())
