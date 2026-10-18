"""Execute only the reviewed import prefix; stop before policy/environment IO."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
raw = source.read_bytes()
spec = importlib.util.spec_from_file_location('native_smoke_prefix_only', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tree = ast.parse(raw)
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'route_checks')
original = list(sys.path)
class StopBeforeRealChecks(Exception):
    pass
def stop():
    assert sys.path[0] == str(module.RUNTIME)
    assert (module.RUNTIME/'tools/st_world_policy/training_config.py').is_file()
    raise StopBeforeRealChecks()
environment = {'sys': sys, 'RUNTIME': module.RUNTIME, 'helper': stop}
try:
    sys.path[:] = [entry for entry in sys.path if entry != str(module.RUNTIME)]
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), environment)
    try:
        environment['route_checks']({}, initial=True)
    except StopBeforeRealChecks:
        pass
    else:
        raise AssertionError('Expected stop before any actual environment operation')
    assert module.ROOT.name == 'rollout_scale_native_resume_v2'
    assert 'torch' not in sys.modules
    print(json.dumps({'status': 'PASS', 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'new_ROOT': str(module.ROOT), 'cold_Route_RUNTIME_prefix': True,
        'environment_check_executed': False, 'model_GPU_or_scoring': False,
        'lifecycle_tests_rerun': False}))
finally:
    sys.path[:] = original
