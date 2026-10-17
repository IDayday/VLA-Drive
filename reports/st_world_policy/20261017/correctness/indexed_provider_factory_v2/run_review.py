"""Run only independent counterexamples and four adjacent legacy/indexed checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


EXPECTED = 'ac6b93a43de6d41585a1c01e3370f5a6db41f456'
root = Path.cwd()
here = Path(__file__).resolve().parent
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED
assert subprocess.check_output(['git', 'status', '--porcelain=v1'], text=True).strip() == ''
sys.path.insert(0, str(root)); sys.path.insert(1, str(root/'tests/st_world'))
paths = ['tools/st_world_policy/teacher_provider.py', 'tools/st_world_policy/provider_pca.py']
before = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
print(json.dumps({'tested_source': EXPECTED, 'runtime': str(root), 'source_sha256': before,
    'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'scope': 'CPU_generated_fixtures_only'}), flush=True)
import pytest
status = pytest.main(['-vv', '-p', 'no:cacheprovider',
    str(here.parent/'indexed_provider_factory_v1/test_independent_provider_boundaries.py'),
    str(here/'test_preopen_scope_boundaries.py'),
    'tests/st_world/test_provider_training.py::test_provider_scope_reuses_exact_basis_without_fake_cache_identity',
    'tests/st_world/test_indexed_provider_factory.py::test_indexed_pca_basis_unchanged_scope_distinct_and_no_eager_fit_records',
    'tests/st_world/test_indexed_provider_factory.py::test_indexed_factory_no_eager_parse_and_preserves_keys_math_masks_rng_resume',
    'tests/st_world/test_indexed_provider_factory.py::test_actual_indexed_cli_metadata_only_and_strict_arguments'])
assert before == {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
assert subprocess.check_output(['git', 'status', '--porcelain=v1'], text=True).strip() == ''
print(json.dumps({'exit_code': int(status), 'source_unchanged': True, 'runtime_clean': True}), flush=True)
raise SystemExit(status)
