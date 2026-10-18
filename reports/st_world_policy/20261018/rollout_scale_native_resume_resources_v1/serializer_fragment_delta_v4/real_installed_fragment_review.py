"""Actual installed DeepSpeed fragment class on CPU; no engine/model or GPU.

The test-only Inductor setting prevents unused compiler process-pool warmup.
No substitute class/module or AST class extraction is used.
"""
import copy
from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('DS_ACCELERATOR') == 'cpu'
assert os.environ.get('TORCHINDUCTOR_COMPILE_THREADS') == '1'
started = time.monotonic()
import torch
import numpy as np
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.tensor_fragment import fragment_address

assert get_accelerator().device_name() == 'cpu'
assert importlib.metadata.version('deepspeed') == '0.16.9'
installed_source = Path(inspect.getfile(fragment_address)).resolve()
assert installed_source == Path('/root/miniconda3/envs/ddp/lib/python3.10/site-packages/deepspeed/utils/tensor_fragment.py')
source = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('serializer_resource_review', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

address = fragment_address(numel=19, start=7)
master = torch.arange(4, dtype=torch.float32)
value = {'single_partition_of_fp32_groups': [master],
         'base_optimizer_state': {'state': {0: {'step': torch.tensor(0.),
              'exp_avg': torch.zeros_like(master), 'exp_avg_sq': torch.zeros_like(master)}}},
         'param_slice_mappings': [{'named_parameter': address}]}
python_rng = random.getstate()
numpy_rng = np.random.get_state()
torch_rng = torch.get_rng_state().clone()
original_master = master.clone()
first = m.logical_state(value)
assert first == m.logical_state(copy.deepcopy(value))
encoded_address = m.logical_state(address)
assert 'numel' in json.dumps(encoded_address) and 'start' in json.dumps(encoded_address)
def values(node):
    if isinstance(node, dict):
        return [item for value in node.values() for item in values(value)]
    if isinstance(node, list):
        return [item for value in node for item in values(value)]
    return [node]
assert 19 in values(encoded_address) and 7 in values(encoded_address)
assert m.logical_state(replace(address, numel=20)) != encoded_address
assert m.logical_state(replace(address, start=8)) != encoded_address
assert address.numel == 19 and address.start == 7 and torch.equal(master, original_master)
@dataclass
class NotADeepSpeedFragment:
    numel: int
    start: int
try:
    m.logical_state(NotADeepSpeedFragment(19, 7))
except ValueError:
    pass
else:
    raise AssertionError('Arbitrary dataclass was silently accepted')
assert random.getstate() == python_rng
current_numpy_rng = np.random.get_state()
assert current_numpy_rng[0] == numpy_rng[0] and np.array_equal(current_numpy_rng[1], numpy_rng[1])
assert current_numpy_rng[2:] == numpy_rng[2:]
assert torch.equal(torch.get_rng_state(), torch_rng)
assert not torch.cuda.is_initialized()  # Pure flag read, no CUDA initialization/query.
usage = resource.getrusage(resource.RUSAGE_SELF)
report = {'status': 'PASS', 'reviewer': '/root/road_failure_resources_review_v1',
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'actual_installed_class': str(type(address)),
          'installed_class_source': str(installed_source),
          'installed_class_source_sha256': hashlib.sha256(installed_source.read_bytes()).hexdigest(),
          'deepspeed_version': importlib.metadata.version('deepspeed'),
          'accelerator': get_accelerator().device_name(),
          'test_only_compile_threads': os.environ['TORCHINDUCTOR_COMPILE_THREADS'],
          'dataclass_numel_and_start_preserved': True, 'changed_fields_change_encoding': True,
          'nested_CPU_tensor_optimizer_like_input': True, 'real_full_engine_created': False,
          'arbitrary_dataclass_rejected': True, 'input_tensors_and_RNG_unchanged': True,
          'wall_seconds': time.monotonic()-started,
          'CPU_seconds': usage.ru_utime+usage.ru_stime,
          'post_test_current_RSS_bytes': int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
          'rusage_highwater_may_include_pre_exec_bytes': usage.ru_maxrss*1024,
          'large_payload_reads': 0, 'actual_model_or_official_calls': 0, 'CUDA_initialized': False,
          'old_suites_or_profiles_rerun': False,
          'boundary': 'Real installed class import; tiny CPU values, not a full native/engine integration test.'}
assert report['CPU_seconds'] < 60 and report['post_test_current_RSS_bytes'] < 2*2**30
Path(sys.argv[2]).write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status': 'PASS', 'wall_seconds': report['wall_seconds'], 'CPU_seconds': report['CPU_seconds']}))
