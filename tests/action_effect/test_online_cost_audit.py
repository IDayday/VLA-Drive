"""CPU cost bookkeeping negatives, not full-VLA scientific qualification."""
from types import SimpleNamespace
import os
import subprocess
import sys
import pytest
import torch

from tools.action_effect import audit_online_cost as a
from tools.action_effect.common import artifact


def test_only_reported_operator_FLOPs_are_counted():
    events=[SimpleNamespace(key='aten::addmm',count=2,input_shapes=[[2,4],[4,3]],flops=48),
            SimpleNamespace(key='aten::_scaled_dot_product_flash_attention_for_cpu',count=2,input_shapes=[],flops=0),
            SimpleNamespace(key='aten::relu',count=1,input_shapes=[[2,3]],flops=None)]
    out=a.event_summary(events)
    assert out['operator_covered_FLOPs_lower_bound']==48
    assert out['attention_operators'][0]['reported_FLOPs']==0
    assert len(out['zero_reported_operators'])==2 and 'not zero work' in out['coverage']


def test_tiny_actual_CPU_linear_profiler_agrees_with_known_matmul_count(tmp_path):
    # Other tests may legitimately initialize CUDA. Prove this CPU tool never
    # does so in its own fresh CPU-only process, as the actual audit is run.
    code = '''
import sys, torch
from pathlib import Path
from tools.action_effect.audit_online_cost import profile_call
from tools.action_effect.common import artifact
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
model=torch.nn.Linear(4,3).eval(); inputs=torch.ones(2,4)
trace=Path(sys.argv[1])
out=profile_call(lambda:model(inputs), trace=trace)
assert out['operator_covered_FLOPs_lower_bound']==2*2*4*3
assert out['trace']==artifact(trace)
assert not torch.cuda.is_initialized()
'''
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, '-c', code, str(tmp_path/'trace.json')],
                   env=env, check=True, capture_output=True, text=True)


def test_exact_comparison_preserves_signed_zero_and_rejects_nonfinite():
    assert a.exact_tensor(torch.tensor([1.]),torch.tensor([1.]))
    assert not a.exact_tensor(torch.tensor([0.]),torch.tensor([-0.]))
    assert not a.exact_tensor(torch.tensor([float('nan')]),torch.tensor([float('nan')]))
    assert not a.exact_tensor(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))


def test_wrong_source_hash_and_path_escape_are_rejected(tmp_path):
    file=tmp_path/'module.py';file.write_text('X=1\n')
    with pytest.raises(ValueError):a.validate_source(tmp_path,{'source_inventory':{'module.py':'wrong'}})
    with pytest.raises(ValueError):a.validate_source(tmp_path,{'source_inventory':{'../module.py':'wrong'}})
    with pytest.raises(ValueError):a.validate_source(tmp_path,{})


def test_GPU_visible_environment_refused_before_reading_weights(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    args=SimpleNamespace(output=str(tmp_path/'out'))
    with pytest.raises(ValueError,match='CPU-only'):a.audit(args)
    assert not (tmp_path/'out').exists()
