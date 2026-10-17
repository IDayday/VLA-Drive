"""CPU-only rejection and startup ordering tests; no CUDA allocation/dispatch."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.st_world_policy import calibrate_temporal_scale as cli

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('calibration_scoped_launcher', ROOT /
    'reports/st_world_policy/20261017/temporal_scale_calibration_v1/run_once.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
CFG = {'runtime_environment': cli.RUNTIME_ENVIRONMENT}
TRAINING = {'runtime': {'deterministic': True}}


@pytest.mark.parametrize('value', [None, '', ':16:8', ':4096:7'])
def test_missing_or_conflicting_environment_fails_before_cuda_check(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    else:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', value)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: pytest.fail('Environment check must come first'))
    with pytest.raises(ValueError, match='before starting'):
        cli.runtime_preflight(CFG, TRAINING)


def test_matching_environment_checks_without_initializing_cuda(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CUDA initialization forbidden'))
    assert not torch.cuda.is_initialized()
    result = cli.runtime_preflight(CFG, TRAINING)
    assert result['runtime_environment'] == cli.RUNTIME_ENVIRONMENT
    assert result['cuda_initialized_at_check'] is False
    assert result['deterministic_algorithms'] is True


def test_late_environment_repair_is_rejected(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='uninitialized CUDA context'):
        cli.runtime_preflight(CFG, TRAINING)


def test_determinism_and_declared_contract_cannot_be_relaxed(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    with pytest.raises(ValueError, match='deterministic=True'):
        cli.runtime_preflight(CFG, {'runtime': {'deterministic': False}})
    with pytest.raises(ValueError, match='registered'):
        cli.runtime_preflight({'runtime_environment': {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}}, TRAINING)
    with pytest.raises(ValueError, match='missing or unknown'):
        cli.validate({'schema': cli.SCHEMA})


def test_measure_fails_before_admission_payload_or_output(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    output = tmp_path / 'output'
    config.write_text(json.dumps({**CFG, 'output_root': str(output)}))
    monkeypatch.setattr(cli, 'validate', lambda cfg: (TRAINING, None, None, None))
    monkeypatch.setattr(cli, 'admission_check', lambda *a: pytest.fail('Admission/device inspection not reached'))
    monkeypatch.setattr(cli, 'execute', lambda *a: pytest.fail('Payload loading not reached'))
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(sys, 'argv', ['calibration', 'measure', '--config', str(config),
                                   '--admission', 'unused', '--admission-sha256', 'unused'])
    assert cli.main() == 2
    assert not output.exists()


def test_execute_also_checks_before_payload_loading(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ValueError, match='before starting'):
        cli.execute(CFG, TRAINING, None, None)


def test_launcher_sets_child_environment_without_mutating_parent():
    parent = {'OTHER': 'retained'}
    actual = launcher.child_runtime_environment(parent)
    assert parent == {'OTHER': 'retained'}
    assert actual == {**parent, **cli.RUNTIME_ENVIRONMENT}
    assert launcher.child_runtime_environment(actual) == actual


@pytest.mark.parametrize('value', ['', ':16:8', 'garbage'])
def test_launcher_rejects_conflicting_inherited_environment(value):
    with pytest.raises(ValueError, match='Incompatible inherited'):
        launcher.child_runtime_environment({'CUBLAS_WORKSPACE_CONFIG': value})


@pytest.mark.parametrize('lease,expected', [(3039, 'lease'), (3040, 'sources_reached')])
def test_lease_uses_reduced_wall_plus_cleanup(monkeypatch, lease, expected):
    control = dict(schema='st_loss_scale_owned_control_v1', limits=launcher.LIMITS,
        host=launcher.socket.gethostname(), runtime=str(launcher.RUNTIME), output_root=str(launcher.ROOT),
        runtime_environment=cli.RUNTIME_ENVIRONMENT, issued_at_unix=0, latest_start_unix=1,
        valid_until_unix=lease, sources=[{'path': 'sentinel'}])
    def pin(value, **kwargs):
        if value['path'] == 'sentinel':
            raise ValueError('sources_reached')
        return control
    monkeypatch.setattr(launcher, 'pin_read', pin)
    monkeypatch.setattr(launcher.time, 'time', lambda: 0)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ValueError, match=expected):
        launcher.controls(SimpleNamespace(control='unused', control_sha256='unused', child=False))


def test_second_attempt_has_separate_root_and_remaining_budget():
    assert launcher.ROOT.name == 'loss_scale_calibration_real4_v2'
    assert launcher.LIMITS['wall_seconds'] == 3000
    assert launcher.LIMITS['cleanup_seconds'] == 40
    assert 416.6046734880656 + 3000 + 40 <= 3600 + 40
    assert 51_404_923_553 + launcher.LIMITS['read_bytes'] <= 256 << 30
    assert launcher.LIMITS['read_bytes'] == 208 << 30
    assert launcher.LIMITS['optimizer_updates'] == launcher.LIMITS['new_tensor_bytes'] == 0
