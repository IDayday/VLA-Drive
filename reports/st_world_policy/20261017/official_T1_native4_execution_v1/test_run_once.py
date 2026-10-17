"""CPU-only bounded control/ownership/hook fixtures; never a trainer invocation."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('t1_adapter', HERE/'run_once.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def pinned(path, value):
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def recipe():
    cfg = json.loads((m.RUNTIME/m.REL/'official_T1_native4_preparation_v1/PROPOSED_CONFIG_STEP2.json').read_text())
    reg = {'source_sha': m.SOURCE, 'execution': cfg['execution'], 'weights_usable_for_formal_init': False,
           'run_id': 'T1_official_real64_phase2_v1'}
    return reg, cfg


def test_exact_two_invocations_original_timeline():
    reg, cfg = recipe(); proposed = copy.deepcopy(cfg)
    m.validate_recipe(reg, cfg, 2, proposed)
    cfg['execution']['stop_after'] = 4; cfg['continuation'] = {'actual_parent': 'fixture'}
    reg['run_id'] = 'T1_official_real64_phase4_v1'
    m.validate_recipe(reg, cfg, 4, proposed)
    assert m.LIMITS['total_scene_exposures'] == 16 and m.LIMITS['max_invocations'] == 2
    cfg['execution']['max_updates'] = 4
    with pytest.raises(ValueError, match='recipe'): m.validate_recipe(reg, cfg, 4, proposed)


@pytest.mark.parametrize('phase', [0, 1, 3, 6])
def test_no_third_or_alternative_phase(phase):
    reg, cfg = recipe()
    with pytest.raises(ValueError, match='phase'): m.validate_recipe(reg, cfg, phase, cfg)


def test_parent_must_be_actual_success_and_same_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    regpin = {'path': '/fixed/REGISTRATION.json', 'sha256': 'a'*64}
    folder = tmp_path/'runs/T1_official_real64_phase2_v1/checkpoints/step_2'; folder.mkdir(parents=True)
    cp = pinned(folder/'COMPLETE.json', {'state': {'completed': 2}, 'world_size': 2})
    previous = dict(schema='st_T1_native4_execution_result_v1', status='COMPLETE', phase=2,
        cleanup={'live_owned_remaining': 0}, completed=2, native_receipt=cp,
        registration=regpin, cumulative_rchar_bytes=123)
    (tmp_path/'phase2').mkdir()
    pp = pinned(tmp_path/'phase2/RESULT.json', previous)
    a = {'phase': 4, 'previous_phase': pp, 'parent_receipt': cp}
    cfg = {'continuation': {'parent_registration': regpin, 'parent_receipt_sha256': cp['sha256'], 'parent_tag': folder.name}}
    assert m.check_prior(a, cfg)[0] == 123
    previous['status'] = 'FAILED_NO_AUTO_RETRY'; a['previous_phase'] = pinned(tmp_path/'phase2/RESULT.json', previous)
    with pytest.raises(ValueError, match='successful'): m.check_prior(a, cfg)


def test_pin_mutation_symlink_and_atomic_no_retry(tmp_path):
    pin = pinned(tmp_path/'control.json', {'ok': True})
    assert m.pin_read(pin) == {'ok': True}
    (tmp_path/'control.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'): m.pin_read(pin)
    link = tmp_path/'link'; link.symlink_to(tmp_path/'control.json')
    with pytest.raises(ValueError, match='symlink'): m.pin_read({'path': str(link), 'sha256': '0'*64})
    marker = tmp_path/'STARTED.json'; m.atomic_once(marker, {'attempt': 1})
    with pytest.raises(FileExistsError): m.atomic_once(marker, {'attempt': 2})
    assert json.loads(marker.read_text())['attempt'] == 1


def test_restore_hook_does_not_block_updates_or_repeat_on_save():
    events = []
    class Engine:
        def load_checkpoint(self, load_dir, tag=None, load_module_strict=True,
                            load_optimizer_states=True, load_lr_scheduler_states=True,
                            load_module_only=False, custom_load_fn=None):
            events.append('load'); return ('folder', {'completed': 2})
        def backward(self): events.append('backward')
        def step(self): events.append('step')
    trainer = SimpleNamespace(restore_rng=lambda s, g: events.append('restore'))
    audit = SimpleNamespace(after_load=lambda *a: events.append('live_load'),
                            after_restore=lambda *a: events.append('live_rng'))
    observer = SimpleNamespace(collective_audit=lambda fn, desc: fn())
    original = Engine.load_checkpoint, trainer.restore_rng, Engine.backward, Engine.step
    engine = Engine()
    with m.observe_restore(trainer, Engine, audit, observer, lambda _: {}):
        engine.load_checkpoint('/native', 'step2'); trainer.restore_rng({}, {})
        engine.backward(); engine.step(); engine.backward(); engine.step()
        trainer.restore_rng({}, {})  # original native save also restores its captured RNG
    assert events == ['load', 'live_load', 'restore', 'live_rng', 'backward', 'step', 'backward', 'step', 'restore']
    assert original == (Engine.load_checkpoint, trainer.restore_rng, Engine.backward, Engine.step)


def test_restore_rejects_non_strict_and_restores_hooks_on_failure():
    class Engine:
        def load_checkpoint(self, load_dir, tag=None, load_module_strict=True,
                            load_optimizer_states=True, load_lr_scheduler_states=True,
                            load_module_only=False, custom_load_fn=None): return None
    trainer = SimpleNamespace(restore_rng=lambda *a: None)
    original = Engine.load_checkpoint, trainer.restore_rng
    with pytest.raises(ValueError, match='Incomplete'):
        with m.observe_restore(trainer, Engine, None, None, None):
            Engine().load_checkpoint('/n', 't', load_optimizer_states=False)
    assert original == (Engine.load_checkpoint, trainer.restore_rng)


def helpers():
    sys.path.insert(0, str(m.RUNTIME/m.REL/'spatial_probe_actual_preparation_v1'))
    import supervise_once
    core = m.load_module('test_t1_io_core', m.RUNTIME/m.REL/'full_future_metadata_preparation_v1/run_build_once.py')
    return supervise_once, core


def test_reused_terminal_accounting_does_not_double_charge_child():
    _, core = helpers()
    row = core.terminal_read_accounting({'rchar': 10}, {'rchar': 20}, {'rchar': 125}, 100, 1000, final_child_available=True)
    assert row['charged_parent_child_rchar_bytes'] == 115
    assert core.charge_after_reap(row, {'rchar': 130}) == 120
    bad = core.terminal_read_accounting({'rchar': 10}, {'rchar': 20}, {'rchar': 30}, 100, 1000, final_child_available=True)
    assert bad['status'] == 'INCOMPLETE_CONSERVATIVE' and bad['charged_parent_child_rchar_bytes'] == 1000


@pytest.mark.parametrize('stop', [False, True])
def test_owned_monitor_tiny_children_no_gpu(tmp_path, monkeypatch, stop):
    s, core = helpers(); real_popen = subprocess.Popen
    expected_cpus = sorted(os.sched_getaffinity(0))
    assert len(expected_cpus) <= 2  # test lease: never borrow production CPUs 2-5
    def tiny_popen(command, **kwargs):
        assert command[:3] == ['taskset', '-c', '2-5']
        command[2] = ','.join(map(str, expected_cpus))
        kwargs['cwd'] = tmp_path
        return real_popen(command, **kwargs)
    monkeypatch.setattr(m.subprocess, 'Popen', tiny_popen)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'LIMITS', {**m.LIMITS, 'rss_bytes': 256 << 20, 'invocation_seconds': .1 if stop else 10})
    script = ('import subprocess,sys,time;'
              f'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep({2 if stop else .4})"]);'
              'p.wait();print("tiny child fixture")')
    with (tmp_path/'lock').open('w') as lock:
        out = m.monitor([sys.executable, '-c', script], dict(os.environ, CUDA_VISIBLE_DEVICES=''), lock.fileno(), s, core, 64 << 20)
    assert out['cleanup']['live_owned_remaining'] == 0
    assert out['accounting']['status'] == 'COMPLETE'
    assert out['failure'] == ('WALL_LIMIT' if stop else None)


def test_disk_budget_counts_native_and_partial_files(tmp_path):
    root = tmp_path/'runs'; root.mkdir()
    (root/'model.pt').write_bytes(b'a'*7); (root/'optimizer.pt.tmp').write_bytes(b'b'*11)
    assert m.checkpoint_bytes(root) == 18
    (root/'external').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='symlink'): m.checkpoint_bytes(root)


def test_existing_native_comparator_exact_cpu_tensor_bytes():
    # Reuse the real comparator, but no model/native checkpoint/image is loaded.
    m.pin_read({'path': str(m.OBSERVER), 'sha256': m.OBSERVER_SHA}, False)
    observer = m.load_module('test_existing_native_comparator', m.OBSERVER)
    import torch
    original = torch.tensor([1., 2., 3.], dtype=torch.float32)
    live = {'weight': original, 'optimizer': {'step': 2, 'exp_avg': original/10}}
    same = {'weight': original.clone(), 'optimizer': {'step': 2, 'exp_avg': original.clone()/10}}
    report = observer.summary(); observer.compare(live, same, 'fixture', report)
    assert observer.passed(report) and report['tensor_elements'] == 6
    same['weight'][1] += .01
    changed = observer.summary(); observer.compare(live, same, 'fixture', changed)
    assert not observer.passed(changed) and changed['differences'][0]['kind'] == 'tensor_bytes'


def test_actual_reused_helper_pins():
    for name, digest in m.HELPERS.items():
        assert m.pin_read({'path': str(m.RUNTIME/m.REL/name), 'sha256': digest}, False)


@pytest.mark.parametrize('end', [float('inf'), float('nan'), 'later'])
def test_unbounded_resource_window_rejected_before_any_runtime_read(monkeypatch, end):
    a = dict(schema='st_T1_native4_execution_admission_v1', status='APPROVED_BOUNDED_T1_ONLY',
        limits=m.LIMITS, hostname=m.socket.gethostname(), runtime={'path': str(m.RUNTIME), 'commit': m.SOURCE},
        adapter={'path': str(Path(m.__file__).absolute()), 'sha256': 'a'*64},
        issued_at_unix=0, latest_start_unix=1e20, valid_until_unix=end)
    monkeypatch.setattr(m, 'pin_read', lambda pin, decode=True: a if decode else b'fixture')
    with pytest.raises(ValueError, match='Finite'):
        m.controls(SimpleNamespace(admission='/fixture', admission_sha256='a'*64))
