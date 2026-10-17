"""Narrow independent controls tests; no model/payload/ledger access."""
import ast
import builtins
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / 'w_storage_source_compat_v1'
spec = importlib.util.spec_from_file_location('independent_exact_compat', IMPL / 'compat.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


@pytest.fixture
def wrapper(monkeypatch):
    monkeypatch.setitem(sys.modules, 'compat', c)
    spec = importlib.util.spec_from_file_location('independent_compat_wrapper', IMPL / 'run_once.py')
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    return r


def admitted(wrapper):
    return dict(
        source_compat_controller={'path': str(IMPL / 'run_once.py'),
            'sha256': hashlib.sha256((IMPL / 'run_once.py').read_bytes()).hexdigest()},
        frozen_W_source_compatibility={'path': 'reviewed-proof', 'sha256': 'a' * 64},
        preparation={'path': 'preparation', 'sha256': 'b' * 64},
        stage='dev_extract', storage_policy={'path': 'unchanged-policy'},
        checkpoint_commitments={'identity': 'unchanged-commitments'})


def configure(wrapper, monkeypatch, admission, events, command='run', observe=None):
    preparation = {'runtime': {'path': '/synthetic/unchanged-runtime'}}
    s = types.SimpleNamespace()
    s.main = lambda: events.append(('supervisor', list(sys.argv))) or 19
    s.validate = lambda *args: events.append(('validate', args[:3]))
    ctl = types.SimpleNamespace(UP=Path('/synthetic/original-supervisor'))
    ctl.document = lambda path, sha: (preparation if path == 'preparation' else admission, None)
    ctl.controls = lambda arm, supervisor: (None, None, s)
    ctl.module = observe
    monkeypatch.setattr(wrapper, 'original_control', lambda: ctl)
    monkeypatch.setattr(c, 'verify', lambda pin: events.append(('verify', pin)))
    monkeypatch.setattr(sys, 'argv', ['wrapper', command, '--arm', 'SP1', '--admission',
        'admission', '--admission-sha256', 'c' * 64] +
        (['--attempt', '/synthetic/attempt', '--read-cap', '321'] if command == 'observe' else []))
    return preparation


@pytest.mark.parametrize('fault', ['controller_path', 'controller_sha', 'missing_proof'])
def test_admission_rejects_before_supervisor_or_install(wrapper, monkeypatch, fault):
    admission = admitted(wrapper); events = []
    if fault == 'controller_path': admission['source_compat_controller']['path'] = '/foreign/wrapper'
    if fault == 'controller_sha': admission['source_compat_controller']['sha256'] = '0' * 64
    if fault == 'missing_proof': admission.pop('frozen_W_source_compatibility')
    configure(wrapper, monkeypatch, admission, events)
    with pytest.raises(ValueError): wrapper.main()
    assert events == []


def test_run_keeps_original_supervisor_arguments(wrapper, monkeypatch):
    admission = admitted(wrapper); events = []
    configure(wrapper, monkeypatch, admission, events)
    monkeypatch.setattr(c, 'install', lambda pin: pytest.fail('parent must not install observer adapter'))
    assert wrapper.main() == 19
    assert [event[0] for event in events] == ['verify', 'supervisor']
    assert events[-1][1] == ['/synthetic/original-supervisor/supervise_once.py',
        '--admission', 'admission', '--admission-sha256', 'c' * 64]


def test_bad_proof_stops_before_supervisor(wrapper, monkeypatch):
    events = []; configure(wrapper, monkeypatch, admitted(wrapper), events)
    def reject(pin): raise ValueError('actual proof validator rejection')
    monkeypatch.setattr(c, 'verify', reject)
    with pytest.raises(ValueError, match='proof validator'): wrapper.main()
    assert events == []


def test_observer_exact_delta_and_install_before_first_runtime_import(wrapper, monkeypatch):
    events = []; admission = admitted(wrapper)
    original = (ROOT / 'spatial_probe_actual_preparation_v1/observed_cli.py').read_text()
    original_ast = ast.parse(original)
    original_counters = next(n for n in original_ast.body if isinstance(n, ast.ClassDef) and n.name == 'Counters')
    old = '    baseline = proc_io()  # before torch/runtime imports, including their actual syscalls'
    added = "    baseline = COMPAT_BOOT_IO\n    provenance = INSTALL_COMPAT()\n    write_once(Path(attempt) / 'SOURCE_COMPATIBILITY.json', provenance)"
    provenance = {'actual_source_shas_preserved': [c.OLD, c.NEW]}
    monkeypatch.setattr(c, 'install', lambda pin: events.append(('install', pin)) or provenance)
    class ImportBoundary(Exception): pass
    original_import = builtins.__import__
    def no_runtime_import(name, *args, **kwargs):
        if name == 'torch':
            frame = sys._getframe(1)
            assert frame.f_locals['baseline'] is wrapper.BOOT_IO
            assert frame.f_locals['read_cap'] == 321
            assert frame.f_locals['policy'] == admission['storage_policy']
            assert frame.f_locals['commitments'] == admission['checkpoint_commitments']
            events.append(('runtime_import_boundary', name))
            raise ImportBoundary()
        return original_import(name, *args, **kwargs)
    def observer(name, transform):
        assert name == 'observed_cli'
        transformed = transform(original)
        assert transformed.replace(added, old) == original
        parsed = ast.parse(transformed)
        counters = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Counters')
        assert ast.dump(counters) == ast.dump(original_counters)
        # Execute the actual transformed run function only up to first Torch
        # import; none of the original imports/runtime code are executed.
        fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        module = types.ModuleType('actual_transformed_observer_prefix')
        module.Path = Path; module.sys = sys
        module.write_once = lambda path, value: events.append(('provenance', str(path), value))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-observer-prefix>', 'exec'), module.__dict__)
        return module
    configure(wrapper, monkeypatch, admission, events, 'observe', observer)
    monkeypatch.setenv('ST_PROBE_SUPERVISOR_LOCK_FD', '77')
    original_fstat = wrapper.os.fstat
    monkeypatch.setattr(wrapper.os, 'fstat', lambda fd: events.append(('lock', fd)) if fd == 77 else original_fstat(fd))
    monkeypatch.setattr(wrapper.os, 'getpid', lambda: 123)
    monkeypatch.setattr(wrapper.os, 'getpgrp', lambda: 123)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(builtins, '__import__', no_runtime_import)
    with pytest.raises(ImportBoundary): wrapper.main()
    assert [event[0] for event in events] == [
        'verify', 'validate', 'lock', 'install', 'provenance', 'runtime_import_boundary']
    assert events[-2] == ('provenance', '/synthetic/attempt/SOURCE_COMPATIBILITY.json', provenance)


@pytest.mark.parametrize('fault', ['dirty', 'wrong_head'])
def test_proof_requires_exact_clean_runtime(monkeypatch, fault):
    def git(root, *args):
        if args == ('rev-parse', 'HEAD'):
            return ((c.OLD if fault == 'dirty' else '0' * 40) + '\n').encode()
        if args == ('status', '--porcelain'): return b' M foreign.py\n'
        pytest.fail('must reject before reading the tree')
    monkeypatch.setattr(c, 'git', git)
    with pytest.raises(ValueError, match='Exact clean original runtime'): c.evidence()


def test_install_rejects_foreign_package_without_mutation(monkeypatch):
    monkeypatch.delitem(sys.modules, c.TARGET, raising=False)
    monkeypatch.setattr(c, 'verify', lambda pin: {'schema': c.SCHEMA})
    monkeypatch.setattr(c.importlib, 'import_module', lambda name: types.SimpleNamespace(__file__='/foreign/tools/st_world_policy/__init__.py'))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    with pytest.raises(ValueError, match='Foreign policy package'): c.install({'path': 'synthetic', 'sha256': 'a'*64})
    assert c.TARGET not in sys.modules
