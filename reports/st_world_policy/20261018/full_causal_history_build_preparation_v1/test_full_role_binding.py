"""Small controller-boundary fixtures; no population, payload or live ledger."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_history_once as c
import storage_binding as s


def inputs(role):
    inherited = s.inherited_binding()
    plan = json.loads(c.PLAN_PATH.read_bytes())
    assert hashlib.sha256(c.PLAN_PATH.read_bytes()).hexdigest() == c.PLAN_SHA
    spec = plan['roles'][role]
    now = time.time()
    cp = dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT',
              remaining_bytes=2703 << 30, free_floor_bytes=1 << 40,
              checked_at_unix=now-1, valid_until_unix=now+20000)
    a = dict(schema='st_full_causal_history_owned_admission_v1',
        status='APPROVED_FULL_CAUSAL_HISTORY_METADATA_ONLY', role=role,
        reviewer='independent_fixture_not_real_admission', runtime=str(c.RUNTIME),
        source_sha=c.RUNTIME_COMMIT, output_root=spec['output_root'],
        control_root=spec['control_root'], control_reserved_bytes=1 << 20,
        hostname=socket.gethostname(), limits=c.limits(role, [10,11]),
        w_source_compatibility=inherited.W_COMPAT, registration=spec['registration'],
        builder_budget=spec['builder_budget'], issued_at_unix=now-1,
        latest_start_unix=now+1, valid_until_unix=now+19000,
        checkpoint_receipt={'path':'/fixture/not-read','sha256':'1'*64})
    return inherited, spec, a, cp, now


@pytest.mark.parametrize('role', ['train', 'dev'])
def test_exact_original_controller_binding(role):
    inherited, spec, a, cp, now = inputs(role)
    cpus = c.validate(a, cp, role, spec, inherited, now=now)
    raw = s.source_bytes(c.ORIGINAL, inherited.PINS[str(c.ORIGINAL.relative_to(c.RUNTIME))])
    m = c.controller(raw, a, cp, role, spec, inherited, cpus)
    assert m.RUNTIME == c.RUNTIME
    assert str(m.ROOT) == spec['output_root']
    assert str(m.CONTROL_ROOT) == spec['control_root']
    assert m.LIMITS == c.limits(role, cpus)
    m.validate(a, now)
    assert m.BOUND_SCENES == (101592 if role == 'train' else 1696)
    assert m.BOUND_SELECTION == spec['selection']
    assert m.ENTRY_IO is c.ENTRY_IO
    assert m.ENTRY_MONOTONIC == c.ENTRY
    assert m.BOUND_CP_REMAINING == 2703 << 30
    assert len(m.BOUND_EXTRA_SOURCES) == 3
    assert 'core.monitor(command' in c.bound_source(raw)
    assert "'--role',BOUND_ROLE,'--child'" in c.bound_source(raw)


@pytest.mark.parametrize('mutation', ['role','max_scenes','disk','wall','read','rss','runtime','author','source','registration','cp_expiry','lease'])
def test_binding_rejects_changed_contract(mutation):
    inherited, spec, a, cp, now = inputs('train')
    if mutation == 'role': a['role'] = 'dev'
    if mutation == 'max_scenes': a['registration'] = dict(a['registration'],sha256='f'*64)
    if mutation == 'disk': a['limits']['max_disk_bytes'] += 1
    if mutation == 'wall': a['limits']['wall_seconds'] += 1
    if mutation == 'read': a['limits']['read_bytes'] += 1
    if mutation == 'rss': a['limits']['rss_bytes'] += 1
    if mutation == 'runtime': a['runtime'] += '-other'
    if mutation == 'author': a['reviewer'] = '/root/st_indexprep_boundary_fix'
    if mutation == 'source': a['source_sha'] = 'f'*40
    if mutation == 'registration': a['builder_budget'] = dict(a['builder_budget'],sha256='f'*64)
    if mutation == 'cp_expiry': cp['valid_until_unix'] = now + 60
    if mutation == 'lease': a['valid_until_unix'] = now + 900
    with pytest.raises(ValueError): c.validate(a,cp,'train',spec,inherited,now=now)


def test_original_controller_source_drift_refused():
    inherited, *_ = inputs('dev')
    raw = s.source_bytes(c.ORIGINAL,inherited.PINS[str(c.ORIGINAL.relative_to(c.RUNTIME))])
    with pytest.raises(ValueError): c.bound_source(raw.replace(b'cpu_affinity=(6,7)',b'cpu_affinity=(1,2)'))


def test_selection_streaming_regular_identity_and_negative_paths(tmp_path):
    p=tmp_path/'list.json';raw=b'{"scenes":[]}'
    p.write_bytes(raw);pin=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
    assert c.selection_check(pin,len(raw))['verified_population_sha256']==pin['sha256']
    with pytest.raises(ValueError): c.selection_check(pin,len(raw)+1)
    with pytest.raises(ValueError): c.selection_check(dict(pin,sha256='f'*64),len(raw))
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError): c.selection_check(dict(pin,path=str(link)),len(raw))
    fifo=tmp_path/'fifo';os.mkfifo(fifo)
    start=time.monotonic()
    with pytest.raises(ValueError): c.selection_check(dict(pin,path=str(fifo)),0)
    assert time.monotonic()-start<1


def test_single_invocation_retains_no_retry_and_global_lock(tmp_path):
    import fcntl
    inherited,spec,a,cp,_=inputs('train')
    raw=s.source_bytes(c.ORIGINAL,inherited.PINS[str(c.ORIGINAL.relative_to(c.RUNTIME))])
    m=c.controller(raw,a,cp,'train',spec,inherited,[10,11])
    lock=tmp_path/'FULL_POPULATION.lock'
    first=os.open(lock,os.O_RDWR|os.O_CREAT,0o600)
    second=os.open(lock,os.O_RDWR)
    try:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m.control_census(tmp_path)
        m.control_once(tmp_path,'INVOCATION.json',{'fixture':True})
        with pytest.raises(FileExistsError): m.control_once(tmp_path,'INVOCATION.json',{'fixture':True})
        assert json.loads((tmp_path/'INVOCATION.json').read_text())=={'fixture':True}
    finally:
        os.close(second);os.close(first)


def test_append_only_policy_preserves_v3_core_and_all_old_claims():
    plan=json.loads(c.PLAN_PATH.read_bytes())
    old=json.loads((c.RUNTIME/'reports/st_world_policy/20261018/full_causal_history_build_preparation_v1/metadata_v3/PACKAGE_ONLY_POLICY_PROPOSAL.json').read_bytes())
    path=Path(plan['append_only_policy_proposal']['path']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==plan['append_only_policy_proposal']['sha256']
    new=json.loads(raw)
    assert new['claims'][:-2]==old['claims']
    assert {k:v for k,v in new.items() if k!='claims'}=={k:v for k,v in old.items() if k!='claims'}
    assert sum(sum(row[k] for k in ('output_bytes','staging_bytes','metadata_bytes')) for row in new['claims'][-2:])==(2176-2)<<20
    for role,spec in plan['roles'].items():
        assert spec['expected_registration']['max_scenes']==spec['scenes']
        assert spec['expected_registration']['cwd']==str(c.RUNTIME)
        assert all(p['path'].startswith(str(c.RUNTIME)+'/') for p in spec['expected_registration']['source_pins'])
