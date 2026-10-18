"""Stat-only remaining-checkpoint reconciliation; no mutation or launch."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016')
PRIOR = Path('/mnt/project/VLA-Drive-st-flow-resume-efficiency-inheritance-20261018/reports/st_world_policy/20261018/checkpoint_remaining_reconciliation_v1')
RUNTIME = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
seen = set()
metadata_bytes = 0


def read(path, expected=None):
    global metadata_bytes
    path = Path(path)
    assert not path.is_symlink() and path.is_file() and path.stat().st_size < 2 << 20
    raw = path.read_bytes(); metadata_bytes += len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert expected is None or digest == expected
    return json.loads(raw), dict(path=str(path), sha256=digest)


def receipt(folder, *, identity, old_files=None, world_size=4):
    value, pin = read(folder / 'COMPLETE.json')
    body = {k:v for k,v in value.items() if k != 'receipt_identity'}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == value['receipt_identity']
    assert value['schema'] == 'st_native_checkpoint_v1' and value['world_size'] == world_size
    assert value['state']['identity'] == identity
    names = {'mp_rank_00_model_states.pt'} | {f'bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt' for i in range(world_size)} | {f'rng_rank{i}.pt' for i in range(world_size)}
    assert set(value['files']) == names == {p.name for p in folder.glob('*.pt')}
    checked = []
    for name, info in value['files'].items():
        p = folder / name
        s = p.lstat(); assert stat.S_ISREG(s.st_mode) and s.st_size == info['bytes']
        if old_files is not None:
            old = next(r for r in old_files if r['name'] == name)
            for k, current in [('device', s.st_dev), ('inode', s.st_ino), ('bytes', s.st_size),
                               ('mtime_ns', s.st_mtime_ns), ('ctime_ns', s.st_ctime_ns)]:
                assert old[k] == current, 'Previously credited file changed: ' + str(p)
            assert old['recorded_sha256'] == info['sha256']
        key = (s.st_dev, s.st_ino)
        credit = min(s.st_size, s.st_blocks * 512) if key not in seen else 0
        seen.add(key)
        checked.append(dict(name=name, device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
            allocated_bytes=s.st_blocks * 512, credited_bytes=credit,
            mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
            recorded_sha256=info['sha256'], payload_hash_recomputed=False))
    return dict(COMPLETE=pin, completed=value['state']['completed'], state=value['state'],
        credited_bytes=sum(x['credited_bytes'] for x in checked), files=checked)


def existing_allocation(root, budget, authority, *, expected_world, expected_endpoint):
    """An existing named allocation; completed does not release unused budget."""
    result, result_pin = read(root / 'RESULT.json')
    assert result['status'] in ('COMPLETE', 'PAUSED') and result['completed'] == expected_endpoint
    config, config_pin = read(root / 'resolved_config.json')
    assert config['execution']['world_size'] == expected_world
    with (root/'RUN.lock').open('rb') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN)
    items, partial = [], []
    for folder in sorted(p for p in (root/'checkpoints').iterdir() if p.is_dir()):
        if not (folder/'COMPLETE.json').is_file():
            partial.append(str(folder)); continue
        item=receipt(folder, identity=result['identity'], world_size=expected_world)
        assert 0 <= item['completed'] <= expected_endpoint
        items.append(item)
    credit=sum(x['credited_bytes'] for x in items)
    assert credit <= budget and max(x['completed'] for x in items) == expected_endpoint
    return dict(root=str(root), authority=authority, total_retained_allocation_bytes=budget,
        credited_bytes=credit, remaining_retention_bytes=budget-credit,
        result=result_pin, config=config_pin, checkpoints=items, uncredited_partial_directories=partial,
        unused_reservation_released=False)


def main():
    old, old_pin = read(PRIOR / 'CONSUMED_REMAINING_SNAPSHOT.json', '86dfb91df58672f3af483f57a08dd505f93a0eef2e0a83ae14119f169c3da0c7')
    cp, cp_pin = read(PRIOR / 'CHECKPOINT_COMMITMENTS_REMAINING_V2.json', '7b88fcbf0a4b4505aa251fdbf2e146c6b94dd3fcb15cef8269cee4d1989fa60c')
    assert cp['total_retained_allocation_bytes'] == 4143 << 30
    assert old['credited_completed_checkpoint_bytes'] == cp['credited_already_on_disk_bytes'] == 747345660520
    inherited = []
    for run in old['runs']:
        items = []
        for original in run['checkpoints']:
            path = Path(original['COMPLETE']['path'])
            rc, _ = read(path, original['COMPLETE']['sha256'])
            items.append(receipt(path.parent, identity=rc['state']['identity'], old_files=original['files']))
        inherited.append(dict(prior_run_identity={k:v for k,v in run.items() if k != 'checkpoints'}, checkpoints=items))
    original_credit = sum(x['credited_bytes'] for r in inherited for x in r['checkpoints'])
    assert original_credit == 747345660520
    new = []
    for cell in ('SP2', 'SP3'):
        registration_path = ROOT / 'formal/spatial/registrations' / f'{cell}_fulltrain_seed42_official_20k_v1_resource_v2/REGISTRATION.json'
        reg, reg_pin = read(registration_path)
        lease, lease_pin = read(reg['artifacts']['resource_lease']['path'], reg['artifacts']['resource_lease']['sha256'])
        assert lease['max_run_output_bytes'] == 720 << 30
        assert lease['checkpoint_slot_accounting'] == dict(checkpoint_slot_upper_bytes=40 << 30,
            extra_nonperiodic_or_failed_checkpoint_slots=2, in_flight_checkpoint_slots=1, regular_saves=15)
        run = ROOT / 'formal/spatial/runs' / reg['run_id']
        status, status_pin = read(run / 'status.json')
        result, result_pin = read(run / 'RESULT.json')
        # The large run identity includes the full source manifest. Do not reread
        # it for storage credit; native resume checks it before loading state.
        identity_pin = dict(path=str(run/'identity.json'), bytes=(run/'identity.json').stat().st_size,
            state_identity=status['identity'], full_identity_check='DEFERRED_TO_NATIVE_RESUME')
        assert status['status'] == result['status'] == 'PAUSED'
        assert status['termination_reason'] == result['termination_reason'] == 'BUDGET_PAUSED'
        assert status['completed'] == result['completed'] == (11762 if cell == 'SP2' else 11582)
        with (run / 'RUN.lock').open('rb') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN)
        directories = [p for p in (run / 'checkpoints').iterdir() if p.is_dir()]
        assert len(directories) <= 18
        complete, partial = [], []
        for folder in sorted(directories):
            if (folder / 'COMPLETE.json').is_file():
                c = receipt(folder, identity=status['identity'])
                assert 5000 < c['completed'] <= status['completed']
                assert c['credited_bytes'] < 40 << 30
                complete.append(c)
            else:
                partial.append(str(folder))
        periodic = [x for x in complete if x['completed'] % 1000 == 0]
        extra = [x for x in complete if x['completed'] % 1000 != 0]
        assert len(periodic) == 6 and len(extra) == 1 and not partial
        assert [x['completed'] for x in periodic] == list(range(6000, 12000, 1000))
        assert extra[0]['completed'] == status['completed']
        future_periodic = list(range(12000, 20001, 1000))
        unconsumed_extra = 2 - len(extra)
        max_directories = len(directories) + len(future_periodic) + unconsumed_extra + 1
        assert max_directories == 18
        credit = sum(x['credited_bytes'] for x in complete)
        new.append(dict(cell=cell, run_id=reg['run_id'], registration=reg_pin, original_lease=lease_pin,
            identity=identity_pin, status=status_pin, result=result_pin,
            completed=status['completed'], remaining_optimizer_updates=20000-status['completed'],
            checkpoints=complete, credited_bytes=credit, partial_directories=partial,
            original_retention_bytes=720 << 30, remaining_retention_bytes=(720 << 30)-credit,
            future_periodic_steps=future_periodic, unconsumed_nonperiodic_or_failed_slots=unconsumed_extra,
            in_flight_slots=1, maximum_retained_directories=max_directories, directory_cap=18,
            recipe_or_save_frequency_changed=False))
    added_credit = sum(r['credited_bytes'] for r in new)
    # Other checkpoints were always inside the original gross 2703 GiB but the
    # earlier minimal correction conservatively credited none of their bytes.
    # Establish each allocation from its old immutable authority before statting.
    basis, basis_pin=read(old['original_budget_basis']['path'],old['original_budget_basis']['sha256'])
    assert basis['reservation_basis']=='Retain all original 4*240 +187 +4*209 GiB commitments; do not subtract existing files'
    other=[]
    for c in ('C0','C1','C2','C3','C4'):
        run_id=f'{c}_fulltrain_seed42_official_4096_v1'
        original=next(x for x in basis['runs'] if x['run_id']==run_id)
        reg, reg_pin=read(ROOT/'formal/consequence/registrations'/run_id/'REGISTRATION.json',original['registration_sha256'])
        other.append(existing_allocation(Path(original['root']), (187 if c=='C0' else 209)<<30,
            dict(basis=basis_pin, registration=reg_pin), expected_world=4, expected_endpoint=4096))
    p2703, p2703_pin=read(ROOT/'resources/history32_actual_20261017T2345/CHECKPOINT_COMMITMENTS_2703G.json','015e7f6dddf7b1cab6823f88310db4297ed9ffd83653e9b6fab8cc666cc82856')
    boundary, boundary_pin=read(p2703['PCA256_boundary']['path'],p2703['PCA256_boundary']['sha256'])
    assert boundary['first_wave_per_run_native_space_reserve_bytes']==240<<30
    other.append(existing_allocation(ROOT/'formal/teacher_channel_control_v1/runs/SP2_PCA256_fulltrain_seed42_official_5k_v1',240<<30,
        dict(allocation=p2703_pin,boundary=boundary_pin),expected_world=4,expected_endpoint=5000))
    p2383,p2383_pin=read(p2703['parent_actual_2383']['path'],p2703['parent_actual_2383']['sha256'])
    p2303,p2303_pin=read(p2383['parent_2303GiB']['path'],p2383['parent_2303GiB']['sha256'])
    assert p2303['new_checkpoint_reservation_bytes']==120<<30
    t1root=Path(p2303['new_checkpoint_root'])
    assert t1root==ROOT/'smoke/official_history_T1_native4_v1/runs'
    t1=[]
    for n in (2,4):
        t1.append(existing_allocation(t1root/f'T1_official_real64_phase{n}_v1',120<<30,
            dict(shared_allocation=p2303_pin),expected_world=2,expected_endpoint=n))
    assert sum(x['credited_bytes'] for x in t1)<=120<<30
    other.append(dict(group='T1_original_shared120GiB',authority=p2303_pin,
        total_retained_allocation_bytes=120<<30,credited_bytes=sum(x['credited_bytes'] for x in t1),runs=t1))
    profile80,profile80_pin=read(p2383['additional_profile_80GiB']['path'],p2383['additional_profile_80GiB']['sha256'])
    assert profile80['new_reservation_bytes']==80<<30
    root64=Path(profile80['output_root'])/'runs/indexed_provider_fulltrain_updates64_world2_v1'
    other.append(existing_allocation(root64,80<<30,dict(allocation=profile80_pin),expected_world=2,expected_endpoint=64))
    assignment,assignment_pin=read(cp['parent_gross_receipt']['path'],cp['parent_gross_receipt']['sha256'])
    assign,assign_pin=read(assignment['actual_T3_hold80_assignment']['path'],assignment['actual_T3_hold80_assignment']['sha256'])
    assert assign['assigned_bytes']==80<<30 and assign['aggregate_bytes']==2703<<30
    other.append(existing_allocation(Path(assign['output_root'])/'runs'/assign['assigned_run_id'],80<<30,
        dict(allocation=assign_pin),expected_world=1,expected_endpoint=4))
    other_credit=sum(x['credited_bytes'] for x in other)
    total_credit = original_credit + added_credit + other_credit
    remaining = (4143 << 30) - total_credit
    now = time.time(); free = shutil.disk_usage(ROOT).free
    full_required = remaining + (1024 + 768 + 10) * (1 << 30)
    assert free >= full_required
    sys.path.insert(0, str(RUNTIME))
    from tools.st_world_policy.cache_storage_budget import StorageBudget
    lp = ROOT / 'resources/cache_storage_ledger_v1/CLAIMS.json'
    ledger, ledger_pin = read(lp)
    b = StorageBudget.__new__(StorageBudget); b.core = ledger['core']; b.state_dir = lp.parent
    b._check_disk(ledger, lp.parent, free_reserve_bytes=remaining + (1 << 40))
    snapshot = dict(schema='st_checkpoint_consumed_remaining_metadata_snapshot_v2', reviewer='/root/spatial20k_efficiency',
        checked_at_unix=now, previous_snapshot=old_pin, prior_remaining_receipt=cp_pin,
        status='VERIFIED_METADATA_AND_STAT_ONLY', inherited_runs=inherited, continuation_runs=new, other_original_allocations=other,
        old_credit_revalidated_bytes=original_credit, newly_credited_continuation_bytes=added_credit,
        newly_credited_other_original_allocations_bytes=other_credit,
        total_credited_bytes=total_credit, total_retained_allocation_bytes=4143 << 30,
        remaining_checkpoint_bytes=remaining, unique_payload_files=len(seen), metadata_bytes_read=metadata_bytes,
        tensor_payload_bytes_read=0, original_retention_limits_changed=False, files_deleted_or_modified=False,
        ledger_mutated=False, credit_rule='Existing complete native checkpoints inside original declared allocations only; min(logical, allocated), globally unique (device,inode); same old stat identity rechecked, no payload read.')
    s = HERE / 'CONSUMED_REMAINING_SNAPSHOT.json'; s.write_text(json.dumps(snapshot,indent=2)+'\n')
    sp = dict(path=str(s), sha256=hashlib.sha256(s.read_bytes()).hexdigest())
    updated = dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT', reviewer='/root/spatial20k_efficiency',
        checked_at_unix=now, valid_until_unix=now+86400,
        parent_gross_receipt=cp['parent_gross_receipt'], parent_remaining_receipt=cp_pin,
        total_retained_allocation_bytes=4143 << 30, credited_already_on_disk_bytes=total_credit,
        remaining_bytes=remaining, credited_checkpoint_snapshot=sp, free_floor_bytes=1024 << 30,
        whole_cache_future_reserve_bytes=768 << 30, separate_metadata_reserve_bytes=10 << 30,
        free_bytes_observed=free, required_free_bytes=full_required, headroom_bytes=free-full_required,
        proposed_one_80GiB_capacity_profile_headroom_bytes=free-full_required-(80 << 30),
        native_disk_guard='PASS_WITH_ACTUAL_LEDGER', actual_ledger_snapshot=ledger_pin,
        commitment_release_bytes=0, original_total_retention_limit_changed=False,
        all_old_completed_files_remain_retained=True, files_deleted_or_modified=False, ledger_mutated=False,
        quota_verified=False, formal_launch_approved=False,
        scope='Same SP2/SP3 20k identities: revalidate old credit and add actual complete periodics6000..11000 plus pause and already allocated C/PCA/T1/T3/full64 checkpoint bytes; no cache credit or original commitment release.')
    p=HERE/'CHECKPOINT_COMMITMENTS_REMAINING_V3.json';p.write_text(json.dumps(updated,indent=2)+'\n')
    print(json.dumps({'snapshot':sp,'remaining_receipt':dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        'old_credit':original_credit,'additional_credit':added_credit,'remaining':remaining,
        'free':free,'required':full_required,'headroom':free-full_required,'unique_payload_files':len(seen)},indent=2))


if __name__ == '__main__': main()
