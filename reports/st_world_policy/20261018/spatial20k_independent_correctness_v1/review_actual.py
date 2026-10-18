"""Independent metadata-only review of materialized SP2/SP3 continuation.

No launch, model construction, torch.load, native payload hashing, or approval
assembly. The unchanged trainer owns strict serialized payload verification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
LEAD = Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
OWN = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.st_world_policy.check_formal_approval import verify_registration, file_hash, digest
from tools.st_world_policy.budget_extension import _recipe
from tools.st_world_policy.training_config import validate_config
from tools.st_world_policy.training_state import require_native_inventory


def ref(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': file_hash(p)}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True)
    args = p.parse_args()
    registration = Path(args.registration).resolve()
    reg = read(registration)
    arm = reg['run_id'].split('_', 1)[0]
    assert arm in ('SP2', 'SP3')
    assert reg['run_id'] == f'{arm}_fulltrain_seed42_official_20k_v1'
    assert '/root/spatial20k_correctness' not in reg['implementer_agents']
    prepdir = LEAD/'reports/st_world_policy/20261018/spatial20k_native_continuation_preparation_v1'
    prep = read(prepdir/'PREPARATION.json')
    entry = prep['arms'][arm]
    fp = verify_registration(registration, repo=REPO)
    cfg = read(reg['artifacts']['config']['path'])
    validate_config(cfg)
    assert cfg == read(entry['config']['path'])
    assert file_hash(entry['config']['path']) == entry['config']['sha256']
    parent_path = Path(cfg['continuation']['parent_registration']['path'])
    assert ref(parent_path) == entry['parent_registration'] == cfg['continuation']['parent_registration']
    parent = read(parent_path)
    pfp = verify_registration(parent_path, repo=REPO)
    pcfg = read(parent['artifacts']['config']['path'])
    assert _recipe(cfg) == _recipe(pcfg)
    assert parent['source_fingerprint'] == reg['source_fingerprint']
    assert reg['source_sha'] == prep['source_sha']
    assert cfg['execution'] == {**pcfg['execution'], 'stop_after': 20000}
    assert cfg['execution']['global_batch'] == 32
    assert cfg['execution']['micro_batch'] == 4
    assert cfg['execution']['world_size'] == 4
    assert pcfg['execution']['stop_after'] == 5000
    assert cfg['execution']['max_updates'] == pcfg['execution']['max_updates'] == 20000
    assert cfg['runtime']['max_seconds'] == 28800
    assert reg['implementer_agents'] == parent['implementer_agents']
    assert reg['weights_usable_for_formal_init'] is True
    for key in ('data_manifest', 'teacher_manifest', 'target_contract', 'base_binding', 'environment'):
        assert reg['artifacts'][key] == parent['artifacts'][key], key
    parent_run = Path(pcfg['runtime']['output_root']) / parent['run_id']
    identity = read(parent_run/'identity.json')
    assert digest({k:v for k,v in identity.items() if k != 'identity'}) == identity['identity']
    assert identity['config'] == pcfg
    assert identity['registration_sha256'] == file_hash(parent_path)
    assert identity['weights_usable_for_formal_init'] is True
    folder = parent_run/'checkpoints'/cfg['continuation']['parent_tag']
    receipt = read(folder/'COMPLETE.json')
    assert ref(folder/'COMPLETE.json') == entry['parent_receipt']
    assert file_hash(folder/'COMPLETE.json') == cfg['continuation']['parent_receipt_sha256']
    assert digest({k:v for k,v in receipt.items() if k != 'receipt_identity'}) == receipt['receipt_identity']
    assert receipt['state'] == entry['parent_state']
    assert receipt['state']['identity'] == identity['identity']
    assert receipt['state']['completed'] == 5000
    assert receipt['state']['termination_reason'] in ('RUNNING_CHECKPOINT', 'BUDGET_PAUSED')
    assert receipt['world_size'] == 4
    require_native_inventory(receipt['files'], 4)
    assert {x.name for x in folder.glob('*.pt')} == set(receipt['files'])
    for name, info in receipt['files'].items():
        assert (folder/name).stat().st_size == info['bytes']
    out = Path(cfg['runtime']['output_root'])/reg['run_id']
    assert out != parent_run and not out.exists()
    assert not (registration.parent/'LAUNCH.json').exists()
    lease = read(reg['artifacts']['resource_lease']['path'])
    assert lease['run_id'] == reg['run_id'] and lease['source_sha'] == reg['source_sha']
    assert lease['first_continuation_only'] and lease['no_automatic_resume']
    assert lease['max_invocation_seconds'] == 28800
    assert lease['maximum_optimizer_updates_this_stage'] == 15000
    assert lease['valid_from_unix'] <= time.time()
    assert lease['valid_until_unix'] >= time.time()+29580
    for host in lease['hostnames']:
        assert len(set(lease['cuda_visible_devices_by_host'][host].split(','))) == 4
        assert len(set(lease['gpu_uuids_by_host'][host])) == 4
    cpref = lease['checkpoint_commitments']
    assert ref(cpref['path']) == cpref
    cp = read(cpref['path'])
    assert cp['schema'] == 'st_checkpoint_storage_commitments_v1' and cp['status'] == 'CURRENT'
    assert cp['remaining_bytes'] == 4143*2**30 and cp['free_floor_bytes'] == 1024*2**30
    assert cp['checked_at_unix'] <= time.time() and cp['valid_until_unix'] >= time.time()+29580
    assert lease['remaining_cache_growth_bytes'] >= 0
    assert lease['separate_conservative_metadata_buffer_bytes'] >= 0
    actualenv = read(LEAD/'reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/ENVIRONMENTS.json')
    expectedenv = read(reg['artifacts']['environment']['path'])
    for host in lease['hostnames']:
        matching = [x for x in actualenv['observations'] if x['host'] == host.removesuffix('-worker-0')]
        assert len(matching) == 1 and matching[0]['environment'] == expectedenv
    launcher = LEAD/'reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/launch20k_once.py'
    assert file_hash(launcher) == 'e175da58b7e9f0b128c3059c13ab3a2dd7b0c65063daead80a6253165efe79e2'
    for source, expected in prep['source_pins'].items():
        assert file_hash(REPO/source) == expected
    oldreview = parent_path.parent/'reviews/CORRECTNESS.json'
    old = read(oldreview)
    assert old['status'] == 'PASS' and old['role'] == 'correctness' and old['fingerprint'] == pfp
    assert read(parent_path.parent/'RUN_APPROVAL.json')['reviews']['correctness'] == ref(oldreview)
    inherited = [
        Path('/mnt/project/st-world-policy-20261016/audits/correctness')/name
        for name in ('OFFICIAL_WORLD_LIVE_NATIVE_RESTORE_REVIEW.json',
                     'OFFICIAL_WORLD_RESUME_EXECUTION_REVIEW.json',
                     'OFFICIAL_FA2_AA_VARIABILITY.json',
                     'OFFICIAL_RUNTIME_BOUNDARY_FIX_REVIEW.json')]
    expectedrefs = {x['path']: x['sha256'] for x in old['evidence_artifacts']}
    for evidence in inherited:
        assert file_hash(evidence) == expectedrefs[str(evidence)]
    details = {
        'schema': 'st_spatial20k_actual_correctness_audit_v1', 'status': 'PASS',
        'reviewer': '/root/spatial20k_correctness', 'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'fingerprint': fp, 'parent_fingerprint': pfp, 'parent_state': receipt['state'],
        'config_exact_registered_proposal': True, 'recipe_unchanged_except_stop_and_native_parent': True,
        'max_schedule_updates': 20000, 'additional_updates_ceiling': 15000,
        'first_invocation_resume_flag': False, 'world_size': 4, 'global_batch': 32,
        'source_pins_unchanged': prep['source_pins'], 'native_state_load_evidence': 'INHERITED_REAL_FOUR_RANK',
        'current_parent_payload_inventory': 'NAMES_AND_SIZES_MATCH_RECEIPT',
        'native_checkpoint_payload_bytes_read': 0, 'new_model_or_teacher_runs': 0,
        'checkpoint_commitments': cpref, 'retained_and_new_commitments_gib': 4143,
        'launch_fd_argv_and_negative_test_source': ref(OWN/'test_launcher.py'),
        'launcher_cpu_tests': ref(OWN/'LAUNCHER_TESTS.log'), 'launcher': ref(launcher),
        'limitations': [
            'Metadata-only changed-scope review; real strict payload hashing/loading occurs once in the unchanged native trainer.',
            'Exact state restoration does not imply bitwise future gradients under production FA2.',
            'This review does not dispatch, grant other review roles, or approve another invocation/resume.',
            'Fresh actual GPU occupancy and free-space checks remain mandatory in the launcher.']}
    dest = OWN/arm
    dest.mkdir(exist_ok=False)
    (dest/'AUDIT.json').write_text(json.dumps(details, indent=2)+'\n')
    evidence = [ref(dest/'AUDIT.json'), ref(launcher), ref(oldreview), ref(OWN/'LAUNCHER_TESTS.log'),
                ref(OWN/'test_launcher.py'), ref(prepdir/'REGISTRATION_AND_APPROVAL.md')]
    evidence += [ref(x) for x in inherited]
    review = {'schema': 'st_world_review_v1', 'status': 'PASS', 'role': 'correctness',
              'reviewer_agent': '/root/spatial20k_correctness',
              'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'fingerprint': fp,
              'evidence_artifacts': evidence,
              'decision_scope': f'Only first native {arm} continuation from exact5000 to common20000, unchanged source, recipe and data.',
              'limits': details['limitations']}
    (dest/'CORRECTNESS.json').write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps({'status': 'CORRECTNESS_PASS_NOT_LAUNCHED', 'review': ref(dest/'CORRECTNESS.json')}))


if __name__ == '__main__':
    main()
