"""Actual A endpoint plus read-only negative fixtures; no policy execution."""
import copy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from scripts.analysis.fm_transfer import package_rl_endpoint as subject

ROOT = Path('/mnt/project/fm-transfer-20261016')
OUT = ROOT / 'qualification/endpoint_publication_20261017'
START = time.time()
CHECKS = []


def passed(name, **evidence):
    CHECKS.append(dict(check=name, status='PASS', **evidence))


def reject(name, suffix, edit):
    original = subject.read
    def changed(path):
        value = original(path)
        if str(path).endswith(suffix):
            value = copy.deepcopy(value)
            edit(value)
        return value
    output = OUT / name
    try:
        with patch.object(subject, 'read', side_effect=changed):
            subject.package(ROOT, 'A_DDP_TIME', output)
    except (ValueError, FileNotFoundError) as exc:
        assert not output.exists()
        passed(name, rejection=str(exc), real_input_files_unmodified=True)
    else:
        raise AssertionError('Invalid evidence accepted: ' + name)


def main():
    require_files = [ROOT/'analysis/A_DDP_TIME_RL_complete_training.json',
                     ROOT/'qualification/A_RL20k_analysis/completion.json',
                     ROOT/'analysis/A_DDP_TIME_RL_update020000_population.json',
                     ROOT/'analysis/A_DDP_TIME_RL_update020000_saved_batch.json']
    originals = {str(p):subject.sha(p) for p in require_files}
    copies, derived, receipt = subject.collect(ROOT, 'A_DDP_TIME')
    assert len(receipt['checks']) == 4
    passed('actual_complete_A_endpoint_validation', crosschecks=4)
    package = OUT/'A_package'
    receipt = subject.package(ROOT, 'A_DDP_TIME', package)
    for row in receipt['copied_analysis']:
        assert subject.sha(package/row['path']) == row['sha256']
    seal = subject.read(package/'COMPLETE.json')
    assert seal['execution_sha256'] == subject.sha(package/'execution.json')
    passed('actual_A_package_hashes_and_last_complete', files=len(receipt['copied_analysis']))
    population = subject.read(package/'A_RL_update020000_population.json')
    gradient = subject.read(package/'A_RL_update020000_saved_batch.json')
    assert 'checked_behavior_payloads' not in population
    assert population['checked_behavior_payload_count'] > 0
    assert 'verified_scene_scores' not in gradient['transfer_diagnostic_identity']
    assert gradient['transfer_diagnostic_identity']['verified_scene_score_count'] > 0
    passed('individual_private_payloads_omitted')
    existing = {str(p):subject.sha(p) for p in package.iterdir()}
    try:
        subject.package(ROOT, 'A_DDP_TIME', package)
    except FileExistsError:
        assert existing == {str(p):subject.sha(p) for p in package.iterdir()}
        passed('existing_output_rejected_without_overwrite')
    else:
        raise AssertionError('Existing output was overwritten')
    reject('reject_wrong_budget', 'A_DDP_TIME_RL_complete_training.json',
           lambda x:x['cumulative'].update(actual_global_optimizer_updates=15000))
    reject('reject_open_cost_ledger', 'A_DDP_TIME_RL_complete_training.json',
           lambda x:x['cost']['invocation_ledger'].update(incomplete_invocations=1))
    reject('reject_ready_instead_of_complete', 'update020000_inner0/COMPLETE',
           lambda x:x.update(phase='READY'))
    reject('reject_mixed_own_SFT', 'A_DDP_TIME_RL_update020000_DEV_paired_own_SFT.json',
           lambda x:x.update(reference='wrong own SFT'))
    reject('reject_inconsistent_gain', 'A_DDP_TIME_RL_update020000_DEV_paired_own_SFT.json',
           lambda x:x['modes']['E_NATIVE']['paired_comparison'].update(positive_gain_sum=0))
    reject('reject_mixed_rank_identity', 'A_DDP_TIME_RL_update020000_DEV/rank0.json',
           lambda x:x.update(identity_sha256='wrong evaluation'))
    b_output = OUT/'B_incomplete_must_not_publish'
    try:
        subject.package(ROOT, 'B_SOURCE_ACTION_DIT', b_output)
    except FileNotFoundError as exc:
        assert not b_output.exists()
        passed('actual_unfinished_B_rejected', missing_path=str(exc.filename))
    else:
        raise AssertionError('B completed during qualification; inspect actual new evidence')
    repo = Path(subject.__file__).resolve().parents[3]
    public_verifier = repo/'reports/ddp_simwam_fm_transfer_20261016/reproduce_B_RL20k_completion.py'
    assert subject.sha(public_verifier) == subject.sha(ROOT/'resources/verify_B_RL20k_completion.py')
    passed('already_published_B_completion_verifier_matches', sha256=subject.sha(public_verifier))
    assert originals == {str(p):subject.sha(p) for p in require_files}
    result = dict(status='PASS_ENDPOINT_PACKAGER_NOT_B_ENDPOINT_OR_NAVTEST_RESULT',
                  source_sha256=subject.sha(subject.__file__), qualification_sha256=subject.sha(__file__),
                  started_unix=START, ended_unix=time.time(), checks=CHECKS,
                  original_A_evidence_hashes=originals, new_candidates=0, new_scores=0,
                  optimizer_updates=0, actual_B_endpoint_complete=False,
                  note='Negative fixtures alter in-memory copies only; real training/evaluation artifacts preserved.')
    with (OUT/'validation.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'], checks=len(CHECKS))))


if __name__ == '__main__':
    main()
