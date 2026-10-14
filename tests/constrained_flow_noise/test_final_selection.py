"""Guard against unsafe checkpoint selection and premature Navtest access."""
import json
import pytest

from scripts.analysis.constrained_flow_noise.lock_final import choose, make_lock, formal_run_paths
from starVLA.rl.constrained_flow_noise.contracts import fingerprint


def row(raw, score, safe):
    return dict(raw_exposure=raw, PDMS=score, safe_DEV_selection_eligible=safe)


def test_unsafe_high_score_never_wins_and_earlier_exact_tie():
    rows = [row(203184, 99., False), row(50796, 94., True), row(25398, 94., True)]
    assert choose(rows)['raw_exposure'] == 25398
    assert choose([row(25398, 99., False)]) is None


def test_pending_formal_training_cannot_write_navtest_lock(tmp_path):
    output = tmp_path/'final_lock.json'
    with pytest.raises(ValueError, match='Both registered formal lines'):
        make_lock(tmp_path, {}, dict(status='WAITING_FOR_FORMAL_ENDPOINTS'), output)
    assert not output.exists()


def test_native_continuation_requires_matching_lineage_and_complete_boundary(tmp_path):
    parent = tmp_path/'formal_C_SDE_seed42_v1'
    child = tmp_path/'formal_C_SDE_seed42_v1_probability_repair'
    checkpoint = tmp_path/'recovered_checkpoint'
    value = dict(qualification_only=False, variant='C_SDE')
    value['sha256'] = fingerprint(value)
    for directory, name in [(parent, 'run_identity.json'), (child, 'run_identity.json'),
                            (checkpoint, 'identity.json')]:
        directory.mkdir()
        (directory/name).write_text(json.dumps(value))
    (checkpoint/'COMPLETE').touch()
    (child/'native_continuation_origin.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), scientific_identity=value['sha256'],
        qualification_only=False, numerical_execution_identity='execution')))
    (child/'execution_identity.json').write_text(json.dumps(dict(sha256='execution')))
    pointer = tmp_path/'formal_C_SDE_active_continuation.json'
    pointer.write_text(json.dumps(dict(parent=str(parent), output=str(child), checkpoint=str(checkpoint))))
    assert formal_run_paths(tmp_path, 'C_SDE') == [parent, child]
    changed = dict(qualification_only=True, variant='C_SDE')
    changed['sha256'] = fingerprint(changed)
    (child/'run_identity.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='scientific lineage'):
        formal_run_paths(tmp_path, 'C_SDE')
    (child/'run_identity.json').write_text(json.dumps(value))
    (checkpoint/'COMPLETE').unlink()
    with pytest.raises(ValueError, match='Invalid formal continuation'):
        formal_run_paths(tmp_path, 'C_SDE')
