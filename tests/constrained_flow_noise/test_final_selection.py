"""Guard against unsafe checkpoint selection and premature Navtest access."""
import pytest

from scripts.analysis.constrained_flow_noise.lock_final import choose, make_lock


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
