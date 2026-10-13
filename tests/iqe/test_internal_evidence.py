import pytest
from iqe.internal_baseline.audit_training import replay_consumption
from iqe.internal_baseline.sampling import TwoStreamSampler


def test_evidence_replays_consumed_streams_and_rejects_wrong_actual_ids():
    def new():
        return TwoStreamSampler(['a', 'b'], [.95, .02], natural_seed=2, comparison_seed=3,
                                pointwise_updates=1, bank_hash='locked')
    sampler = new()
    rows = []
    for step in range(2):
        rows.append({'sample_ids': [x['scene_id'] for x in sampler.peek(128)]})
        sampler.consume(128)
    counts = replay_consumption(new(), rows)
    assert sum(counts['natural'].values()) == 192
    assert sum(counts['comparison'].values()) == 64
    rows[1]['sample_ids'][0] = 'injected'
    with pytest.raises(ValueError, match='consumed sequence'):
        replay_consumption(new(), rows)
