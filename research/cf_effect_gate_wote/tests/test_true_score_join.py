from __future__ import annotations

import inspect

from research.cf_effect_gate_wote.src import (
    cache_wote_features,
    oracle_effect_data,
    train_probe,
)
from research.cf_effect_gate_wote.src.independent_label_store import (
    SIX_FACTOR_LABEL_SCHEMA_VERSION,
)


def test_raw_probe_batch_requires_stored_score_labels() -> None:
    assert "score_labels" in oracle_effect_data.RawProbeBatch.__annotations__
    source = inspect.getsource(oracle_effect_data._scene_batch_values)
    assert "scene.score_labels" in source
    assert "pdms_from_factors" not in source


def test_feature_cache_label_none_does_not_load_published_labels() -> None:
    assert cache_wote_features._score_dictionary_for_label_source.__name__
    assert cache_wote_features._score_dictionary_for_label_source.__annotations__
    assert cache_wote_features._score_dictionary_for_label_source(
        None, "none"  # type: ignore[arg-type]
    ) is None
    assert SIX_FACTOR_LABEL_SCHEMA_VERSION == "independent_wote_labels_4s_six_factor.v2"


def test_effect_builder_interface_has_no_score_or_label_input() -> None:
    from research.cf_effect_gate_wote.src.replay_effect_builder import (
        ReplayGroundedEffectBuilder,
    )

    parameters = tuple(inspect.signature(ReplayGroundedEffectBuilder.build).parameters)
    assert parameters == ("self", "candidates", "context")
    source = inspect.getsource(ReplayGroundedEffectBuilder.build).lower()
    assert "oracle_index" not in source
    assert "metric_cache.trajectory" not in source


def test_effect_cache_reads_only_base_trajectory_from_frozen_cache() -> None:
    source = inspect.getsource(train_probe.cache_replay_effects)
    assert 'iter_shards(("trajectory",))' in source
    for forbidden in (
        'arrays["final_rewards"]',
        'arrays["selected_index"]',
        'arrays["reward_feature"]',
        'arrays["factor_labels"]',
        'arrays["score_labels"]',
    ):
        assert forbidden not in source
