from __future__ import annotations

import inspect

import numpy as np

from research.cf_effect_gate_wote.src import (
    cache_wote_features,
    oracle_effect_data,
    train_probe,
)
from research.cf_effect_gate_wote.src.independent_label_store import (
    SIX_FACTOR_LABEL_SCHEMA_VERSION,
)
from research.cf_effect_gate_wote.src.feature_store import (
    CacheIdentity,
    FeatureShardWriter,
    SceneCacheRecord,
    stable_array_hash,
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


def test_probe_decodes_only_registered_effect_groups() -> None:
    assert oracle_effect_data._effect_keys_for_model("direct_current") == ()
    assert oracle_effect_data._effect_keys_for_model("wote_full_future") == ()
    assert set(oracle_effect_data._effect_keys_for_model("shared_logged_future")) == {
        "shared_actor_mask",
        "shared_logged_future",
    }
    assert "actor_engineered_effect" not in oracle_effect_data._effect_keys_for_model(
        "full_primitive_action_effect"
    )


def test_gate2o_writer_can_preserve_exact_float32_anchors(tmp_path) -> None:
    trajectory = np.linspace(-10, 10, 256 * 8 * 3, dtype=np.float32).reshape(
        1, 256, 8, 3
    )
    identity = CacheIdentity(
        run_id="gate2o-exact-anchors",
        split="smoke",
        checkpoint_sha256="a" * 64,
        wote_commit_sha="b" * 40,
        label_source="none",
        candidate_bank_hash="c" * 64,
    )
    writer = FeatureShardWriter(
        tmp_path / "cache", identity, float32_keys=("trajectory",)
    )
    writer.write_shard(
        0,
        {"trajectory": trajectory},
        (
            SceneCacheRecord(
                scene_token="scene",
                candidate_indices=tuple(range(256)),
                trajectory_hash=stable_array_hash(trajectory[0]),
                label_hash=None,
                candidate_bank_hash="c" * 64,
            ),
        ),
    )
    writer.finalize()
    with np.load(tmp_path / "cache" / "shard-00000.npz") as archive:
        assert archive["trajectory"].dtype == np.float32
        np.testing.assert_array_equal(archive["trajectory"], trajectory)
