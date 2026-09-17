"""CPU integration tests on local official NAVSIM metric caches."""
from pathlib import Path
import time
from concurrent.futures import TimeoutError
import numpy as np
import pytest
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.reward import (
    RewardService,
    OfficialEvaluator,
    build_cache_index,
)
from starVLA.rl.flow_grpo.evaluation import original_protocol_scores


@pytest.fixture(scope="module")
def real_scene():
    cfg, sft = resolve_config("configs/flow_grpo/full_sft.yaml")
    train, _ = split_tokens(cfg)
    token = train[0]
    evaluator = OfficialEvaluator(
        Path("navsim").resolve(), build_cache_index(cfg["paths"]["metric_cache"])
    )
    cache = evaluator.load_cache(token)
    return cfg, token, cache.human_trajectory.poses


def test_adapter_matches_original_single_scene_evaluator(real_scene, tmp_path):
    cfg, token, traj = real_scene
    physical = np.stack(
        [
            traj,
            traj * 0,
            np.stack(
                [np.linspace(0, 4, 8), np.linspace(3, 24, 8), np.full(8, np.pi / 2)],
                axis=-1,
            ),
        ]
    )[None]
    reference = original_protocol_scores(
        [token], [traj], Path("navsim").resolve(), cfg["paths"]["metric_cache"]
    ).iloc[0]["score"]
    results = []
    for workers in [0, 2]:
        service = RewardService(
            Path("navsim").resolve(),
            cfg["paths"]["metric_cache"],
            "train",
            [token],
            workers=workers,
            cache_dir=tmp_path / f"cache{workers}",
        )
        try:
            rows = service.score([token], physical)[0]
            assert rows[0].score == reference
            assert rows[2].score == 0 and rows[2].status == "valid"
            assert [
                r.score for r in service.score([token], physical[:, ::-1].copy())[0]
            ][::-1] == [r.score for r in rows]
            assert service.score([token], physical[:, :1])[0][0].score == reference
            results.append([r.score for r in rows])
        finally:
            service.close()
    assert results[0] == results[1]


def test_missing_cache_nan_and_worker_exception_fail(real_scene):
    cfg, token, traj = real_scene
    with pytest.raises(FileNotFoundError):
        RewardService(
            "navsim", cfg["paths"]["metric_cache"], "train", ["missing"], workers=0
        )
    service = RewardService(
        Path("navsim").resolve(),
        cfg["paths"]["metric_cache"],
        "train",
        [token],
        workers=1,
    )
    with pytest.raises(ValueError):
        service.score([token], np.full((1, 1, 8, 3), np.nan))
    assert service.errors == 1 and service.pool is None


def test_spawn_worker_timeout_is_bounded(real_scene):
    cfg, token, traj = real_scene
    service = RewardService(
        Path("navsim").resolve(),
        cfg["paths"]["metric_cache"],
        "train",
        [token],
        workers=1,
        timeout=0.1,
    )
    service.pool.submit(time.sleep, 10)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        service.score([token], traj[None, None])
    assert time.monotonic() - started < 5 and service.pool is None


def test_train_test_separation_and_normalization(real_scene):
    cfg, token, traj = real_scene
    with pytest.raises(ValueError):
        RewardService(
            "navsim", cfg["paths"]["metric_cache"], "test", [token], workers=0
        )
    from infer import deal_action_1225

    norm = np.stack(
        [
            (traj[:, 0] - 10.172484) / 8.805105,
            (traj[:, 1] - 0.360762) / 2.277741,
            np.sin(traj[:, 2]),
            np.cos(traj[:, 2]),
        ],
        axis=-1,
    )
    recovered = deal_action_1225(norm[None], act_norm=1)[0]
    np.testing.assert_allclose(recovered[:, :2], traj[:, :2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.sin(recovered[:, 2]), np.sin(traj[:, 2]), rtol=1e-12, atol=1e-12
    )
