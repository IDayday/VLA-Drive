from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from starVLA.candidate_bank import (
    CandidateBankBuildIdentity,
    CandidateBankReader,
    CandidateBankWriter,
    finalize_candidate_bank,
    prepare_candidate_bank_root,
)
from starVLA.candidate_bank.schema import CANDIDATE_METRICS
from starVLA.model.modules.register_planner.outputs import RegisterGeneratorOutput
from starVLA.model.modules.scene_encoder import SceneContext
from starVLA.training.build_register_candidate_bank import (
    CandidateBankReport,
    score_and_write_candidate_batch,
)
from starVLA.training import build_register_candidate_bank
from starVLA.training.navsim_metric_supervisor import (
    DynamicMetricSupervisor,
    _OUTPUT_METRICS,
)


def _record(token="token", proposal_num=4, dense=False):
    value = {
        "token": token,
        "ego_state": torch.zeros(4),
        "scene_global_tokens": torch.zeros(4, 32, dtype=torch.float16),
        "proposals": torch.zeros(proposal_num, 8, 3, dtype=torch.float16),
        "gt_trajectory": torch.zeros(8, 3),
        "metrics": {
            name: torch.arange(proposal_num, dtype=torch.float32)
            for name in CANDIDATE_METRICS
        },
    }
    if dense:
        value["scene_dense_memory"] = torch.zeros(7, 32, dtype=torch.float16)
        value["attention_mask"] = torch.ones(7, dtype=torch.bool)
    return value


def _identity(proposal_num=4, dense=False):
    return CandidateBankBuildIdentity(
        split="train",
        world_size=1,
        proposal_num=proposal_num,
        generator_checkpoint_sha256="checkpoint-hash",
        generator_config_hash="config-hash",
        repository_commit="commit",
        metric_cache_root="metric-cache",
        datalist_path="train.pkl",
        scene_dim=32,
        scene_queries=4,
        include_dense_memory=dense,
    )


def _prepare(root, proposal_num=4, dense=False, **kwargs):
    identity = _identity(proposal_num, dense)
    digest = prepare_candidate_bank_root(root, identity=identity, **kwargs)
    return identity, digest


def _finalize(root, identity_hash, proposal_num=4, dense=False):
    return finalize_candidate_bank(
        root,
        manifest_fields={
            "split": "train",
            "proposal_num": proposal_num,
            "generator_checkpoint_sha256": "checkpoint-hash",
            "generator_config_hash": "config-hash",
            "repository_commit": "commit",
            "metric_cache_root": "metric-cache",
            "scene_dim": 32,
            "scene_queries": 4,
            "include_dense_memory": dense,
        },
        world_size=1,
        expected_build_identity_hash=identity_hash,
    )


def test_bank_roundtrip(tmp_path):
    tmp_path = tmp_path / "train"
    _, identity_hash = _prepare(tmp_path)
    with CandidateBankWriter(
        tmp_path,
        rank=0,
        proposal_num=4,
        scene_queries=4,
        scene_dim=32,
        expected_build_identity_hash=identity_hash,
    ) as writer:
        writer.put(_record())
    _finalize(tmp_path, identity_hash)
    reader = CandidateBankReader(tmp_path)
    torch.testing.assert_close(reader.get("token")["proposals"], _record()["proposals"])
    reader.close()


def test_bank_manifest_checkpoint_hash(tmp_path):
    tmp_path = tmp_path / "train"
    _, identity_hash = _prepare(tmp_path)
    with CandidateBankWriter(
        tmp_path,
        rank=0,
        proposal_num=4,
        scene_queries=4,
        scene_dim=32,
        expected_build_identity_hash=identity_hash,
    ) as writer:
        writer.put(_record())
    manifest = _finalize(tmp_path, identity_hash)
    assert manifest.generator_checkpoint_sha256 == "checkpoint-hash"


def test_bank_contains_64_proposals(tmp_path):
    tmp_path = tmp_path / "train"
    _, identity_hash = _prepare(tmp_path, proposal_num=64)
    with CandidateBankWriter(
        tmp_path,
        rank=0,
        proposal_num=64,
        scene_queries=4,
        scene_dim=32,
        expected_build_identity_hash=identity_hash,
    ) as writer:
        writer.put(_record(proposal_num=64))
    _finalize(tmp_path, identity_hash, proposal_num=64)
    reader = CandidateBankReader(tmp_path)
    assert reader.get("token")["proposals"].shape == (64, 8, 3)


def test_bank_metric_shapes():
    assert all(value.shape == (4,) for value in _record()["metrics"].values())


class SpySupervisor:
    def __init__(self):
        self.shapes = []

    def score(self, tokens, proposals):
        self.shapes.append(tuple(proposals.shape))
        base = torch.zeros(proposals.shape[:2], device=proposals.device)
        return {name: base.clone() for name in CANDIDATE_METRICS}


def test_bank_scores_full_pool_once(tmp_path):
    proposals = torch.zeros(2, 64, 8, 3)
    generated = RegisterGeneratorOutput(
        proposals=proposals,
        proposal_list=[proposals],
        final_tokens=torch.zeros(2, 64, 32),
        token_list=[torch.zeros(2, 64, 32)],
    )
    scene = SceneContext(
        global_tokens=torch.zeros(2, 4, 32),
        dense_memory=torch.zeros(2, 7, 32),
        memory_key_padding_mask=torch.zeros(2, 7, dtype=torch.bool),
    )
    examples = []
    for index in range(2):
        action = np.zeros((8, 4), dtype=np.float32)
        action[:, 3] = 1
        examples.append({"token": f"t{index}", "action": action})
    spy = SpySupervisor()
    report = CandidateBankReport(64)
    with CandidateBankWriter(
        tmp_path, rank=0, proposal_num=64, scene_queries=4, scene_dim=32
    ) as writer:
        score_and_write_candidate_batch(
            examples=examples,
            scene=scene,
            generator_output=generated,
            ego_state=torch.zeros(2, 4),
            metric_supervisor=spy,
            writer=writer,
            report=report,
        )
    assert spy.shapes == [(2, 64, 8, 3)]


def test_native_40_metric_pool_is_not_resampled():
    supervisor = object.__new__(DynamicMetricSupervisor)
    supervisor.backend = "local"
    supervisor._executor = None
    captured = []

    def score_one(_token, trajectories_40):
        captured.append(trajectories_40.copy())
        return {
            name: np.zeros(trajectories_40.shape[0], dtype=np.float32)
            for name in _OUTPUT_METRICS
        }

    supervisor._score_one = score_one
    proposals = torch.arange(2 * 3 * 40 * 3, dtype=torch.float32).reshape(
        2, 3, 40, 3
    )
    result = supervisor.score_40(["a", "b"], proposals)

    assert all(value.shape == (2, 3) for value in result.values())
    np.testing.assert_array_equal(captured[0], proposals[0].numpy())
    np.testing.assert_array_equal(captured[1], proposals[1].numpy())


def test_candidate_bank_deterministic_rebuild(tmp_path):
    roots = [tmp_path / "one" / "train", tmp_path / "two" / "train"]
    for root in roots:
        _, identity_hash = _prepare(root)
        with CandidateBankWriter(
            root,
            rank=0,
            proposal_num=4,
            scene_queries=4,
            scene_dim=32,
            expected_build_identity_hash=identity_hash,
        ) as writer:
            writer.put(_record())
        _finalize(root, identity_hash)
    first, second = CandidateBankReader(roots[0]), CandidateBankReader(roots[1])
    torch.testing.assert_close(first.get("token")["proposals"], second.get("token")["proposals"])
    first.close()
    second.close()


def test_bank_overwrite_removes_stale_rank_keys(tmp_path):
    root = tmp_path / "bank" / "train"
    identity, identity_hash = _prepare(root)
    with CandidateBankWriter(
        root,
        rank=0,
        proposal_num=4,
        scene_queries=4,
        scene_dim=32,
        expected_build_identity_hash=identity_hash,
    ) as writer:
        writer.put(_record("stale"))

    identity_hash = prepare_candidate_bank_root(
        root, identity=identity, overwrite=True
    )
    with CandidateBankWriter(
        root,
        rank=0,
        proposal_num=4,
        scene_queries=4,
        scene_dim=32,
        overwrite=True,
        expected_build_identity_hash=identity_hash,
    ) as writer:
        writer.put(_record("fresh"))
    _finalize(root, identity_hash)
    reader = CandidateBankReader(root)
    assert reader.tokens() == ("fresh",)
    reader.close()


def test_bank_resume_rejects_changed_build_identity(tmp_path):
    root = tmp_path / "bank" / "train"
    identity, _ = _prepare(root)
    changed = replace(identity, datalist_path="different-train.pkl")
    with pytest.raises(RuntimeError, match="resume identity mismatch"):
        prepare_candidate_bank_root(root, identity=changed, resume=True)


def test_bank_distributed_loader_does_not_pad_duplicate_scenes():
    source = inspect.getsource(build_register_candidate_bank)
    assert "DataLoaderConfiguration(even_batches=False)" in source
    assert "prepare_model(model, evaluation_mode=True)" in source
