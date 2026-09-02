from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.cf_effect_gate_wote.src.effect_tokenizer import MODEL_VARIANTS
from research.cf_effect_gate_wote.src.models.matched_hybrid_oracle_effect_probe import (
    CHECKPOINT_SCHEMA,
    MatchedHybridOracleEffectProbe,
    checkpoint_payload,
    load_matched_v3_checkpoint,
    trainable_parameter_count,
)
from research.cf_effect_gate_wote.src.models.top_aware_direct_scorer import (
    TopAwareDirectScorerConfig,
    TopAwareDirectScorerV3,
    checkpoint_payload as direct_checkpoint_payload,
)


def _direct_checkpoint(path: Path, seed: int = 0) -> TopAwareDirectScorerV3:
    torch.manual_seed(17 + seed)
    model = TopAwareDirectScorerV3(
        TopAwareDirectScorerConfig(representation="hybrid_current")
    ).eval()
    payload = direct_checkpoint_payload(
        model,
        seed=seed,
        objective={"objective": "O0"},
        selection={"safety_lambda": 0.5},
        metadata={"unit_test": True},
    )
    torch.save(payload, path)
    return model


def _inputs(candidates: int = 5) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(23)
    return (
        torch.randn(1, candidates, 8, 3, generator=generator),
        torch.randn(1, 8, generator=generator),
        torch.randn(1, 64, 256, generator=generator),
        torch.randn(1, candidates, 256, generator=generator),
        torch.randn(1, candidates, 32, 64, generator=generator),
    )


def test_matched_initialization_exactly_reproduces_direct_logits(tmp_path: Path) -> None:
    direct_path = tmp_path / "hybrid_current-seed0.pt"
    direct = _direct_checkpoint(direct_path)
    matched = MatchedHybridOracleEffectProbe().eval()
    matched.initialize_from_direct_checkpoint(direct_path)
    trajectory, ego, bev, candidate, auxiliary = _inputs()

    with torch.inference_mode():
        expected = direct(trajectory, ego, bev, candidate, candidate_chunk=64)
        actual = matched(trajectory, ego, bev, candidate, auxiliary)

    assert torch.equal(actual["factor_logits"], expected["factor_logits"])
    assert torch.equal(actual["factors"], expected["factors"])
    assert torch.equal(actual["score"], expected["factor_score"])
    assert tuple(actual["factors"].shape) == (1, 5, 6)


def test_all_variants_use_one_exact_parameter_count() -> None:
    counts = [
        trainable_parameter_count(MatchedHybridOracleEffectProbe())
        for _ in MODEL_VARIANTS
    ]
    assert len(set(counts)) == 1
    assert counts[0] > 0


def test_matched_checkpoint_requires_direct_initialization(tmp_path: Path) -> None:
    model = MatchedHybridOracleEffectProbe()
    with pytest.raises(ValueError, match="not initialized"):
        checkpoint_payload(
            model,
            model_type="direct_current",
            seed=0,
            metadata={},
        )

    direct_path = tmp_path / "hybrid_current-seed0.pt"
    _direct_checkpoint(direct_path)
    model.initialize_from_direct_checkpoint(direct_path)
    path = tmp_path / "matched.pt"
    torch.save(
        checkpoint_payload(
            model,
            model_type="direct_current",
            seed=0,
            metadata={"unit_test": True},
        ),
        path,
    )
    restored, payload = load_matched_v3_checkpoint(path)
    assert payload["schema_version"] == CHECKPOINT_SCHEMA
    assert trainable_parameter_count(restored) == trainable_parameter_count(model)


def test_legacy_structured_checkpoint_is_rejected_by_matched_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": "structured_six_factor_probe.v2"}, path)
    with pytest.raises(ValueError, match="refuses schema"):
        load_matched_v3_checkpoint(path)
