from types import SimpleNamespace

import pytest

from navsim.agents.EpisodeDrive.formal_initialization import (
    compare_formal_vlm_audits,
    scan_forbidden_state_keys,
    validate_formal_initialization_config,
)


def _initialization(variant="base"):
    return SimpleNamespace(
        mode="vlm_pretrained_random_planning_stack",
        variant=variant,
        prohibit_agent_checkpoint=True,
        shared_trainable_init_path="/tmp/shared.pt",
    )


def _vlm():
    return SimpleNamespace(
        vlm_path="/tmp/standalone-internvl",
        initialize_from_config=False,
    )


def test_formal_initialization_prohibits_agent_checkpoints():
    with pytest.raises(ValueError, match="prohibits checkpoint_path"):
        validate_formal_initialization_config(
            _initialization(),
            checkpoint_path="m0-agent.ckpt",
            stage1_checkpoint_path=None,
            vlm_config=_vlm(),
        )
    with pytest.raises(ValueError, match="prohibits stage1_checkpoint_path"):
        validate_formal_initialization_config(
            _initialization(),
            checkpoint_path=None,
            stage1_checkpoint_path="m0-stage1.ckpt",
            vlm_config=_vlm(),
        )


def test_formal_initialization_requires_pretrained_loading():
    vlm = _vlm()
    vlm.initialize_from_config = True
    with pytest.raises(ValueError, match="initialize_from_config=false"):
        validate_formal_initialization_config(
            _initialization(),
            checkpoint_path=None,
            stage1_checkpoint_path=None,
            vlm_config=vlm,
        )


def test_forbidden_agent_state_key_detection():
    findings = scan_forbidden_state_keys(
        [
            "vision_model.encoder.layers.0.attn.qkv.weight",
            "agent.action_head.scorer.weight",
            "optimizer_states",
            "future_register_predictor.residual_output.weight",
        ]
    )
    assert findings == (
        "agent.action_head.scorer.weight",
        "future_register_predictor.residual_output.weight",
        "optimizer_states",
    )


def _audit(vocab):
    return {
        "model_architectures": ["InternVLChatModel"],
        "vision_architectures": ["InternVisionModel"],
        "language_architectures": ["Qwen2ForCausalLM"],
        "vision_block_count": 24,
        "vision_hidden_size": 1024,
        "patch_size": 14,
        "image_size": 448,
        "llm_hidden_size": 1536,
        "vocab_size": len(vocab),
        "prompt_template": "internvl2_5",
        "token_id_map": vocab,
    }


def test_formal_pair_requires_exact_token_ids():
    base = _audit({"a": 0, "b": 1})
    same = _audit({"a": 0, "b": 1})
    result = compare_formal_vlm_audits(base, same)
    assert result["formal_pair_compatible"]

    changed = _audit({"a": 0, "b": 2})
    with pytest.raises(RuntimeError, match="Silent vocabulary resizing is prohibited"):
        compare_formal_vlm_audits(base, changed)
