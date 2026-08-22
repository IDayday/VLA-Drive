import hashlib
from pathlib import Path

from scripts.audit_multitrajectory_config import (
    audit_configs,
    qwen_parameter_manifests,
)
from starVLA.training.config_loader import load_training_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "starVLA/config/training"
QWEN = ROOT / "models/Qwen3-VL-2B-WorldAction"
B1 = CONFIG / "qwenpi_matched_b1_action_only.yaml"
B2 = CONFIG / "qwenpi_matched_b2_qformer_single.yaml"
B3 = CONFIG / "qwenpi_matched_b3_drivor.yaml"
B4 = CONFIG / "qwenpi_matched_b4_full.yaml"


def _cfg(path=B4):
    return load_training_config(path)


def test_main_config_uses_baseline_dit_shape():
    config = _cfg()
    action = config.framework.action_model
    assert action.hidden_size == 1536
    assert action.diffusion_model_cfg.num_layers == 24
    assert action.diffusion_model_cfg.cross_attention_dim == 1536
    assert action.diffusion_model_cfg.output_dim == 1536
    assert 1536 // 64 == 24


def test_main_dit_hidden_size_is_1536():
    assert _cfg().framework.action_model.hidden_size == 1536


def test_main_dit_num_layers_is_24():
    assert _cfg().framework.action_model.diffusion_model_cfg.num_layers == 24


def test_flow_train_repeat_is_8():
    assert _cfg().framework.action_model.flow_train_repeats == 8


def test_action_horizon_remains_8():
    assert _cfg().framework.action_model.action_horizon == 8


def test_dynamic_candidate_count_is_independent_of_flow_repeat():
    config = _cfg()
    assert config.framework.action_model.flow_train_repeats == 8
    assert config.framework.hierarchical_scorer.dynamic.num_candidates == 64


def test_candidate_chunk_size_does_not_change_candidate_count():
    dynamic = _cfg().framework.hierarchical_scorer.dynamic
    assert dynamic.num_candidates == 64
    assert dynamic.candidate_chunk_size == 8
    assert dynamic.num_candidates // dynamic.candidate_chunk_size == 8


def test_main_qformer_contract():
    scene = _cfg().framework.scene_encoder
    assert scene.input_dim == 2048
    assert scene.hidden_dim == 256
    assert scene.output_dim == 256
    assert scene.num_queries == 16
    assert scene.num_layers == 4
    assert scene.num_heads == 8
    assert scene.ffn_dim == 1024
    assert scene.detach_qwen_input is True


def test_main_hierarchical_candidate_contract():
    scorer = _cfg().framework.hierarchical_scorer
    assert scorer.dynamic.num_candidates == 64
    assert scorer.dynamic.dynamic_topm == 32
    assert scorer.joint.vocab_size == 8192
    assert scorer.joint.coarse_topk == 256
    assert scorer.refinement.num_stages == 1
    assert scorer.refinement.num_layers == 3
    assert 8192 + 32 == 8224


def test_qwen_model_path_matches_baseline(monkeypatch):
    monkeypatch.setenv("QWEN_VLM_PATH", str(QWEN))
    configs = [_cfg(path) for path in (B1, B2, B3, B4)]
    assert len({str(config.framework.qwenvl.base_vlm) for config in configs}) == 1


def test_qwen_trainable_names_match_baseline(monkeypatch):
    monkeypatch.setenv("QWEN_VLM_PATH", str(QWEN))
    baseline, _ = qwen_parameter_manifests(_cfg(B1))
    method, _ = qwen_parameter_manifests(_cfg(B4))
    assert method == baseline
    assert len(method) == 309


def test_qwen_frozen_names_match_baseline(monkeypatch):
    monkeypatch.setenv("QWEN_VLM_PATH", str(QWEN))
    _, baseline = qwen_parameter_manifests(_cfg(B1))
    _, method = qwen_parameter_manifests(_cfg(B4))
    assert method == baseline
    assert len(method) == 316
    assert "model.model.language_model.embed_tokens.weight" in method
    assert sum(name.startswith("model.model.visual.") for name in method) == 315


def test_B1_B4_qwen_policy_equal(monkeypatch):
    monkeypatch.setenv("QWEN_VLM_PATH", str(QWEN))
    assert qwen_parameter_manifests(_cfg(B1)) == qwen_parameter_manifests(_cfg(B4))


def test_B1_B4_dit_shape_equal():
    first, fourth = _cfg(B1), _cfg(B4)
    assert first.framework.action_model.hidden_size == fourth.framework.action_model.hidden_size
    assert (
        first.framework.action_model.diffusion_model_cfg.num_layers
        == fourth.framework.action_model.diffusion_model_cfg.num_layers
    )


def test_B1_B4_flow_repeat_equal():
    assert (
        _cfg(B1).framework.action_model.flow_train_repeats
        == _cfg(B4).framework.action_model.flow_train_repeats
        == 8
    )


def test_B1_B4_auxiliary_flags_equal():
    first, fourth = _cfg(B1), _cfg(B4)
    for config in (first, fourth):
        assert config.datasets.video_data.load_2d_data == 0
        assert config.datasets.gs_data.load_3d_data == 0
        assert config.datasets.reward_data.load_reward_data == 0
        assert config.w_depth == 0
        assert config.rgb_query_loss == 0
        assert config.gs_query_loss == 0


def test_all_matched_configs_pass_audit(monkeypatch):
    monkeypatch.setenv("QWEN_VLM_PATH", str(QWEN))
    for method in (B1, B2, B3, B4):
        assert audit_configs(str(B1), str(method)) == []


def test_legacy_config_unchanged():
    legacy = CONFIG / "cfg_yaw_1225.yaml"
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    assert digest == "fd1ae397ba64de0c9a4fdf77b24a8474f6f54b2d9194124a2fe9c2e3ff97307e"
