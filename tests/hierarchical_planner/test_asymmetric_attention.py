import torch

from starVLA.model.modules.trajectory_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.attention import AsymmetricDecoder


def test_asymmetric_attention_dimensions_and_independent_kv_parameters():
    decoder = AsymmetricDecoder(
        num_layers=3,
        query_dim=256,
        memory_dim=2048,
        num_heads=8,
        ffn_dim=1024,
        return_intermediate=True,
    )
    first, second = decoder.layers[:2]
    assert first.cross_attn.embed_dim == 256
    assert first.cross_attn.kdim == 2048
    assert first.cross_attn.vdim == 2048
    assert first.cross_attn.k_proj_weight is not second.cross_attn.k_proj_weight
    assert first.cross_attn.v_proj_weight is not second.cross_attn.v_proj_weight


def test_scorers_have_no_shared_scene_to_planning_projection():
    dynamic = DrivoRDynamicScorer(
        scene_dim=64,
        model_dim=32,
        ffn_dim=64,
        ego_state_dim=4,
        num_layers=2,
        num_heads=4,
    )
    coarse = DriveSuprimCoarseScorer(
        static_vocab=torch.randn(16, 40, 3),
        vocab_size=16,
        scene_dim=64,
        model_dim=32,
        ffn_dim=64,
        ego_state_dim=4,
        num_layers=2,
        num_heads=4,
        coarse_topk=4,
    )
    fine = DriveSuprimFineRefiner(
        scene_dim=64,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=4,
    )
    for module in (dynamic, coarse, fine):
        names = dict(module.named_modules())
        assert not any(
            name.endswith(("memory_proj", "scene_proj", "vision_proj"))
            for name in names
        )
        decoder_layers = [
            child
            for child in module.modules()
            if hasattr(child, "cross_attn") and hasattr(child, "memory_dim")
        ]
        assert decoder_layers
        assert all(layer.cross_attn.kdim == 64 for layer in decoder_layers)
        assert all(layer.cross_attn.vdim == 64 for layer in decoder_layers)


def test_fine_padding_mask_reaches_every_layer(monkeypatch):
    fine = DriveSuprimFineRefiner(
        scene_dim=64,
        model_dim=32,
        ffn_dim=64,
        num_layers=3,
        num_heads=4,
    )
    seen = []
    for layer in fine.fine_decoder.layers:
        original = layer.cross_attn.forward

        def wrapped(*args, _original=original, **kwargs):
            seen.append(kwargs.get("key_padding_mask"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(layer.cross_attn, "forward", wrapped)

    from starVLA.model.modules.trajectory_scorer import DriveSuprimCoarseScorer

    coarse_model = DriveSuprimCoarseScorer(
        static_vocab=torch.randn(16, 40, 3),
        vocab_size=16,
        scene_dim=64,
        model_dim=32,
        ffn_dim=64,
        ego_state_dim=4,
        num_layers=1,
        num_heads=4,
        coarse_topk=4,
    )
    coarse = coarse_model(torch.randn(2, 4, 64), torch.randn(2, 1, 4))
    mask = torch.tensor(
        [[False, False, False, True], [False, False, True, True]],
        dtype=torch.bool,
    )
    output = fine(coarse, torch.randn(2, 4, 64), mask)
    assert len(output.layer_metric_logits) == 3
    assert len(seen) == 3
    assert all(value is mask for value in seen)
