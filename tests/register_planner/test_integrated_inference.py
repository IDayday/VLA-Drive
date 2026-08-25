import builtins
import inspect

import torch

from starVLA.model.framework.QwenRegisterPlanner import QwenRegisterPlanner
from starVLA.model.modules.register_planner.selectors import (
    DynamicDriveSuprimSelector,
    HybridDriveSuprimSelector,
)
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.drivesuprim_joint_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
)


def _drivor():
    return DrivoRDynamicScorer(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
        decoder_style="donor_register",
        proj_drop=0.0,
        drop_path=0.0,
    )


def _fine():
    return DriveSuprimFineRefiner(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_heads=1,
        num_layers=2,
    )


def _planner(tiny_factory, selector_type):
    proposal_num = 1 if selector_type == "none" else 4
    config = tiny_factory.config(proposal_num, selector_type)
    config.framework.name = "QwenRegisterPlanner"
    selector = None
    if selector_type == "drivor_suprim_dynamic":
        selector = DynamicDriveSuprimSelector(_fine())
    elif selector_type == "drivor_suprim_hybrid":
        coarse = DriveSuprimCoarseScorer(
            static_vocab=torch.zeros(16, 40, 3),
            vocab_size=16,
            scene_dim=32,
            model_dim=32,
            ffn_dim=64,
            num_heads=1,
            num_layers=2,
            coarse_topk=4,
        )
        selector = HybridDriveSuprimSelector(coarse, _fine())
    return QwenRegisterPlanner(
        config,
        qwen_vl_interface=tiny_factory.qwen(),
        qwen_hidden_extractor=tiny_factory.extractor,
        drivor_scorer=None if selector_type == "none" else _drivor(),
        suprim_selector=selector,
        load_checkpoints=False,
    )


def test_integrated_drivor_inference(tiny_factory):
    output = _planner(tiny_factory, "drivor")(
        tiny_factory.examples(with_action=False)
    )
    assert output["trajectory_navsim_8"].shape == (2, 8, 3)
    assert output["selected_index"].shape == (2,)
    assert output["drivor_score"].shape == (2,)


def test_integrated_suprim_dynamic_inference(tiny_factory):
    output = _planner(tiny_factory, "drivor_suprim_dynamic")(
        tiny_factory.examples(with_action=False)
    )
    assert output["trajectory_navsim_8"].shape == (2, 8, 3)
    assert output["suprim_score"].shape == (2,)
    assert torch.all(output["selected_source"] == 1)


def test_integrated_suprim_hybrid_inference(tiny_factory):
    output = _planner(tiny_factory, "drivor_suprim_hybrid")(
        tiny_factory.examples(with_action=False)
    )
    assert output["trajectory_navsim_8"].shape == (2, 8, 3)
    assert output["selected_source"].shape == (2,)
    assert output["all_proposals"].shape == (2, 4, 8, 3)


def test_inference_does_not_read_candidate_bank(tiny_factory, monkeypatch):
    planner = _planner(tiny_factory, "drivor")
    original = builtins.open

    def forbidden(*args, **kwargs):
        value = str(args[0]) if args else ""
        if "candidate" in value or "lmdb" in value:
            raise AssertionError("inference attempted to read a candidate bank")
        return original(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", forbidden)
    planner(tiny_factory.examples(with_action=False))
    source = inspect.getsource(QwenRegisterPlanner)
    assert "CandidateBank" not in source


def test_inference_does_not_read_metric_cache(tiny_factory):
    planner = _planner(tiny_factory, "drivor")
    planner(tiny_factory.examples(with_action=False))
    assert "DynamicMetricSupervisor" not in inspect.getsource(QwenRegisterPlanner)


def test_normalized_action_contract(tiny_factory):
    output = _planner(tiny_factory, "none")(
        tiny_factory.examples(with_action=False)
    )
    normalized = output["normalized_actions"]
    assert normalized.shape == (2, 8, 4)
    torch.testing.assert_close(
        normalized[..., 2:].square().sum(dim=-1),
        torch.ones(2, 8),
    )
