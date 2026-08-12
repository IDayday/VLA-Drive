import torch

from starVLA.model.modules.vggt_query.alignment import VGGTQueryAligner
from starVLA.model.modules.vggt_query.planner_bridge import PlanningQueryBridge


def test_alignment_loss_is_masked_and_has_student_gradient():
    torch.manual_seed(3)
    student = torch.randn(2, 7, 8, requires_grad=True)
    teacher = torch.randn(2, 7, 6)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[0, -2:] = False
    aligner = VGGTQueryAligner(
        student_dim=8,
        teacher_dim=6,
        special_query_count=3,
        cosine_weight=1.0,
        smooth_l1_weight=0.1,
        relational_weight=0.05,
    )

    output = aligner(student, teacher, mask)
    assert output.loss.ndim == 0
    assert set(output.losses) == {
        "cosine",
        "smooth_l1",
        "relational",
        "global",
        "spatial",
        "scene_relation",
    }
    assert "cosine_special" in output.metrics
    assert "cosine_spatial" in output.metrics
    assert "distributed_retrieval_top1" in output.metrics
    assert "distributed_retrieval_top5" in output.metrics
    output.loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_planner_bridge_exposes_action_conditioned_geometry_and_diagnostics():
    torch.manual_seed(7)
    action_queries = torch.randn(2, 8, 16, requires_grad=True)
    geometry_queries = torch.randn(2, 27, 16, requires_grad=True)
    valid_mask = torch.ones(2, 27, dtype=torch.bool)
    valid_mask[:, -1] = False
    bridge = PlanningQueryBridge(hidden_dim=16, num_heads=4, initial_gate=0.5)

    enhanced, diagnostics = bridge(action_queries, geometry_queries, valid_mask)

    assert enhanced.shape == action_queries.shape
    assert diagnostics["planner_bridge_attention"].shape == (2, 8, 27)
    assert 0.0 < diagnostics["planner_bridge_gate"].item() < 1.0
    assert diagnostics["planner_bridge_delta_norm"].item() > 0.0
    enhanced.square().mean().backward()
    assert geometry_queries.grad is not None
    assert geometry_queries.grad.abs().sum() > 0
