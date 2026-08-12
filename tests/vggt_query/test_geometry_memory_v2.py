import torch

from starVLA.model.modules.vggt_query.alignment import VGGTQueryAligner
from starVLA.model.modules.vggt_query.geometry_memory import (
    SharedGeometryAdapter,
    extract_qwen_spatial_memory,
)
from starVLA.model.modules.vggt_query.planning_heads import (
    AuxiliaryTrajectoryHead,
    PhysicalGeometryHead,
    WaypointGeometryReader,
)
from starVLA.model.modules.vggt_query.types import VGGTQueryLayout


def test_qwen_visual_hidden_resamples_to_180_spatial_slots():
    # Three 4x6 post-merge visual maps embedded in a language sequence.
    torch.manual_seed(11)
    batch, views, rows, cols, hidden = 2, 3, 4, 6, 16
    input_ids = torch.zeros(batch, 90, dtype=torch.long)
    last_hidden = torch.randn(batch, 90, hidden)
    image_token_id = 99
    for sample in range(batch):
        input_ids[sample, 5 : 5 + views * rows * cols] = image_token_id
    # Raw vision grid is twice the language-grid resolution (merge size 2).
    image_grid = torch.tensor([[[1, 8, 12]] * views] * batch)

    spatial, valid = extract_qwen_spatial_memory(
        last_hidden,
        input_ids=input_ids,
        image_grid_thw=image_grid,
        image_token_id=image_token_id,
        spatial_merge_size=2,
        view_count=views,
        output_size=(6, 10),
    )

    assert spatial.shape == (batch, 180, hidden)
    assert valid.shape == (batch, 180)
    assert valid.all()


def test_shared_adapter_reader_and_auxiliary_heads_have_v2_shapes():
    torch.manual_seed(13)
    layout = VGGTQueryLayout()
    global_queries = torch.randn(2, 15, 32, requires_grad=True)
    spatial_queries = torch.randn(2, 180, 32, requires_grad=True)
    adapter = SharedGeometryAdapter(input_dim=32, memory_dim=24)
    memory = adapter(global_queries, spatial_queries)
    assert memory.shape == (2, 195, 24)

    mask = torch.ones(2, 195, dtype=torch.bool)
    action_queries = torch.randn(2, 8, 32, requires_grad=True)
    reader = WaypointGeometryReader(
        action_dim=32,
        memory_dim=24,
        num_heads=4,
        layout=layout,
    )
    readout, diagnostics = reader(action_queries, memory, mask)
    assert readout.shape == (2, 8, 32)
    assert diagnostics["planner_attention"].shape == (2, 8, 195)
    assert diagnostics["attention_front_view_mass"].ndim == 0
    assert diagnostics["attention_waypoint_js_divergence"].ndim == 0

    geometry_head = PhysicalGeometryHead(memory_dim=24, hidden_dim=16)
    geometry_target = torch.randn(2, 180, 3)
    confidence = torch.rand(2, 180)
    geometry_mask = torch.ones(2, 180, dtype=torch.bool)
    geometry_output = geometry_head(
        memory[:, 15:], geometry_target, confidence, geometry_mask
    )
    assert geometry_output.prediction.shape == geometry_target.shape
    assert geometry_output.loss.ndim == 0

    aux_head = AuxiliaryTrajectoryHead(input_dim=32, hidden_dim=24, action_dim=4)
    target_action = torch.randn(2, 8, 4)
    aux_output = aux_head(readout, target_action)
    assert aux_output.prediction.shape == target_action.shape
    assert aux_output.loss.ndim == 0

    (geometry_output.loss + aux_output.loss).backward()
    assert global_queries.grad is not None
    assert spatial_queries.grad is not None


def test_v2_planning_modules_accept_deepspeed_bfloat16_parameters_without_autocast():
    """Inference diagnostics must work after DeepSpeed converts weights to BF16."""

    torch.manual_seed(17)
    layout = VGGTQueryLayout(
        view_count=1,
        special_per_view=1,
        spatial_rows=1,
        spatial_cols=2,
        teacher_dim=8,
    )
    global_queries = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    spatial_queries = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    adapter = SharedGeometryAdapter(input_dim=8, memory_dim=8).to(torch.bfloat16)
    memory = adapter(global_queries, spatial_queries)

    aligner = VGGTQueryAligner(
        student_dim=8,
        teacher_dim=8,
        special_query_count=1,
    ).to(torch.bfloat16)
    teacher = torch.randn(2, 3, 8)
    alignment_mask = torch.ones(2, 3, dtype=torch.bool)
    alignment_output = aligner(memory, teacher, alignment_mask)
    projected = alignment_output.projected_queries

    reader = WaypointGeometryReader(
        action_dim=8,
        memory_dim=8,
        num_heads=2,
        layout=layout,
    ).to(torch.bfloat16)
    action_queries = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    valid_mask = torch.ones(2, 3, dtype=torch.bool)
    readout, _ = reader(action_queries, projected, valid_mask)

    geometry_head = PhysicalGeometryHead(memory_dim=8, hidden_dim=8).to(torch.bfloat16)
    geometry_target = torch.randn(2, 2, 3)
    confidence = torch.ones(2, 2)
    geometry_mask = torch.ones(2, 2, dtype=torch.bool)
    geometry_output = geometry_head(
        projected[:, 1:], geometry_target, confidence, geometry_mask
    )

    auxiliary_head = AuxiliaryTrajectoryHead(
        input_dim=8, hidden_dim=8, action_dim=4
    ).to(torch.bfloat16)
    auxiliary_output = auxiliary_head(readout, torch.randn(2, 2, 4))

    (
        alignment_output.loss
        + geometry_output.loss
        + auxiliary_output.loss
    ).backward()

    assert memory.dtype == torch.bfloat16
    assert projected.dtype == torch.float32
    assert readout.dtype == torch.bfloat16
    assert geometry_output.prediction.dtype == torch.bfloat16
    assert auxiliary_output.prediction.dtype == torch.bfloat16
    assert adapter.adapter[0].weight.grad is not None
    assert aligner.student_projection.weight.grad is not None
    assert reader.cross_attention.out_proj.weight.grad is not None
