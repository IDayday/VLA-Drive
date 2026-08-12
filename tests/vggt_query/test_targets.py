import torch

from starVLA.model.modules.vggt_query.targets import (
    extract_vggt_layer11_memory_targets,
    extract_vggt_query_targets,
    extract_vggt_spatial_query_targets,
    select_vggt_global_teacher_layer,
)


def test_selects_pure_global_half_from_cached_layer_11():
    cached = [None] * 24
    frame = torch.full((2, 3, 21, 4), -3.0)
    global_features = torch.arange(2 * 3 * 21 * 4, dtype=torch.float32).reshape(
        2, 3, 21, 4
    )
    cached[11] = torch.cat((frame, global_features), dim=-1)

    selected = select_vggt_global_teacher_layer(
        cached,
        layer_index=11,
        branch_dim=4,
    )

    assert selected.shape == (2, 3, 21, 4)
    torch.testing.assert_close(selected, global_features)


def test_rejects_missing_or_malformed_global_teacher_layer():
    cached = [None] * 24
    try:
        select_vggt_global_teacher_layer(cached, layer_index=11, branch_dim=4)
    except RuntimeError as error:
        assert "layer 11" in str(error)
    else:
        raise AssertionError("missing cached layer must fail")

    cached[11] = torch.zeros(1, 3, 21, 7)
    try:
        select_vggt_global_teacher_layer(cached, layer_index=11, branch_dim=4)
    except AssertionError as error:
        assert "feature dim" in str(error)
    else:
        raise AssertionError("malformed frame/global concat must fail")


def test_layer11_production_target_is_180_pure_spatial_queries():
    # Three views, 5 special tokens, and a synthetic 4x4 source patch map.
    tokens = torch.arange(1 * 3 * 21 * 8, dtype=torch.float32).reshape(1, 3, 21, 8)
    validity = torch.ones(1, 3, 4, 4)

    features, valid_mask = extract_vggt_spatial_query_targets(
        tokens,
        spatial_validity=validity,
        patch_start_idx=5,
        patch_grid_size=4,
        output_size=(6, 10),
    )

    assert features.shape == (1, 180, 8)
    assert valid_mask.shape == (1, 180)
    assert valid_mask.all()


def test_layer11_memory_combines_15_global_and_180_spatial_queries():
    tokens = torch.arange(1 * 3 * 21 * 8, dtype=torch.float32).reshape(1, 3, 21, 8)
    validity = torch.ones(1, 3, 4, 4)

    features, valid_mask = extract_vggt_layer11_memory_targets(
        tokens,
        spatial_validity=validity,
        patch_start_idx=5,
        patch_grid_size=4,
        output_size=(6, 10),
    )

    assert features.shape == (1, 195, 8)
    assert valid_mask.shape == (1, 195)
    torch.testing.assert_close(features[:, :15], tokens[:, :, :5].reshape(1, 15, 8))
    assert valid_mask.all()


def test_extracts_special_and_spatial_targets_in_view_major_order():
    # [B=1, V=3, 5 special + 4x4 patches, D=4]
    tokens = torch.arange(1 * 3 * 21 * 4, dtype=torch.float32).reshape(1, 3, 21, 4)

    features, valid_mask = extract_vggt_query_targets(
        tokens,
        patch_start_idx=5,
        patch_grid_size=4,
        pooled_grid_size=2,
    )

    assert features.shape == (1, 27, 4)  # 3*5 special + 3*2*2 spatial
    assert valid_mask.shape == (1, 27)
    assert valid_mask.all()
    torch.testing.assert_close(features[:, :15], tokens[:, :, :5].reshape(1, 15, 4))

    first_view_patches = tokens[:, 0, 5:].reshape(1, 4, 4, 4)
    expected_first_cell = first_view_patches[:, :2, :2].mean(dim=(1, 2))
    torch.testing.assert_close(features[:, 15], expected_first_cell)


def test_rejects_an_incorrect_vggt_patch_layout():
    tokens = torch.zeros(1, 3, 20, 8)
    try:
        extract_vggt_query_targets(tokens, patch_start_idx=5, patch_grid_size=4, pooled_grid_size=2)
    except AssertionError as error:
        assert "token count" in str(error)
    else:
        raise AssertionError("invalid VGGT layout must fail")
