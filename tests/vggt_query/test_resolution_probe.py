import torch

from starVLA.model.modules.vggt_query.resolution_probe import (
    NormalizedSlotStatistics,
    crop_and_pool_valid_patches,
    summarize_scene_descriptors,
)


def test_crop_and_pool_removes_padding_before_rectangular_pooling():
    # Two padded rows surround a 2x4 content crop. Each scalar patch value is
    # duplicated across two channels so the expected pooling remains obvious.
    values = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [200.0, 200.0, 200.0, 200.0],
        ]
    )
    patches = values[None, None, :, :, None].repeat(1, 1, 1, 1, 2)
    validity = torch.zeros(1, 1, 4, 4)
    validity[:, :, 1:3] = 1.0

    pooled, mask = crop_and_pool_valid_patches(
        patches,
        validity,
        output_size=(1, 2),
    )

    assert pooled.shape == (1, 1, 1, 2, 2)
    assert mask.shape == (1, 1, 1, 2)
    assert mask.all()
    expected = torch.tensor([[[[[3.5, 3.5], [5.5, 5.5]]]]])
    torch.testing.assert_close(pooled, expected)


def test_normalized_slot_statistics_matches_a_brute_force_template_baseline():
    features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, -1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ]
    )
    mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, True],
        ]
    )
    stats = NormalizedSlotStatistics(slot_count=2, feature_dim=2)
    stats.update(features, mask)
    summary = stats.summary()

    unit = torch.nn.functional.normalize(features, dim=-1)
    template_cosines = []
    same_slot_pairs = []
    for slot in range(2):
        values = unit[mask[:, slot], slot]
        template = torch.nn.functional.normalize(values.mean(dim=0), dim=0)
        template_cosines.extend((values @ template).tolist())
        if len(values) > 1:
            similarities = values @ values.T
            off_diagonal = ~torch.eye(len(values), dtype=torch.bool)
            same_slot_pairs.extend(similarities[off_diagonal].tolist())

    assert summary["observations"] == int(mask.sum())
    torch.testing.assert_close(
        torch.tensor(summary["slot_template_cosine"]),
        torch.tensor(sum(template_cosines) / len(template_cosines)),
    )
    torch.testing.assert_close(
        torch.tensor(summary["same_slot_cross_scene_cosine"]),
        torch.tensor(sum(same_slot_pairs) / len(same_slot_pairs)),
    )


def test_scene_descriptor_summary_reports_cross_scene_margin():
    descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    summary = summarize_scene_descriptors(descriptors, chunk_size=2)

    expected_nearest = 2.0**-0.5
    expected_cross_mean = 4.0 * expected_nearest / 6.0
    torch.testing.assert_close(
        torch.tensor(summary["cross_scene_cosine_mean"]),
        torch.tensor(expected_cross_mean),
    )
    torch.testing.assert_close(
        torch.tensor(summary["nearest_other_cosine_mean"]),
        torch.tensor(expected_nearest),
    )
    torch.testing.assert_close(
        torch.tensor(summary["self_to_nearest_margin_mean"]),
        torch.tensor(1.0 - expected_nearest),
    )
