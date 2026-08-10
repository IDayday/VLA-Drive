import torch
from torch import nn

from starVLA.model.modules.field2plan.geometry_field_writer import GeometryFieldWriter
from starVLA.model.modules.field2plan.types import CameraBatch


def _forward_facing_camera() -> CameraBatch:
    # camera_x=-ego_y, camera_y=-ego_z, camera_z=ego_x
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    transform = torch.eye(4)
    transform[:3, :3] = rotation
    intrinsics = torch.tensor(
        [[8.0, 0.0, 7.5], [0.0, 8.0, 7.5], [0.0, 0.0, 1.0]]
    )
    return CameraBatch(
        intrinsics=intrinsics.reshape(1, 1, 3, 3),
        ego_to_camera=transform.reshape(1, 1, 4, 4),
        image_hw=torch.tensor([[[16.0, 16.0]]]),
        view_names=("cam_f0",),
        frame_index=3,
    )


def test_geometry_writer_projects_valid_field_and_backpropagates() -> None:
    writer = GeometryFieldWriter(
        input_channels=4,
        output_channels=6,
        field_size=(8, 8),
        x_range_m=(1.0, 9.0),
        y_range_m=(-2.0, 2.0),
        height_anchors_m=(0.0, 1.0),
    )
    features = torch.randn(1, 1, 4, 16, 16, requires_grad=True)

    output = writer(features, _forward_facing_camera())

    assert output.field.shape == (1, 6, 8, 8)
    assert output.projection_valid.shape == (1, 1, 2, 8, 8)
    assert 0.0 < output.valid_ratio.item() <= 1.0
    output.field.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_geometry_writer_avoids_accelerator_specific_convolution_backward() -> None:
    """The trainable field projection must work through GEMM-only kernels.

    The development PPU reproducibly faults in BF16 Conv2d dgrad for the real
    24x24 field although an isolated synthetic convolution succeeds.  A
    channel-last pointwise MLP is mathematically sufficient for the Phase-1
    writer and avoids that shape/data-dependent vendor kernel.
    """

    writer = GeometryFieldWriter(
        input_channels=8,
        output_channels=16,
        field_size=(24, 24),
        x_range_m=(-8.0, 56.0),
        y_range_m=(-32.0, 32.0),
        height_anchors_m=(0.0, 1.0, 2.0),
    )

    assert not any(isinstance(module, nn.Conv2d) for module in writer.modules())
