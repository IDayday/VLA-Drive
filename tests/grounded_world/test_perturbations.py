import torch

from starVLA.model.modules.grounded_world.perturbations import (
    build_consequence_perturbations,
)


def test_consequence_perturbations_are_deterministic_and_not_labels() -> None:
    trajectory = torch.zeros(2, 8, 3)
    trajectory[..., 0] = torch.linspace(1.0, 20.0, 8)
    first = build_consequence_perturbations(trajectory)
    second = build_consequence_perturbations(trajectory)
    assert first.physical.shape == (2, 11, 8, 3)
    assert torch.equal(first.physical, second.physical)
    assert len(set(first.source_names)) == 11
    assert not hasattr(first, "labels")
