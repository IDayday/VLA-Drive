"""Target isolation contract preventing privileged-future model inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ALLOWED_TOP_LEVEL_FIELDS = frozenset({"input", "target"})
FORBIDDEN_INPUT_COMPONENTS = frozenset(
    {
        "future_actor_states",
        "future_agent_states",
        "future_images",
        "future_bev",
        "future_dynamic_occupancy",
        "future_tracked_objects",
        "metric_cache",
        "replay_grounded_consequence",
        "log_replay",
        "reactive_model",
        "expert_future_trajectory",
        "structured_future_target",
        "consequence_target",
    }
)


def _mapping_paths(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = (*prefix, str(key))
            paths.append(path)
            paths.extend(_mapping_paths(child, path))
    return paths


def assert_no_future_leakage(model_input: Mapping[str, Any]) -> None:
    """Reject target-only future state anywhere in the model input tree."""

    violations: list[str] = []
    for path in _mapping_paths(model_input):
        leaf = path[-1].lower()
        if leaf in FORBIDDEN_INPUT_COMPONENTS or leaf.startswith("future_"):
            violations.append(".".join(path))
    if violations:
        raise AssertionError(f"privileged future fields found in model input: {violations}")


def isolate_sample(model_input: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Create the explicit research sample boundary and validate it eagerly."""

    copied_input = dict(model_input)
    copied_target = dict(target)
    assert_no_future_leakage(copied_input)
    return {"input": copied_input, "target": copied_target}


def collate_isolated_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Minimal dependency-free collate preserving the input/target boundary."""

    inputs: list[Any] = []
    targets: list[Any] = []
    for sample in samples:
        if set(sample) != ALLOWED_TOP_LEVEL_FIELDS:
            raise AssertionError(f"sample must contain exactly input/target, got {set(sample)}")
        model_input = sample["input"]
        if not isinstance(model_input, Mapping):
            raise TypeError("sample input must be a mapping")
        assert_no_future_leakage(model_input)
        inputs.append(model_input)
        targets.append(sample["target"])
    return {"input": inputs, "target": targets}
