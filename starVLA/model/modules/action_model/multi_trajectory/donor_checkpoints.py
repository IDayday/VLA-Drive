"""Audited warm-start conversion for official DrivoR/DriveSuprim weights.

Only semantically identical 256-dimensional embeddings and metric heads are
transferred.  The 2048-dimensional Q-Former and every asymmetric decoder
attention layer retain their target initialization and are explicitly listed
as requiring training; no old same-width memory projection is treated as
compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor


CONVERSION_FORMAT_VERSION = 2
DRIVOR_DONOR_PREFIX = "agent._drivor_model."
SUPRIM_TEACHER_PREFIX = "agent.model.teacher.model."
SUPRIM_STUDENT_PREFIX = "agent.model.student.model."


@dataclass(frozen=True)
class DonorConversionReport:
    component: str
    donor_repository: str
    donor_commit: str
    transferred_keys: Tuple[str, ...]
    requires_training: Tuple[str, ...]
    excluded_donor_prefixes: Tuple[str, ...]
    inference_ready: bool
    scene_dim: int = 2048
    planning_dim: int = 256
    format_version: int = CONVERSION_FORMAT_VERSION

    def as_metadata(self) -> Dict[str, object]:
        return asdict(self)


def unwrap_tensor_state(checkpoint: object) -> Mapping[str, Tensor]:
    value = checkpoint
    if isinstance(value, Mapping) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, Mapping):
        raise TypeError("donor checkpoint must be a mapping or contain state_dict")
    if not all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    ):
        raise TypeError("donor state_dict must contain string-to-tensor entries")
    return value


def _strip_required_prefix(
    state: Mapping[str, Tensor], prefix: str, *, label: str
) -> Dict[str, Tensor]:
    selected = {
        key[len(prefix) :]: tensor
        for key, tensor in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise KeyError(f"{label} checkpoint has no keys under prefix {prefix!r}")
    return selected


def _fill_audited_state(
    target_state: Mapping[str, Tensor],
    donor_state: Mapping[str, Tensor],
    source_key_for_target,
    transferable,
) -> Tuple[Dict[str, Tensor], Tuple[str, ...], Tuple[str, ...]]:
    converted: Dict[str, Tensor] = {}
    transferred = []
    requires_training = []
    for target_key, target_tensor in target_state.items():
        source_key = source_key_for_target(target_key)
        source_tensor = donor_state.get(source_key)
        if (
            transferable(target_key)
            and source_tensor is not None
            and tuple(source_tensor.shape) == tuple(target_tensor.shape)
        ):
            converted[target_key] = source_tensor.detach().clone()
            transferred.append(target_key)
        else:
            converted[target_key] = target_tensor.detach().clone()
            requires_training.append(target_key)
    return converted, tuple(sorted(transferred)), tuple(sorted(requires_training))


def convert_drivor_donor_state(
    donor_state: Mapping[str, Tensor],
    dynamic_scorer_state: Mapping[str, Tensor],
) -> Tuple[Dict[str, Tensor], DonorConversionReport]:
    """Transfer DrivoR trajectory embedding/heads, never old scene attention."""

    donor = _strip_required_prefix(
        donor_state, DRIVOR_DONOR_PREFIX, label="DrivoR"
    )

    def source_key(target_key: str) -> str:
        return (
            target_key.replace("trajectory_pos_embed.", "pos_embed.")
            .replace("ego_encoder.", "hist_encoding.")
            .replace("metric_heads.", "scorer.")
        )

    def transferable(target_key: str) -> bool:
        return target_key.startswith(("trajectory_pos_embed.", "metric_heads."))

    converted, transferred, requires_training = _fill_audited_state(
        dynamic_scorer_state, donor, source_key, transferable
    )
    report = DonorConversionReport(
        component="dynamic_scorer",
        donor_repository="valeoai/DrivoR",
        donor_commit="f02665403df799c1b4ddd8b0d34e073f0555c13a",
        transferred_keys=transferred,
        requires_training=requires_training,
        excluded_donor_prefixes=(
            "image_backbone.",
            "scene_embeds",
            "init_feature.",
            "trajectory_decoder.",
            "traj_head.",
            "scorer_attention.",
        ),
        inference_ready=False,
    )
    return converted, report


def convert_suprim_donor_state(
    donor_state: Mapping[str, Tensor],
    selector_state: Mapping[str, Tensor],
) -> Tuple[Dict[str, Tensor], DonorConversionReport]:
    """Transfer vocabulary, embeddings, and heads; initialize new decoders."""

    teacher = _strip_required_prefix(
        donor_state, SUPRIM_TEACHER_PREFIX, label="DriveSuprim teacher"
    )
    student = _strip_required_prefix(
        donor_state, SUPRIM_STUDENT_PREFIX, label="DriveSuprim student"
    )
    donor = dict(teacher)
    if "_trajectory_head.vocab" in student:
        donor["_trajectory_head.vocab"] = student["_trajectory_head.vocab"]

    def source_key(target_key: str) -> str:
        return "_trajectory_head.vocab" if target_key == "static_vocab" else target_key

    def transferable(target_key: str) -> bool:
        return target_key == "static_vocab" or (
            ".transformer." not in target_key
            and not target_key.startswith("status_encoding.")
        )

    converted, transferred, requires_training = _fill_audited_state(
        selector_state, donor, source_key, transferable
    )
    if "static_vocab" in transferred and not torch.equal(
        converted["static_vocab"].to(selector_state["static_vocab"].dtype),
        selector_state["static_vocab"],
    ):
        raise ValueError(
            "configured 8192 vocabulary differs from the official checkpoint vocabulary"
        )
    report = DonorConversionReport(
        component="suprim_selector",
        donor_repository="William-Yao-2000/DriveSuprim",
        donor_commit="80fe792d7654a596d92e20d030d1650f6f605c02",
        transferred_keys=transferred,
        requires_training=requires_training,
        excluded_donor_prefixes=(
            "_backbone.",
            "_trajectory_head.encoder.",
            "_trajectory_offset_head.encoder.",
            "_keyval_embedding.",
            "downscale_layer.",
            "_trajectory_head.transformer.",
            "_trajectory_offset_head.transformer_blocks.",
        ),
        inference_ready=False,
    )
    return converted, report
