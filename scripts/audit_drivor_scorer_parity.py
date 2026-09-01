#!/usr/bin/env python3
"""Audit EpisodeDrive scorer parity against a frozen DrivoR git object.

The reference classes are executed directly from the source stored at the
pinned DrivoR commit. No checkout mutation and no network access are required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Dict, Mapping, Tuple, Type

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.layers.losses.episode_drive_loss import (  # noqa: E402
    EpisodeDriveLoss,
)
from navsim.agents.EpisodeDrive.score_module.scorer import (  # noqa: E402
    DRIVOR_SCORE_HEAD_NAMES,
    Scorer,
    aggregate_drivor_pdm_score,
)
from navsim.agents.EpisodeDrive.transformer_decoder import (  # noqa: E402
    TransformerDecoderScorer,
)


DRIVOR_REPOSITORY = "valeoai/DrivoR"
DRIVOR_COMMIT = "fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a"
UPSTREAM_FILES = {
    "drivor_model.py": (
        "navsim/agents/drivoR/drivor_model.py",
        "bd530b606970fb72bd657363924cbbd4f5f072f8560f8a1cd3d39ee0bd91f59d",
    ),
    "transformer_decoder.py": (
        "navsim/agents/drivoR/transformer_decoder.py",
        "06207e844e801a6d8c16ccbcb7194d9c039cd13567f7b3b6b8ab0b3633ffe43f",
    ),
    "scorer.py": (
        "navsim/agents/drivoR/score_module/scorer.py",
        "f8f3c467d67cf6d593c06cbff71471749ccd92e14cbcc86a12b4a813ec3a4b54",
    ),
    "drivor_loss.py": (
        "navsim/agents/drivoR/layers/losses/drivor_loss.py",
        "e928a06a58d56363ff31bd8664f56a4f27af5b975125ba7c6bdb2d02715f8efb",
    ),
}


def _default_drivor_repo() -> Path:
    candidates = [
        os.environ.get("DRIVOR_REPO"),
        "/mnt/project/external/DrivoR",
        str(REPO_ROOT.parent / "DrivoR"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / ".git").exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Cannot locate a DrivoR git repository. Set DRIVOR_REPO to a clone "
        f"containing commit {DRIVOR_COMMIT}."
    )


def _git_show(repo: Path, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "show", f"{DRIVOR_COMMIT}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"DrivoR clone {repo} does not contain {DRIVOR_COMMIT}:{path}: "
            f"{stderr.strip()}"
        ) from exc


def load_and_verify_upstream_sources(repo: Path) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for name, (path, expected_sha256) in UPSTREAM_FILES.items():
        raw = _git_show(repo, path)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise AssertionError(
                f"Frozen source hash mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        sources[name] = raw.decode("utf-8")

    model_source = sources["drivor_model.py"]
    required_model_fragments = (
        "proposals.reshape(B, N, -1).detach()",
        "self.scorer_attention = TransformerDecoderScorer",
        "pred_logit['no_at_fault_collisions'].sigmoid().log()",
        "torch.argmax(pdm_score, dim=1)",
    )
    for fragment in required_model_fragments:
        if fragment not in model_source:
            raise AssertionError(f"Pinned DrivoR scorer fragment is missing: {fragment}")
    if "mask_valid_ttc = (gt_time_to_collision_within_bound != 2.0).float()" not in sources[
        "drivor_loss.py"
    ]:
        raise AssertionError("Pinned DrivoR TTC invalid-target mask is missing")
    return sources


def _execute_upstream_module(
    module_name: str,
    package: str,
    source: str,
    source_label: str,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__package__ = package
    module.__file__ = source_label
    sys.modules[module_name] = module
    exec(compile(source, source_label, "exec"), module.__dict__)
    return module


def load_upstream_classes(
    sources: Mapping[str, str],
) -> Tuple[Type[nn.Module], Type[nn.Module]]:
    transformer_module = _execute_upstream_module(
        "navsim.agents.EpisodeDrive._frozen_drivor_transformer_decoder",
        "navsim.agents.EpisodeDrive",
        sources["transformer_decoder.py"],
        f"{DRIVOR_REPOSITORY}@{DRIVOR_COMMIT}:transformer_decoder.py",
    )
    scorer_module = _execute_upstream_module(
        "navsim.agents.EpisodeDrive.score_module._frozen_drivor_scorer",
        "navsim.agents.EpisodeDrive.score_module",
        sources["scorer.py"],
        f"{DRIVOR_REPOSITORY}@{DRIVOR_COMMIT}:scorer.py",
    )
    return transformer_module.TransformerDecoderScorer, scorer_module.Scorer


def make_config(**overrides) -> SimpleNamespace:
    values = {
        "b2d": False,
        "proposal_num": 64,
        "num_poses": 8,
        "scorer_ref_num": 4,
        "tf_d_model": 256,
        "tf_d_ffn": 1024,
        "refiner_num_heads": 1,
        "refiner_ls_values": 0.0,
        "double_score": False,
        "agent_pred": False,
        "area_pred": False,
        "bev_map": False,
        "bev_agent": False,
        "one_token_per_traj": True,
        "noc": 1.0,
        "dac": 1.0,
        "ddc": 0.0,
        "ttc": 5.0,
        "ep": 5.0,
        "comfort": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_pos_embed(config: SimpleNamespace) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.num_poses * 3, config.tf_d_ffn),
        nn.ReLU(),
        nn.Linear(config.tf_d_ffn, config.tf_d_model),
    )


def _upstream_pdm_score(
    pred_logit: Dict[str, torch.Tensor], config: SimpleNamespace
) -> torch.Tensor:
    # Verbatim expression from DrivoRModel.forward at the pinned commit.
    return (
        config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +
        (config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
         config.ep * pred_logit['ego_progress'].sigmoid() +
         config.comfort * pred_logit['comfort'].sigmoid()).log()
    )


def _forward_scorer_path(
    pos_embed: nn.Module,
    decoder: nn.Module,
    scorer: nn.Module,
    proposals: torch.Tensor,
    scene_features: torch.Tensor,
    ego_token: torch.Tensor,
):
    batch_size, proposal_count, pose_count, state_size = proposals.shape
    if pose_count != 8 or state_size != 3:
        raise AssertionError(f"Expected complete [8,3] trajectories, got {proposals.shape}")
    embedded_traj = pos_embed(
        proposals.reshape(batch_size, proposal_count, -1).detach()
    )
    scorer_features = decoder(embedded_traj, scene_features) + ego_token
    return scorer(proposals, scorer_features)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _state_shapes(module: nn.Module) -> Dict[str, Tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in module.state_dict().items()}


def _audit_ttc_mask() -> Dict[str, object]:
    logits = {
        name: torch.zeros(
            1,
            3,
            dtype=torch.float64,
            requires_grad=(name == "time_to_collision_within_bound"),
        )
        for name in DRIVOR_SCORE_HEAD_NAMES
    }
    ttc_logits = logits["time_to_collision_within_bound"]
    with torch.no_grad():
        ttc_logits[0, 2] = 100.0
    target_scores = torch.zeros(1, 3, 7)
    target_scores[..., 3] = torch.tensor([[0.0, 1.0, 2.0]])
    losses = EpisodeDriveLoss().score_loss(
        logits,
        None,
        None,
        None,
        target_scores,
        None,
        None,
        None,
        None,
    )[0]
    ttc_loss = losses[1]
    ttc_loss.backward()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    invalid_grad = float(ttc_logits.grad[0, 2].item())
    return {
        "loss": float(ttc_loss.item()),
        "expected": float(expected.item()),
        "loss_abs_diff": _max_abs_diff(ttc_loss.detach(), expected),
        "invalid_grad": invalid_grad,
        "passed": _max_abs_diff(ttc_loss.detach(), expected) == 0.0
        and invalid_grad == 0.0,
    }


def run_audit(drivor_repo: Path, seed: int = 20260901) -> Dict[str, object]:
    sources = load_and_verify_upstream_sources(drivor_repo)
    upstream_decoder_cls, upstream_scorer_cls = load_upstream_classes(sources)
    config = make_config()

    torch.manual_seed(seed)
    local_pos_embed = _make_pos_embed(config)
    local_decoder = TransformerDecoderScorer(
        num_layers=config.scorer_ref_num,
        d_model=config.tf_d_model,
        proj_drop=0.1,
        drop_path=0.2,
        config=config,
    )
    local_scorer = Scorer(config)

    upstream_pos_embed = copy.deepcopy(local_pos_embed)
    upstream_decoder = upstream_decoder_cls(
        num_layers=config.scorer_ref_num,
        d_model=config.tf_d_model,
        proj_drop=0.1,
        drop_path=0.2,
        config=config,
    )
    upstream_scorer = upstream_scorer_cls(config)
    upstream_decoder.load_state_dict(local_decoder.state_dict(), strict=True)
    upstream_scorer.load_state_dict(local_scorer.state_dict(), strict=True)

    modules = (
        local_pos_embed,
        local_decoder,
        local_scorer,
        upstream_pos_embed,
        upstream_decoder,
        upstream_scorer,
    )
    for module in modules:
        module.eval()

    generator = torch.Generator().manual_seed(seed + 1)
    proposals = torch.randn(2, 64, 8, 3, generator=generator)
    scene_features = torch.randn(2, 16, 256, generator=generator)
    ego_token = torch.randn(2, 1, 256, generator=generator)

    with torch.no_grad():
        local_output = _forward_scorer_path(
            local_pos_embed,
            local_decoder,
            local_scorer,
            proposals,
            scene_features,
            ego_token,
        )
        upstream_output = _forward_scorer_path(
            upstream_pos_embed,
            upstream_decoder,
            upstream_scorer,
            proposals,
            scene_features,
            ego_token,
        )
        local_logits = local_output[0]
        upstream_logits = upstream_output[0]
        component_diffs = {
            name: _max_abs_diff(local_logits[name], upstream_logits[name])
            for name in DRIVOR_SCORE_HEAD_NAMES
        }
        local_pdm = aggregate_drivor_pdm_score(local_logits, config)
        upstream_pdm = _upstream_pdm_score(upstream_logits, config)
        local_indices = torch.argmax(local_pdm, dim=1)
        upstream_indices = torch.argmax(upstream_pdm, dim=1)

    proposal_leaf = proposals.detach().clone().requires_grad_(True)
    scene_leaf = scene_features.detach().clone().requires_grad_(True)
    grad_output = _forward_scorer_path(
        local_pos_embed,
        local_decoder,
        local_scorer,
        proposal_leaf,
        scene_leaf,
        ego_token,
    )
    scorer_only_loss = sum(value.sum() for value in grad_output[0].values())
    scorer_only_loss.backward()
    proposal_grad_norm = (
        0.0 if proposal_leaf.grad is None else float(proposal_leaf.grad.norm().item())
    )
    scene_grad_norm = float(scene_leaf.grad.norm().item())

    local_shapes = {
        "decoder": _state_shapes(local_decoder),
        "scorer": _state_shapes(local_scorer),
    }
    upstream_shapes = {
        "decoder": _state_shapes(upstream_decoder),
        "scorer": _state_shapes(upstream_scorer),
    }

    b2d_config = make_config(b2d=True, agent_pred=True, area_pred=True)
    b2d_local = Scorer(b2d_config)
    b2d_upstream = upstream_scorer_cls(b2d_config)
    b2d_shape_parity = _state_shapes(b2d_local) == _state_shapes(b2d_upstream)
    b2d_agent_out = b2d_local.pred_col_agent.mlp[-1].out_features
    b2d_area_out = b2d_local.pred_area.mlp[-1].out_features

    ttc_audit = _audit_ttc_mask()
    report: Dict[str, object] = {
        "repository": DRIVOR_REPOSITORY,
        "commit": DRIVOR_COMMIT,
        "source_sha256": {
            name: expected for name, (_, expected) in UPSTREAM_FILES.items()
        },
        "synthetic_shapes": {
            "proposals": list(proposals.shape),
            "scene_features": list(scene_features.shape),
            "ego_token": list(ego_token.shape),
        },
        "scorer_decoder_layers": len(local_decoder.layers),
        "head_names": list(local_scorer.pred_score.keys()),
        "component_max_abs_diff": component_diffs,
        "pdm_score_max_abs_diff": _max_abs_diff(local_pdm, upstream_pdm),
        "selected_indices_equal": bool(torch.equal(local_indices, upstream_indices)),
        "selected_indices": local_indices.tolist(),
        "proposal_grad_is_none": proposal_leaf.grad is None,
        "proposal_grad_norm": proposal_grad_norm,
        "scene_feature_grad_norm": scene_grad_norm,
        "state_dict_shapes_equal": local_shapes == upstream_shapes,
        "b2d_state_dict_shapes_equal": b2d_shape_parity,
        "b2d_agent_output_dim": b2d_agent_out,
        "b2d_area_output_dim": b2d_area_out,
        "ttc_invalid_mask": ttc_audit,
    }
    checks = (
        tuple(local_scorer.pred_score.keys()) == DRIVOR_SCORE_HEAD_NAMES,
        len(local_decoder.layers) == 4,
        all(diff == 0.0 for diff in component_diffs.values()),
        report["pdm_score_max_abs_diff"] == 0.0,
        report["selected_indices_equal"],
        proposal_grad_norm == 0.0,
        scene_grad_norm > 0.0,
        report["state_dict_shapes_equal"],
        b2d_shape_parity,
        b2d_agent_out == 2 * 6 * 9,
        b2d_area_out == 8 * 2,
        ttc_audit["passed"],
    )
    report["passed"] = all(checks)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drivor-repo",
        type=Path,
        default=None,
        help="DrivoR clone containing the pinned commit (or set DRIVOR_REPO)",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    repo = args.drivor_repo or _default_drivor_repo()
    report = run_audit(repo.resolve(), seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
