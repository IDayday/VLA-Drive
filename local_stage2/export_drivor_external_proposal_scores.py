#!/usr/bin/env python3
"""Export or score proposals with the released DrivOR model.

The default inference-only adapter builds DrivOR scene registers from the
current camera observation, bypasses DrivOR's trajectory decoder, and feeds an
immutable external proposal bank through DrivOR's detached trajectory
embedding and scoring decoder.  ``--native-proposals`` instead executes the
exact DrivOR trajectory decoder and persists its own final 64-proposal bank.
The proposal source then supplies only the immutable split token inventory.
Future annotations, metric caches and official PDM values are deliberately
absent from both processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_DETERMINISTIC_CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _DETERMINISTIC_CUBLAS:
    raise RuntimeError(
        "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python; setting "
        "it after importing torch is too late for deterministic evaluation."
    )

import numpy as np
import torch
from omegaconf import OmegaConf

import navsim
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader


EXPECTED_CANDIDATES = 64
EXPECTED_POSES = 8
EXPECTED_STATE_SIZE = 3
FACTOR_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)
_SEGMENT_SUFFIX = re.compile(r"_\d{5}_\d{5}$")


def _physical_log_name(log_name: str) -> str:
    return _SEGMENT_SUFFIX.sub("", str(log_name))


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_shard(token: str, shard_count: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def _atomic_json_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _comparable_lineage(payload: Mapping[str, object]) -> Dict[str, object]:
    """Remove restart-only metadata before enforcing immutable lineage."""

    comparable = dict(payload)
    comparable.pop("created_utc", None)
    return comparable


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


@dataclass(frozen=True)
class ProposalEntry:
    token: str
    log_name: str
    source_path: Path
    row_index: int


class ProposalSource:
    """Immutable proposal lookup supporting a pickle or chunked replay root."""

    def __init__(
        self,
        entries: Sequence[ProposalEntry],
        pickle_payload: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        self.entries = list(entries)
        self._pickle_payload = pickle_payload
        self._cached_path: Optional[Path] = None
        self._cached_chunk: Optional[Mapping[str, object]] = None

    def proposals(self, entry: ProposalEntry) -> torch.Tensor:
        if self._pickle_payload is not None:
            value = self._pickle_payload[entry.token]["proposals"]
            tensor = torch.as_tensor(value, dtype=torch.float32)
        else:
            if self._cached_path != entry.source_path:
                self._cached_chunk = torch.load(entry.source_path, map_location="cpu")
                self._cached_path = entry.source_path
            assert self._cached_chunk is not None
            tensor = self._cached_chunk["proposals"][entry.row_index].float()
        expected = (EXPECTED_CANDIDATES, EXPECTED_POSES, EXPECTED_STATE_SIZE)
        if tuple(tensor.shape) != expected:
            raise RuntimeError(
                f"Proposal shape mismatch for {entry.token}: {tuple(tensor.shape)} != {expected}"
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite proposal for {entry.token}")
        return tensor.contiguous()


def _load_fold_logs(
    fold_manifest: Optional[Path], fold_role: str
) -> Optional[set[str]]:
    if fold_manifest is None:
        if fold_role != "all":
            raise ValueError("--fold-role requires --fold-manifest")
        return None
    payload = json.loads(fold_manifest.read_text())
    key = {
        "train": "train_physical_logs",
        "validation": "validation_physical_logs",
    }.get(fold_role)
    if key is None:
        return None
    if key not in payload:
        raise KeyError(f"{key} missing from {fold_manifest}")
    return {str(value) for value in payload[key]}


def _load_log_mapping(matrix_path: Optional[Path]) -> Dict[str, str]:
    if matrix_path is None:
        return {}
    with np.load(matrix_path, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str)
        log_names = archive["log_names"].astype(str)
    if len(tokens) != len(set(tokens)):
        raise RuntimeError(f"Duplicate tokens in {matrix_path}")
    return dict(zip(tokens.tolist(), log_names.tolist()))


def _select_entry(
    token: str,
    log_name: str,
    allowed_physical_logs: Optional[set[str]],
    shard_count: int,
    shard_index: int,
) -> bool:
    if allowed_physical_logs is not None:
        if _physical_log_name(log_name) not in allowed_physical_logs:
            return False
    return _stable_shard(token, shard_count) == shard_index


def _build_pickle_source(
    path: Path,
    log_mapping: Mapping[str, str],
    allowed_physical_logs: Optional[set[str]],
    shard_count: int,
    shard_index: int,
    max_scenes: int,
) -> ProposalSource:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected token dictionary in {path}")
    entries: List[ProposalEntry] = []
    for token in sorted(payload):
        log_name = str(log_mapping.get(token, payload[token].get("log_name", "")))
        if not log_name:
            raise RuntimeError(
                f"No log name for {token}; provide --candidate-matrix with tokens/log_names"
            )
        if not _select_entry(
            token,
            log_name,
            allowed_physical_logs,
            shard_count,
            shard_index,
        ):
            continue
        entries.append(ProposalEntry(token, log_name, path, -1))
        if max_scenes and len(entries) >= max_scenes:
            break
    return ProposalSource(entries, pickle_payload=payload)


def _source_manifests(root: Path) -> List[Path]:
    return sorted(root.glob("*/manifest.json"))


def _build_chunk_source(
    root: Path,
    allowed_physical_logs: Optional[set[str]],
    shard_count: int,
    shard_index: int,
    max_scenes: int,
) -> ProposalSource:
    chunks = sorted(root.glob("**/chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No chunk_*.pt files under {root}")
    entries: List[ProposalEntry] = []
    seen: set[str] = set()
    for chunk_path in chunks:
        chunk = torch.load(chunk_path, map_location="cpu")
        tokens = [str(value) for value in chunk["tokens"]]
        log_names = [str(value) for value in chunk["log_names"]]
        if len(tokens) != len(log_names) or len(tokens) != len(chunk["proposals"]):
            raise RuntimeError(f"Malformed replay chunk: {chunk_path}")
        for row_index, (token, log_name) in enumerate(zip(tokens, log_names)):
            if token in seen:
                raise RuntimeError(f"Duplicate replay token: {token}")
            seen.add(token)
            if not _select_entry(
                token,
                log_name,
                allowed_physical_logs,
                shard_count,
                shard_index,
            ):
                continue
            entries.append(ProposalEntry(token, log_name, chunk_path, row_index))
            if max_scenes and len(entries) >= max_scenes:
                return ProposalSource(entries)
    return ProposalSource(entries)


def _load_completed_tokens(shard_dir: Path) -> set[str]:
    completed: set[str] = set()
    for path in sorted(shard_dir.glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu")
        for token in payload["tokens"]:
            token = str(token)
            if token in completed:
                raise RuntimeError(f"Duplicate completed token {token} in {shard_dir}")
            completed.add(token)
    return completed


def _load_model(
    config_path: Path,
    checkpoint_path: Path,
    dino_weights: Path,
    device: torch.device,
) -> Tuple[DrivoRModel, object]:
    full_config = OmegaConf.load(config_path)
    config = full_config["config"] if "config" in full_config else full_config
    config.image_backbone.model_weights = str(dino_weights)
    model = DrivoRModel(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint["state_dict"]
    prefix = "agent._drivor_model."
    state = {
        key[len(prefix) :]: value
        for key, value in raw_state.items()
        if key.startswith(prefix)
    }
    if len(state) != len(raw_state):
        expected_raw_keys = {prefix + key for key in state}
        unexpected = sorted(set(raw_state).difference(expected_raw_keys))
        raise RuntimeError(f"Unexpected non-DrivOR checkpoint keys: {unexpected[:5]}")
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return model, config


def _encode_context(
    model: DrivoRModel, features: Mapping[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model._config.full_history_status:
        ego_status = features["ego_status"].flatten(-2)
    else:
        ego_status = features["ego_status"][:, -1]
    ego_token = model.hist_encoding(ego_status)[:, None]
    if model.num_lidar:
        raise RuntimeError("External scorer adapter currently supports camera-only DrivOR")
    image = features["image"]
    scene_tokens = model.scene_embeds.expand(image.shape[0], -1, -1, -1)
    scene_features = model.image_backbone(image, scene_tokens)
    return scene_features, ego_token


def _stack_factor_logits(pred_logit: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([pred_logit[key] for key in FACTOR_KEYS], dim=-1)


def _aggregate_score(
    factor_logits: torch.Tensor, config: object
) -> torch.Tensor:
    probability = factor_logits.sigmoid()
    return (
        float(config.noc) * probability[..., 0].log()
        + float(config.dac) * probability[..., 1].log()
        + float(config.ddc) * probability[..., 2].log()
        + (
            float(config.ttc) * probability[..., 3]
            + float(config.ep) * probability[..., 4]
            + float(config.comfort) * probability[..., 5]
        ).log()
    )


def _score_proposals(
    model: DrivoRModel,
    config: object,
    proposals: torch.Tensor,
    scene_features: torch.Tensor,
    ego_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, candidate_count = proposals.shape[:2]
    embedded = model.pos_embed(proposals.reshape(batch_size, candidate_count, -1).detach())
    candidate_features = model.scorer_attention(embedded, scene_features) + ego_token
    pred_logit = model.scorer(proposals, candidate_features)[0]
    factor_logits = _stack_factor_logits(pred_logit)
    scores = _aggregate_score(factor_logits, config)
    return scores, factor_logits


def _batch_features(
    entries: Sequence[ProposalEntry],
    loader: SceneLoader,
    builder: DrivoRFeatureBuilder,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    images: List[torch.Tensor] = []
    ego_statuses: List[torch.Tensor] = []
    for entry in entries:
        agent_input = loader.get_agent_input_from_token(entry.token)
        feature = builder.compute_features(agent_input)
        images.append(feature["image"])
        ego_statuses.append(feature["ego_status"])
    return {
        "image": torch.stack(images).to(device=device, dtype=torch.float32),
        "ego_status": torch.stack(ego_statuses).to(device=device, dtype=torch.float32),
    }


def _self_parity(
    model: DrivoRModel,
    config: object,
    features: Mapping[str, torch.Tensor],
    tolerance: float,
) -> Dict[str, object]:
    full = model(dict(features))
    scene_features, ego_token = _encode_context(model, features)
    scores, factor_logits = _score_proposals(
        model, config, full["proposals"], scene_features, ego_token
    )
    reference_factors = _stack_factor_logits(full["pred_logit"])
    score_error = float((scores - full["pdm_score"]).abs().max().item())
    factor_error = float((factor_logits - reference_factors).abs().max().item())
    reference_index = full["pdm_score"].argmax(dim=1)
    adapter_index = scores.argmax(dim=1)
    same_index = bool(torch.equal(reference_index, adapter_index))
    passed = score_error <= tolerance and factor_error <= tolerance and same_index
    if not passed:
        raise RuntimeError(
            "DrivOR external scorer self-parity failed: "
            f"score={score_error:.3e}, factor={factor_error:.3e}, index={same_index}"
        )
    return {
        "score_max_abs_error": score_error,
        "factor_logit_max_abs_error": factor_error,
        "selected_index_equal": same_index,
        "tolerance": tolerance,
        "passed": passed,
    }


def _lineage(
    args: argparse.Namespace,
    source_manifests: Sequence[Path],
) -> Dict[str, object]:
    source: Dict[str, object]
    if args.proposal_pickle is not None:
        source = {
            "kind": "pickle",
            "path": str(args.proposal_pickle.resolve()),
            "sha256": _sha256_file(args.proposal_pickle),
        }
    else:
        source = {
            "kind": "chunk_root",
            "path": str(args.proposal_root.resolve()),
            "manifest_sha256": {
                str(path.relative_to(args.proposal_root)): _sha256_file(path)
                for path in source_manifests
            },
        }
    return {
        "schema_version": 1,
        "adapter": (
            "DrivORNativeProposalExporter"
            if args.native_proposals
            else "DrivORExternalProposalScorer"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "drivor_repo": str(args.drivor_repo.resolve()),
        "drivor_commit": _git_commit(args.drivor_repo),
        "drivor_checkpoint": str(args.checkpoint.resolve()),
        "drivor_checkpoint_sha256": _sha256_file(args.checkpoint),
        "drivor_config": str(args.config.resolve()),
        "drivor_config_sha256": _sha256_file(args.config),
        "dino_weights": str(args.dino_weights.resolve()),
        "dino_weights_sha256": _sha256_file(args.dino_weights),
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "proposal_source": source,
        "proposal_source_role": (
            "token_inventory_only"
            if args.native_proposals
            else "immutable_external_candidate_bank"
        ),
        "candidate_matrix": (
            {
                "path": str(args.candidate_matrix.resolve()),
                "sha256": _sha256_file(args.candidate_matrix),
            }
            if args.candidate_matrix is not None
            else None
        ),
        "fold_manifest": (
            {
                "path": str(args.fold_manifest.resolve()),
                "sha256": _sha256_file(args.fold_manifest),
            }
            if args.fold_manifest is not None
            else None
        ),
        "fold_role": args.fold_role,
        "split": args.split,
        "log_path": str(args.log_path.resolve()),
        "sensor_root": str(args.sensor_root.resolve()),
        "precision": "fp32",
        "candidate_count": EXPECTED_CANDIDATES,
        "pose_count": EXPECTED_POSES,
        "pose_interval_seconds": 0.5,
        "proposal_coordinate_frame": "current_ego_local",
        "inference_inputs": ["current_camera_images", "current_ego_status"]
        + ([] if args.native_proposals else ["proposals"]),
        "native_proposals": bool(args.native_proposals),
        "future_or_evaluator_input": False,
        "official_pdm_input": False,
        "base_scorer_input": False,
        "factor_keys": list(FACTOR_KEYS),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "seed": args.seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drivor-repo", type=Path, default=Path("/mnt/project/external/DrivoR"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/project/external/DrivoR/weights/releases/drivor_Nav1_25epochs.pth"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/mnt/project/external/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml"),
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        default=Path(
            "/mnt/project/external/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proposal-pickle", type=Path)
    source.add_argument("--proposal-root", type=Path)
    parser.add_argument("--candidate-matrix", type=Path)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--fold-role", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--split", choices=("navtrain", "navtest"), required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--self-parity-scenes", type=int, default=4)
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--native-proposals",
        action="store_true",
        help=(
            "Run DrivOR's own trajectory decoder.  The proposal source is "
            "used only as a token/log inventory and its trajectories are "
            "never passed to model forward."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    for path in (args.drivor_repo, args.checkpoint, args.config, args.dino_weights):
        if not path.exists():
            raise FileNotFoundError(path)
    imported_navsim = Path(navsim.__file__).resolve()
    if args.drivor_repo.resolve() not in imported_navsim.parents:
        raise RuntimeError(
            f"Imported navsim from {imported_navsim}, not DrivOR repo {args.drivor_repo}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    allowed_logs = _load_fold_logs(args.fold_manifest, args.fold_role)
    log_mapping = _load_log_mapping(args.candidate_matrix)
    source_manifests: List[Path] = []
    if args.proposal_pickle is not None:
        proposal_source = _build_pickle_source(
            args.proposal_pickle,
            log_mapping,
            allowed_logs,
            args.shard_count,
            args.shard_index,
            args.max_scenes,
        )
    else:
        assert args.proposal_root is not None
        source_manifests = _source_manifests(args.proposal_root)
        proposal_source = _build_chunk_source(
            args.proposal_root,
            allowed_logs,
            args.shard_count,
            args.shard_index,
            args.max_scenes,
        )
    entries = proposal_source.entries
    if not entries:
        raise RuntimeError("No proposal entries selected")

    lineage = _lineage(args, source_manifests)
    shard_dir = args.output_dir / f"shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if _comparable_lineage(manifest.get("lineage", {})) != _comparable_lineage(
            lineage
        ):
            raise RuntimeError(f"Completed output lineage mismatch: {manifest_path}")
        print(json.dumps({"status": "already_complete", "manifest": str(manifest_path)}))
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = shard_dir / "lineage.json"
    if lineage_path.exists():
        existing_lineage = json.loads(lineage_path.read_text())
        # created_utc is informational and differs across process restarts.
        if _comparable_lineage(existing_lineage) != _comparable_lineage(lineage):
            raise RuntimeError(f"Partial output lineage mismatch: {lineage_path}")
        lineage = existing_lineage
    else:
        _atomic_json_dump(lineage, lineage_path)

    completed = _load_completed_tokens(shard_dir)
    entry_tokens = {entry.token for entry in entries}
    if not completed.issubset(entry_tokens):
        raise RuntimeError("Partial output contains tokens outside current selection")
    pending = [entry for entry in entries if entry.token not in completed]

    unique_logs = sorted({entry.log_name for entry in entries})
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=0,
        frame_interval=1,
        has_route=True,
        log_names=unique_logs,
        tokens=[entry.token for entry in entries],
    )
    sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[3],
        lidar_pc=[],
    )
    loader = SceneLoader(args.log_path, args.sensor_root, scene_filter, sensor_config)
    missing = sorted(entry_tokens.difference(loader.tokens))
    if missing:
        raise RuntimeError(f"SceneLoader is missing {len(missing)} tokens, e.g. {missing[:5]}")

    model, config = _load_model(args.config, args.checkpoint, args.dino_weights, device)
    builder = DrivoRFeatureBuilder(config)
    parity_path = shard_dir / "self_parity.json"
    if parity_path.exists():
        parity_payload = json.loads(parity_path.read_text())
        parity_results = list(parity_payload.get("scenes", []))
    else:
        parity_results: List[Dict[str, object]] = []
    parity_tokens = [str(result["token"]) for result in parity_results]
    if len(parity_tokens) != len(set(parity_tokens)):
        raise RuntimeError(f"Duplicate self-parity token in {parity_path}")
    if not set(parity_tokens).issubset(entry_tokens):
        raise RuntimeError(f"Self-parity output contains token outside current selection")
    if not all(bool(result.get("passed", False)) for result in parity_results):
        raise RuntimeError(f"A persisted self-parity check failed in {parity_path}")
    output_buffer: Dict[str, List[object]] = {
        "tokens": [],
        "log_names": [],
        "scores": [],
        "factor_logits": [],
        "selected_indices": [],
    }
    if args.native_proposals:
        output_buffer["proposals"] = []
    chunk_index = len(list(shard_dir.glob("chunk_*.pt")))
    expected_parity_count = min(args.self_parity_scenes, len(entry_tokens))
    parity_remaining = max(0, expected_parity_count - len(parity_results))

    def flush() -> None:
        nonlocal chunk_index
        if not output_buffer["tokens"]:
            return
        payload = {
            "schema_version": 1,
            "factor_keys": FACTOR_KEYS,
            "tokens": list(output_buffer["tokens"]),
            "log_names": list(output_buffer["log_names"]),
            "scores": torch.stack(output_buffer["scores"]).float(),
            "factor_logits": torch.stack(output_buffer["factor_logits"]).half(),
            "selected_indices": torch.as_tensor(
                output_buffer["selected_indices"], dtype=torch.int16
            ),
        }
        if args.native_proposals:
            payload["proposals"] = torch.stack(
                output_buffer["proposals"]
            ).float()
        _atomic_torch_save(payload, shard_dir / f"chunk_{chunk_index:06d}.pt")
        chunk_index += 1
        for values in output_buffer.values():
            values.clear()

    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch_entries = pending[start : start + args.batch_size]
            features = _batch_features(batch_entries, loader, builder, device)
            if parity_remaining:
                for index in range(min(parity_remaining, len(batch_entries))):
                    single = {key: value[index : index + 1] for key, value in features.items()}
                    result = _self_parity(model, config, single, args.parity_tolerance)
                    result["token"] = batch_entries[index].token
                    parity_results.append(result)
                    _atomic_json_dump({"scenes": parity_results}, parity_path)
                parity_remaining -= min(parity_remaining, len(batch_entries))

            if args.native_proposals:
                native_output = model(dict(features))
                proposals = native_output["proposals"].float()
                scores = native_output["pdm_score"].float()
                factor_logits = _stack_factor_logits(
                    native_output["pred_logit"]
                ).float()
            else:
                scene_features, ego_token = _encode_context(model, features)
                proposals = torch.stack(
                    [proposal_source.proposals(entry) for entry in batch_entries]
                ).to(device=device, dtype=torch.float32)
                scores, factor_logits = _score_proposals(
                    model, config, proposals, scene_features, ego_token
                )
            if not torch.isfinite(scores).all() or not torch.isfinite(factor_logits).all():
                raise RuntimeError("DrivOR external scorer produced non-finite output")
            selected = scores.argmax(dim=1)
            for row, entry in enumerate(batch_entries):
                output_buffer["tokens"].append(entry.token)
                output_buffer["log_names"].append(entry.log_name)
                output_buffer["scores"].append(scores[row].float().cpu())
                output_buffer["factor_logits"].append(factor_logits[row].float().cpu())
                output_buffer["selected_indices"].append(int(selected[row].item()))
                if args.native_proposals:
                    output_buffer["proposals"].append(
                        proposals[row].float().cpu()
                    )
            if len(output_buffer["tokens"]) >= args.chunk_size:
                flush()
            print(
                json.dumps(
                    {
                        "processed": min(start + len(batch_entries), len(pending)),
                        "pending_total": len(pending),
                        "already_complete": len(completed),
                        "shard": f"{args.shard_index}/{args.shard_count}",
                    }
                ),
                flush=True,
            )
    flush()

    completed_after = _load_completed_tokens(shard_dir)
    if completed_after != entry_tokens:
        raise RuntimeError(
            f"Incomplete output: {len(completed_after)} / {len(entry_tokens)} tokens"
        )
    if len(parity_results) != expected_parity_count:
        raise RuntimeError(
            "Incomplete DrivOR self-parity audit: "
            f"{len(parity_results)} / {expected_parity_count} scenes"
        )
    manifest = {
        "schema_version": 1,
        "lineage": lineage,
        "scene_count": len(entry_tokens),
        "log_count": len(unique_logs),
        "chunk_count": chunk_index,
        "invalid_scene_count": 0,
        "native_proposals": bool(args.native_proposals),
        "self_parity": {
            "requested_scene_count": args.self_parity_scenes,
            "evaluated_scene_count": len(parity_results),
            "all_passed": all(value["passed"] for value in parity_results),
            "score_max_abs_error": max(
                (float(value["score_max_abs_error"]) for value in parity_results),
                default=0.0,
            ),
            "factor_logit_max_abs_error": max(
                (
                    float(value["factor_logit_max_abs_error"])
                    for value in parity_results
                ),
                default=0.0,
            ),
            "scenes": parity_results,
        },
    }
    _atomic_json_dump(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
