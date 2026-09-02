#!/usr/bin/env python3
"""Jointly score Base and external proposal sets with the frozen DrivoR scorer."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from .candidate_metrics import _cluster_count
from .drivor_adapter import load_drivor_model, score_proposals
from .utils import (
    atomic_json,
    bootstrap_mean,
    cluster_bootstrap_mean,
    load_proposal_pickle,
    sha256_file,
    token_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhanced-export-root", type=Path, required=True)
    parser.add_argument("--base-proposals", type=Path, required=True)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--external-bank", type=Path, required=True)
    parser.add_argument("--external-matrix", type=Path, required=True)
    parser.add_argument("--bank-name", required=True)
    parser.add_argument("--drivor-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--include-duplicate-controls", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _matrix(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    result["tokens"] = result["tokens"].astype(str)
    result["log_names"] = result["log_names"].astype(str)
    return result


def _diverse_indices(scores: np.ndarray, proposals: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    selected: List[int] = []
    for index in order:
        if all(
            np.linalg.norm(proposals[index, :, :2] - proposals[other, :, :2], axis=-1).mean() >= 0.50
            or np.linalg.norm(proposals[index, -1, :2] - proposals[other, -1, :2]) >= 1.00
            for other in selected
        ):
            selected.append(int(index))
        if len(selected) == count:
            break
    for index in order:
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=np.int64)


def _base_diverse_indices(predicted: np.ndarray, proposals: np.ndarray, count: int) -> np.ndarray:
    # Greedily retain scorer-ranked geometric representatives, then fill by
    # scorer rank so the exact fixed budget is guaranteed.
    return _diverse_indices(predicted, proposals, count)


def _settings(
    base: np.ndarray,
    base_true: np.ndarray,
    base_predicted: np.ndarray,
    external: np.ndarray,
    external_true: np.ndarray,
    include_duplicates: bool,
    external_groups: int = 0,
) -> Dict[str, tuple[np.ndarray, np.ndarray, int, bool]]:
    """Return proposals, true scores, base-prefix length, additive flag."""

    values: Dict[str, tuple[np.ndarray, np.ndarray, int, bool]] = {
        "base64": (base, base_true, 64, True),
        "external_only": (external, external_true, 0, False),
        "union_full": (
            np.concatenate([base, external]),
            np.concatenate([base_true, external_true]),
            64,
            True,
        ),
    }
    if external_groups:
        if len(external) % external_groups:
            raise ValueError("External candidates cannot be split into equal groups")
        group_size = len(external) // external_groups
        for group_index in range(external_groups):
            group = slice(group_index * group_size, (group_index + 1) * group_size)
            values[f"external_stage{group_index}"] = (
                external[group], external_true[group], 0, False,
            )
            values[f"union_stage{group_index}"] = (
                np.concatenate([base, external[group]]),
                np.concatenate([base_true, external_true[group]]),
                64,
                True,
            )
    for count in (1, 8, 16, 32):
        if len(external) < count:
            continue
        indices = _diverse_indices(external_true, external, count)
        values[f"union_ideal{count}"] = (
            np.concatenate([base, external[indices]]),
            np.concatenate([base_true, external_true[indices]]),
            64,
            True,
        )
        if count in (8, 16, 32):
            base_count = 64 - count
            top = np.argsort(-base_predicted, kind="stable")[:base_count]
            diverse = _base_diverse_indices(base_predicted, base, base_count)
            values[f"fixed_top_base{base_count}_ideal{count}"] = (
                np.concatenate([base[top], external[indices]]),
                np.concatenate([base_true[top], external_true[indices]]),
                base_count,
                False,
            )
            values[f"fixed_diverse_base{base_count}_ideal{count}"] = (
                np.concatenate([base[diverse], external[indices]]),
                np.concatenate([base_true[diverse], external_true[indices]]),
                base_count,
                False,
            )
    if include_duplicates:
        order = np.argsort(-base_predicted, kind="stable")
        for count in (8, 16):
            indices = order[:count]
            values[f"duplicate{count}"] = (
                np.concatenate([base, base[indices]]),
                np.concatenate([base_true, base_true[indices]]),
                64,
                True,
            )
    return values


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _score_shard(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    shard_dir = args.output_dir / f"shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists() and args.resume:
        print(manifest_path.read_text(), end="")
        return
    export_dir = args.enhanced_export_root / f"shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    chunks = sorted(export_dir.glob("candidate_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No enhanced chunks under {export_dir}")
    base_bank = load_proposal_pickle(args.base_proposals)
    external_bank = load_proposal_pickle(args.external_bank)
    base_matrix = _matrix(args.base_matrix)
    external_matrix = _matrix(args.external_matrix)
    base_row = token_index(base_matrix["tokens"])
    external_row = token_index(external_matrix["tokens"])
    if set(external_bank) != set(external_row):
        raise RuntimeError("External bank and true-score matrix token sets differ")
    available = set(external_bank)
    export_tokens = []
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as archive:
            export_tokens.extend(archive["tokens"].astype(str).tolist())
    expected = [token for token in export_tokens if token in available]
    if not expected:
        raise RuntimeError("No external-bank scenes in enhanced shard")
    # Subset runs must preserve the exact original batch grouping. Repacking
    # sparse rows changes CUDA GEMM reduction order by a few ulps and violates
    # the strict 1e-6 Base parity gate. Inactive rows therefore receive a
    # shape-compatible filler bank; batches do not interact across scenes.
    filler_token = str(external_matrix["tokens"][0])

    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    model, config = load_drivor_model(args.config, args.checkpoint, args.dino_weights, device)

    result: Dict[str, Dict[str, List[object]]] = {}
    tokens_out: List[str] = []
    logs_out: List[str] = []
    base_parity_max = 0.0
    base_selected_equal = True
    settings_order: List[str] | None = None

    with torch.inference_mode():
        for chunk_path in chunks:
            with np.load(chunk_path, allow_pickle=False) as archive:
                tokens = archive["tokens"].astype(str)
                logs = archive["log_names"].astype(str)
                contexts = archive["scene_features"].astype(np.float32)
                statuses = archive["ego_status"].astype(np.float32)
            for start in range(0, len(tokens), args.batch_size):
                rows = list(range(start, min(start + args.batch_size, len(tokens))))
                batch_tokens = [str(tokens[index]) for index in rows]
                active_local = [local for local, token in enumerate(batch_tokens) if token in available]
                if not active_local:
                    continue
                scene_features = torch.as_tensor(contexts[rows], device=device)
                ego_status = torch.as_tensor(statuses[rows], device=device)
                ego_token = model.hist_encoding(ego_status)[:, None]
                scene_settings = []
                for token in batch_tokens:
                    bi = base_row[token]
                    external_token = token if token in available else filler_token
                    ei = external_row[external_token]
                    scene_settings.append(
                        _settings(
                            np.asarray(base_bank[token]["proposals"], dtype=np.float32),
                            np.asarray(base_matrix["candidate_scores"][bi], dtype=np.float64),
                            np.asarray(base_matrix["predicted_scores"][bi], dtype=np.float64),
                            np.asarray(external_bank[external_token]["proposals"], dtype=np.float32),
                            np.asarray(external_matrix["candidate_scores"][ei], dtype=np.float64),
                            args.include_duplicate_controls,
                            4 if args.bank_name.startswith("all_intermediate") else 0,
                        )
                    )
                names = list(scene_settings[0])
                if any(list(value) != names for value in scene_settings):
                    raise RuntimeError("Setting inventory differs within batch")
                if settings_order is None:
                    settings_order = names
                    result = {name: {key: [] for key in ("V_U", "O_U", "M_U", "selected_index", "selected_extra", "N090", "N095", "clusters095")} for name in names}
                elif names != settings_order:
                    raise RuntimeError("Setting inventory differs across batches")
                for name in names:
                    proposal_array = np.stack([value[name][0] for value in scene_settings]).astype(np.float32)
                    true_array = np.stack([value[name][1] for value in scene_settings]).astype(np.float64)
                    base_prefix = scene_settings[0][name][2]
                    scored = score_proposals(
                        model,
                        config,
                        torch.as_tensor(proposal_array, device=device),
                        scene_features,
                        ego_token,
                    )
                    predicted_values = scored["pdm_score"].cpu().numpy().astype(np.float64)
                    selected = predicted_values.argmax(axis=1)
                    for local in active_local:
                        candidate_index = selected[local]
                        scores = true_array[local]
                        proposals = proposal_array[local]
                        result[name]["V_U"].append(float(scores[candidate_index]))
                        result[name]["O_U"].append(float(scores.max()))
                        result[name]["M_U"].append(float(scores.mean()))
                        result[name]["selected_index"].append(int(candidate_index))
                        result[name]["selected_extra"].append(bool(candidate_index >= base_prefix))
                        result[name]["N090"].append(int(np.sum(scores >= 0.90)))
                        result[name]["N095"].append(int(np.sum(scores >= 0.95)))
                        result[name]["clusters095"].append(int(_cluster_count(proposals[scores >= 0.95])))
                    if name == "base64":
                        locked = np.stack([base_matrix["predicted_scores"][base_row[token]] for token in batch_tokens])
                        base_parity_max = max(base_parity_max, float(np.max(np.abs(predicted_values - locked))))
                        base_selected_equal &= bool(np.array_equal(selected, locked.argmax(axis=1)))
                tokens_out.extend([batch_tokens[local] for local in active_local])
                logs_out.extend([str(logs[rows[local]]) for local in active_local])
            print(json.dumps({"bank": args.bank_name, "shard": args.shard_index, "processed_export_chunk": chunk_path.name, "scenes": len(tokens_out)}), flush=True)
    if tokens_out != expected:
        raise RuntimeError("Union scorer silently dropped or reordered tokens")
    if base_parity_max > 1e-6 or not base_selected_equal:
        raise RuntimeError(f"Base scorer parity failed max={base_parity_max} index={base_selected_equal}")
    assert settings_order is not None
    arrays = {
        "tokens": np.asarray(tokens_out),
        "log_names": np.asarray(logs_out),
        "setting_names": np.asarray(settings_order),
    }
    for name in settings_order:
        for key, values in result[name].items():
            arrays[f"{name}__{key}"] = np.asarray(values)
    _atomic_npz(shard_dir / "union_results.npz", **arrays)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bank_name": args.bank_name,
        "scene_count": len(tokens_out),
        "settings": settings_order,
        "base_scorer_max_abs_error": base_parity_max,
        "base_selected_index_equal": base_selected_equal,
        "forward_parity_passed": True,
        "scorer_frozen": True,
        "joint_complete_set_scoring": True,
        "source_id_input": False,
        "external_proposals_detached": True,
        "base_proposals_sha256": sha256_file(args.base_proposals),
        "base_matrix_sha256": sha256_file(args.base_matrix),
        "external_bank": str(args.external_bank.resolve()),
        "external_bank_sha256": sha256_file(args.external_bank),
        "external_matrix": str(args.external_matrix.resolve()),
        "external_matrix_sha256": sha256_file(args.external_matrix),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _aggregate(args: argparse.Namespace) -> None:
    summary_path = args.output_dir / "injection_summary.json"
    if summary_path.exists() and args.resume:
        print(summary_path.read_text(), end="")
        return
    paths = sorted(args.output_dir.glob("shard_*/union_results.npz"))
    manifests = [json.loads((path.parent / "manifest.json").read_text()) for path in paths]
    if len(paths) != args.shard_count or not all(value["forward_parity_passed"] for value in manifests):
        raise RuntimeError("Union scorer shards incomplete or parity invalid")
    joined: Dict[str, List[np.ndarray]] = {}
    setting_names = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            names = archive["setting_names"].astype(str).tolist()
            if setting_names is None: setting_names = names
            elif names != setting_names: raise RuntimeError("Shard setting inventories differ")
            for key in archive.files:
                if key == "setting_names": continue
                joined.setdefault(key, []).append(archive[key])
    arrays = {key: np.concatenate(values) for key, values in joined.items()}
    tokens = arrays["tokens"].astype(str); logs = arrays["log_names"].astype(str)
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("Duplicate token across union shards")
    assert setting_names is not None
    base_v = arrays["base64__V_U"].astype(np.float64)
    base_o = arrays["base64__O_U"].astype(np.float64)
    saturated = (base_v >= 0.95) & ((base_o - base_v) <= 0.01)
    base_order = np.argsort(base_v, kind="stable")
    tail_masks = {"V_B_lt_090": base_v < 0.90}
    for fraction in (0.05, 0.10, 0.20):
        mask = np.zeros(len(base_v), dtype=bool)
        mask[base_order[: max(1, int(np.ceil(fraction * len(base_v))))]] = True
        tail_masks[f"bottom_{int(100 * fraction)}pct"] = mask
    rows = []
    detailed = {}
    for setting_index, name in enumerate(setting_names):
        v = arrays[f"{name}__V_U"].astype(np.float64)
        o = arrays[f"{name}__O_U"].astype(np.float64)
        selected_extra = arrays[f"{name}__selected_extra"].astype(bool)
        delta_v = v - base_v; delta_o = o - base_o
        additive = name == "base64" or name.startswith("union_") or name.startswith("duplicate")
        if additive and np.min(delta_o) < -1e-10:
            raise RuntimeError(f"Union oracle decreased for {name}")
        if name.startswith("duplicate") and np.max(np.abs(delta_o)) > 1e-10:
            raise RuntimeError(f"Duplicate candidates changed oracle for {name}")
        scene_oracle = bootstrap_mean(delta_o, n_bootstrap=args.bootstrap_replicates, seed=args.seed + 100 * setting_index)
        scene_selected = bootstrap_mean(delta_v, n_bootstrap=args.bootstrap_replicates, seed=args.seed + 100 * setting_index + 1)
        cluster_oracle = cluster_bootstrap_mean(delta_o, logs, n_bootstrap=args.bootstrap_replicates, seed=args.seed + 100 * setting_index)
        cluster_selected = cluster_bootstrap_mean(delta_v, logs, n_bootstrap=args.bootstrap_replicates, seed=args.seed + 100 * setting_index + 1)
        false_replacement = float(np.mean(selected_extra[saturated] & (v[saturated] < base_v[saturated] - 1e-12))) if saturated.any() else float("nan")
        row = {
            "bank_name": args.bank_name,
            "setting": name,
            "scene_count": len(v),
            "mean_oracle": float(o.mean()),
            "mean_selected": float(v.mean()),
            "mean_candidate": float(arrays[f"{name}__M_U"].mean()) if f"{name}__M_U" in arrays else float("nan"),
            "delta_oracle": float(delta_o.mean()),
            "delta_oracle_ci_low": scene_oracle["ci_low"],
            "delta_oracle_ci_high": scene_oracle["ci_high"],
            "delta_selected": float(delta_v.mean()),
            "delta_selected_ci_low": scene_selected["ci_low"],
            "delta_selected_ci_high": scene_selected["ci_high"],
            "cluster_delta_oracle_ci_low": cluster_oracle["ci_low"],
            "cluster_delta_oracle_ci_high": cluster_oracle["ci_high"],
            "cluster_delta_selected_ci_low": cluster_selected["ci_low"],
            "cluster_delta_selected_ci_high": cluster_selected["ci_high"],
            "gap_reduction": float(((base_o - base_v) - (o - v)).mean()),
            "improved_scene_share": float(np.mean(delta_v > 1e-12)),
            "degraded_scene_share": float(np.mean(delta_v < -1e-12)),
            "extra_selected_share": float(selected_extra.mean()),
            "selected_extra_better_than_base_selected_share": float(np.mean(selected_extra & (v > base_v + 1e-12))),
            "selected_extra_worse_than_base_selected_share": float(np.mean(selected_extra & (v < base_v - 1e-12))),
            "saturated_false_replacement_rate": false_replacement,
            "N090_increment": float((arrays[f"{name}__N090"] - arrays["base64__N090"]).mean()),
            "N095_increment": float((arrays[f"{name}__N095"] - arrays["base64__N095"]).mean()),
            "high_quality_cluster_increment": float((arrays[f"{name}__clusters095"] - arrays["base64__clusters095"]).mean()),
        }
        tail_detail = {}
        for tail_offset, (tail_name, tail_mask) in enumerate(tail_masks.items()):
            tail_delta = delta_v[tail_mask]
            row[f"{tail_name}_scene_count"] = int(tail_mask.sum())
            row[f"{tail_name}_base_selected"] = float(base_v[tail_mask].mean())
            row[f"{tail_name}_union_selected"] = float(v[tail_mask].mean())
            row[f"{tail_name}_delta_selected"] = float(tail_delta.mean())
            row[f"{tail_name}_improved_share"] = float(np.mean(tail_delta > 1e-12))
            row[f"{tail_name}_degraded_share"] = float(np.mean(tail_delta < -1e-12))
            if tail_name in {"V_B_lt_090", "bottom_10pct"}:
                tail_detail[tail_name] = bootstrap_mean(
                    tail_delta,
                    n_bootstrap=args.bootstrap_replicates,
                    seed=args.seed + 10_000 + 100 * setting_index + tail_offset,
                )
        rows.append(row)
        detailed[name] = {"scene_bootstrap_oracle": scene_oracle, "scene_bootstrap_selected": scene_selected, "cluster_bootstrap_oracle": cluster_oracle, "cluster_bootstrap_selected": cluster_selected, "tail_bootstrap_selected": tail_detail}
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "injection_summary.csv", index=False)
    _atomic_npz(args.output_dir / "union_results.npz", **arrays, setting_names=np.asarray(setting_names))
    external_scores = _matrix(args.external_matrix)["candidate_scores"].astype(np.float64)
    external_tokens = _matrix(args.external_matrix)["tokens"].astype(str)
    external_idx = token_index(external_tokens)
    base_for_ext = _matrix(args.base_matrix)
    base_idx = token_index(base_for_ext["tokens"])
    extra_beats_oracle = []; extra_beats_selected = []
    for token in tokens:
        ei=external_idx[token]; bi=base_idx[token]
        bv=float(base_for_ext["candidate_scores"][bi, int(base_for_ext["selected_indices"][bi])])
        bo=float(np.max(base_for_ext["candidate_scores"][bi])); eo=float(np.max(external_scores[ei]))
        extra_beats_oracle.append(eo > bo + 1e-12); extra_beats_selected.append(eo > bv + 1e-12)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bank_name": args.bank_name,
        "scene_count": len(tokens),
        "log_count": len(np.unique(logs)),
        "scope": "FULL" if len(tokens) == 12146 else "SUBSET",
        "frozen_base_scorer": True,
        "all_candidates_scored_jointly": True,
        "external_candidate_exceeds_base_oracle_share": float(np.mean(extra_beats_oracle)),
        "external_candidate_exceeds_base_selected_share": float(np.mean(extra_beats_selected)),
        "settings": rows,
        "bootstrap": detailed,
        "forward_parity": {
            "max_abs_error": max(float(value["base_scorer_max_abs_error"]) for value in manifests),
            "selected_index_equal": all(bool(value["base_selected_index_equal"]) for value in manifests),
        },
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        _aggregate(args)
    else:
        _score_shard(args)


if __name__ == "__main__":
    main()
