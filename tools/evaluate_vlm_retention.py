#!/usr/bin/env python3
"""Evaluate general VLM retention after DriveDreamer action-only fine-tuning.

The tool deliberately keeps model and dataset paths explicit. It never downloads
model weights and writes only below the caller-provided output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


QWEN_CHECKPOINT_PREFIX = "qwen_vl_interface.model."
MMSTAR_ALLOWED = "ABCD"
MMLU_PRO_ALLOWED = "ABCDEFGHIJ"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_choice(response: str, allowed: str) -> str | None:
    """Extract one unambiguous multiple-choice letter from a model response."""

    normalized = response.strip().upper()
    allowed_class = re.escape(allowed)
    leading = re.match(rf"^[\s\(\[]*([{allowed_class}])(?:[\s\)\].,:;]|$)", normalized)
    if leading:
        return leading.group(1)

    explicit = re.findall(
        rf"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*[\(\[]?([{allowed_class}])(?:[\s\)\].,:;]|$)",
        normalized,
    )
    if len(set(explicit)) == 1:
        return explicit[0]

    standalone = re.findall(rf"(?<![A-Z])([{allowed_class}])(?![A-Z])", normalized)
    return standalone[0] if len(set(standalone)) == 1 else None


def stratified_sample(
    frame: pd.DataFrame,
    *,
    group_column: str,
    per_group: int,
    seed: int,
    id_column: str,
) -> pd.DataFrame:
    """Select a deterministic, order-independent sample from every group."""

    if per_group <= 0:
        return frame.sort_values([group_column, id_column]).reset_index(drop=True)
    selected: list[pd.DataFrame] = []
    for _, group in frame.groupby(group_column, sort=True):
        ranked = group.assign(
            _sample_rank=[
                hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
                for value in group[id_column]
            ]
        ).sort_values(["_sample_rank", id_column])
        selected.append(ranked.head(per_group).drop(columns="_sample_rank"))
    return pd.concat(selected, ignore_index=True).sort_values(
        [group_column, id_column]
    ).reset_index(drop=True)


def _mcnemar_exact_p(baseline_only: int, treatment_only: int) -> float:
    discordant = baseline_only + treatment_only
    if discordant == 0:
        return 1.0
    lower = min(baseline_only, treatment_only)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_statistics(
    baseline: Mapping[str, bool],
    treatment: Mapping[str, bool],
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Return paired accuracy delta, bootstrap CI, and exact McNemar p-value."""

    item_ids = sorted(set(baseline).intersection(treatment))
    if not item_ids:
        raise ValueError("paired comparison has no common item ids")
    base = np.asarray([bool(baseline[item_id]) for item_id in item_ids], dtype=np.float64)
    tuned = np.asarray([bool(treatment[item_id]) for item_id in item_ids], dtype=np.float64)
    delta = tuned - base
    rng = np.random.default_rng(seed)
    if bootstrap_samples > 0:
        draws = rng.integers(0, len(item_ids), size=(bootstrap_samples, len(item_ids)))
        bootstrap_delta = delta[draws].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_delta, [0.025, 0.975]).tolist()
    else:
        ci_low = ci_high = float("nan")
    baseline_only = int(np.sum((base == 1) & (tuned == 0)))
    treatment_only = int(np.sum((base == 0) & (tuned == 1)))
    return {
        "sample_count": len(item_ids),
        "baseline_accuracy": float(base.mean()),
        "treatment_accuracy": float(tuned.mean()),
        "delta_accuracy": float(delta.mean()),
        "delta_bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "baseline_only_correct": baseline_only,
        "treatment_only_correct": treatment_only,
        "mcnemar_exact_p": _mcnemar_exact_p(baseline_only, treatment_only),
    }


def paired_continuous_statistics(
    baseline: Mapping[str, float],
    treatment: Mapping[str, float],
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    item_ids = sorted(set(baseline).intersection(treatment))
    if not item_ids:
        raise ValueError("paired comparison has no common item ids")
    base = np.asarray([float(baseline[item_id]) for item_id in item_ids])
    tuned = np.asarray([float(treatment[item_id]) for item_id in item_ids])
    delta = tuned - base
    rng = np.random.default_rng(seed)
    bootstrap_means: list[np.ndarray] = []
    remaining = bootstrap_samples
    while remaining > 0:
        batch_size = min(100, remaining)
        draws = rng.integers(0, len(item_ids), size=(batch_size, len(item_ids)))
        bootstrap_means.append(delta[draws].mean(axis=1))
        remaining -= batch_size
    if bootstrap_means:
        bootstrap = np.concatenate(bootstrap_means)
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        ci_low = ci_high = float("nan")
    return {
        "sample_count": len(item_ids),
        "baseline_mean": float(base.mean()),
        "treatment_mean": float(tuned.mean()),
        "delta_mean": float(delta.mean()),
        "delta_bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "delta_median": float(np.median(delta)),
        "delta_p10": float(np.quantile(delta, 0.1)),
        "delta_p90": float(np.quantile(delta, 0.9)),
        "improved_count": int(np.sum(delta > 0)),
        "unchanged_count": int(np.sum(delta == 0)),
        "regressed_count": int(np.sum(delta < 0)),
    }


def standard_qwen_key(checkpoint_key: str) -> str | None:
    if not checkpoint_key.startswith(QWEN_CHECKPOINT_PREFIX):
        return None
    return checkpoint_key[len(QWEN_CHECKPOINT_PREFIX) :]


def parameter_groups(checkpoint_key: str) -> tuple[str, str, str]:
    standard = standard_qwen_key(checkpoint_key)
    if standard is None:
        return ("non_qwen", "non_qwen", "non_qwen")
    if standard.startswith("model.visual."):
        block = re.match(r"model\.visual\.blocks\.(\d+)\.", standard)
        if block:
            detail = f"visual.blocks.{block.group(1)}"
        else:
            detail = "visual." + standard.split(".")[2]
        return ("qwen", "visual", detail)
    if standard.startswith("model.language_model."):
        layer = re.match(r"model\.language_model\.layers\.(\d+)\.", standard)
        if layer:
            detail = f"language.layers.{layer.group(1)}"
        elif standard.startswith("model.language_model.embed_tokens."):
            detail = "language.embed_tokens"
        else:
            detail = "language.norm"
        return ("qwen", "language", detail)
    if standard.startswith("lm_head."):
        return ("qwen", "lm_head", "lm_head")
    return ("qwen", "other_qwen", "other_qwen")


def _safetensor_locations(model_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        locations = {
            key: model_dir / filename for key, filename in payload["weight_map"].items()
        }
        return locations, sorted(set(locations.values()))
    tensor_path = model_dir / "model.safetensors"
    if not tensor_path.is_file():
        raise FileNotFoundError(
            f"base model has neither model.safetensors nor a shard index: {model_dir}"
        )
    from safetensors import safe_open

    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        locations = {key: tensor_path for key in tensors.keys()}
    return locations, [tensor_path]


def _tensor_moments(base: Any, tuned: Any, chunk_size: int) -> dict[str, Any]:
    import torch

    if tuple(base.shape) != tuple(tuned.shape):
        raise ValueError(f"tensor shape mismatch: {tuple(base.shape)} != {tuple(tuned.shape)}")
    base_flat = base.reshape(-1)
    tuned_flat = tuned.reshape(-1)
    moments = {
        "numel": base_flat.numel(),
        "base_sq": 0.0,
        "tuned_sq": 0.0,
        "delta_sq": 0.0,
        "dot": 0.0,
        "equal": 0,
        "max_abs_delta": 0.0,
    }
    for start in range(0, base_flat.numel(), chunk_size):
        stop = min(start + chunk_size, base_flat.numel())
        base_chunk = base_flat[start:stop].float()
        tuned_chunk = tuned_flat[start:stop].float()
        delta = tuned_chunk - base_chunk
        moments["base_sq"] += float(torch.sum(base_chunk * base_chunk, dtype=torch.float64))
        moments["tuned_sq"] += float(torch.sum(tuned_chunk * tuned_chunk, dtype=torch.float64))
        moments["delta_sq"] += float(torch.sum(delta * delta, dtype=torch.float64))
        moments["dot"] += float(torch.sum(base_chunk * tuned_chunk, dtype=torch.float64))
        moments["equal"] += int(torch.count_nonzero(base_chunk == tuned_chunk))
        moments["max_abs_delta"] = max(
            moments["max_abs_delta"], float(delta.abs().max())
        )
    return moments


def _finalize_moments(moments: Mapping[str, Any]) -> dict[str, Any]:
    numel = int(moments["numel"])
    base_sq = float(moments["base_sq"])
    tuned_sq = float(moments["tuned_sq"])
    delta_sq = float(moments["delta_sq"])
    denominator = math.sqrt(base_sq * tuned_sq)
    return {
        "numel": numel,
        "relative_l2_delta": math.sqrt(delta_sq / base_sq) if base_sq else 0.0,
        "rms_delta": math.sqrt(delta_sq / numel) if numel else 0.0,
        "cosine_similarity": float(moments["dot"]) / denominator if denominator else 1.0,
        "changed_fraction": 1.0 - int(moments["equal"]) / numel if numel else 0.0,
        "max_abs_delta": float(moments["max_abs_delta"]),
    }


def inspect_drift(args: argparse.Namespace) -> None:
    import torch
    from safetensors import safe_open

    base_dir = Path(args.base_model).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    locations, files = _safetensor_locations(base_dir)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"expected a state dict, got {type(checkpoint)!r}")

    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "numel": 0,
            "base_sq": 0.0,
            "tuned_sq": 0.0,
            "delta_sq": 0.0,
            "dot": 0.0,
            "equal": 0,
            "max_abs_delta": 0.0,
        }
    )
    tensor_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    with ExitStack() as stack:
        readers = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in files
        }
        for index, (checkpoint_key, tuned) in enumerate(checkpoint.items(), start=1):
            standard = standard_qwen_key(checkpoint_key)
            if standard is None:
                continue
            base_key = standard
            if base_key == "lm_head.weight" and base_key not in locations:
                base_key = "model.language_model.embed_tokens.weight"
            if base_key not in locations:
                missing.append(standard)
                continue
            base = readers[locations[base_key]].get_tensor(base_key)
            moments = _tensor_moments(base, tuned, args.chunk_size)
            metrics = _finalize_moments(moments)
            groups = parameter_groups(checkpoint_key)
            tensor_rows.append(
                {
                    "checkpoint_key": checkpoint_key,
                    "base_key": base_key,
                    "component": groups[1],
                    "detail_group": groups[2],
                    **metrics,
                }
            )
            for group in dict.fromkeys(groups):
                target = aggregate[group]
                for field in ("numel", "base_sq", "tuned_sq", "delta_sq", "dot", "equal"):
                    target[field] += moments[field]
                target["max_abs_delta"] = max(
                    target["max_abs_delta"], moments["max_abs_delta"]
                )
            if index % 100 == 0:
                print(f"[drift] scanned {index}/{len(checkpoint)} checkpoint tensors", flush=True)

    group_rows = {
        name: _finalize_moments(values) for name, values in sorted(aggregate.items())
    }
    top_changed = sorted(
        tensor_rows, key=lambda row: row["relative_l2_delta"], reverse=True
    )[:30]
    payload = {
        "schema_version": 1,
        "base_model": str(base_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "matched_qwen_tensor_count": len(tensor_rows),
        "missing_base_keys": missing,
        "groups": group_rows,
        "top_changed_tensors": top_changed,
        "tensors": tensor_rows,
    }
    output = Path(args.output).resolve()
    _atomic_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


def _format_mmstar_prompt(question: str, with_image: bool) -> str:
    modality = "Use the image to answer" if with_image else "Answer"
    return (
        f"{modality} the following multiple-choice question. "
        "Respond with only one uppercase option letter (A, B, C, or D).\n\n"
        f"{question}"
    )


def _format_mmlu_prompt(question: str, options: Sequence[str]) -> str:
    option_text = "\n".join(
        f"{letter}. {option}" for letter, option in zip(MMLU_PRO_ALLOWED, options)
    )
    return (
        "Answer the following multiple-choice question. Respond with only one "
        "uppercase option letter (A through J).\n\n"
        f"Question: {question}\n{option_text}"
    )


def _load_model(
    base_model: Path,
    checkpoint_path: Path | None,
    *,
    device: str,
    min_pixels: int,
    max_pixels: int,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"base VLM config is missing: {base_model / 'config.json'}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=True
        )
        qwen_state = {
            standard: value
            for key, value in checkpoint.items()
            if (standard := standard_qwen_key(key)) is not None
        }
        incompatible = model.load_state_dict(qwen_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict Qwen load unexpectedly failed: {incompatible}")
        del checkpoint, qwen_state
    processor = AutoProcessor.from_pretrained(
        base_model,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=True,
    )
    # Decoder-only batched generation must left-pad; right padding changes the
    # continuation context of shorter prompts and can silently corrupt scores.
    processor.tokenizer.padding_side = "left"
    model.to(device)
    return model, processor


def _generate_batches(
    *,
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    task: str,
    batch_size: int,
    device: str,
    max_new_tokens: int,
) -> Iterable[dict[str, Any]]:
    import torch
    from PIL import Image

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        texts: list[str] = []
        images: list[Any] | None = [] if task == "mmstar_visual" else None
        for row in batch_rows:
            if task.startswith("mmstar"):
                with_image = task == "mmstar_visual"
                content: list[dict[str, Any]] = []
                if with_image:
                    image = Image.open(io.BytesIO(row["image"])).convert("RGB")
                    content.append({"type": "image", "image": image})
                    assert images is not None
                    images.append(image)
                content.append(
                    {
                        "type": "text",
                        "text": _format_mmstar_prompt(row["question"], with_image),
                    }
                )
            elif task == "mmlu_pro":
                content = [
                    {
                        "type": "text",
                        "text": _format_mmlu_prompt(row["question"], row["options"]),
                    }
                ]
            else:
                raise ValueError(f"unknown task: {task}")
            messages = [{"role": "user", "content": content}]
            texts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        allowed = MMSTAR_ALLOWED if task.startswith("mmstar") else MMLU_PRO_ALLOWED
        choice_token_ids: list[list[int]] = []
        for letter in allowed:
            variants = {
                tuple(processor.tokenizer.encode(letter, add_special_tokens=False)),
                tuple(processor.tokenizer.encode(f" {letter}", add_special_tokens=False)),
            }
            single_token_variants = sorted(
                token_ids[0] for token_ids in variants if len(token_ids) == 1
            )
            if not single_token_variants:
                raise ValueError(f"option {letter!r} has no single-token representation")
            choice_token_ids.append(single_token_variants)
        with torch.inference_mode():
            output = model(**inputs, logits_to_keep=1, use_cache=False)
            next_token_logits = output.logits[:, -1, :].float()
        choice_scores = torch.stack(
            [next_token_logits[:, token_ids].max(dim=1).values for token_ids in choice_token_ids],
            dim=1,
        )
        top = torch.topk(choice_scores, k=2, dim=1)
        predictions = [allowed[index] for index in top.indices[:, 0].tolist()]
        margins = (top.values[:, 0] - top.values[:, 1]).tolist()
        for row, predicted, margin in zip(batch_rows, predictions, margins):
            yield {
                "task": task,
                "item_id": str(row["item_id"]),
                "category": str(row["category"]),
                "sub_category": str(row.get("sub_category", "")),
                "gold": str(row["answer"]),
                "prediction": predicted,
                "correct": predicted == str(row["answer"]),
                "response": predicted,
                "choice_logit_margin": float(margin),
            }
        completed = min(batch_start + len(batch_rows), len(rows))
        if completed == len(rows) or completed % max(batch_size * 20, 1) == 0:
            print(f"[{task}] {completed}/{len(rows)}", flush=True)


def _records_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = np.asarray([bool(row["correct"]) for row in records], dtype=np.float64)
    valid = np.asarray([row["prediction"] is not None for row in records], dtype=np.float64)
    by_category: dict[str, Any] = {}
    categories = sorted({str(row["category"]) for row in records})
    for category in categories:
        subset = [row for row in records if str(row["category"]) == category]
        by_category[category] = {
            "sample_count": len(subset),
            "accuracy": float(np.mean([bool(row["correct"]) for row in subset])),
            "valid_choice_rate": float(
                np.mean([row["prediction"] is not None for row in subset])
            ),
        }
    return {
        "sample_count": len(records),
        "accuracy": float(correct.mean()),
        "valid_choice_rate": float(valid.mean()),
        "by_category": by_category,
    }


def run_benchmark(args: argparse.Namespace) -> None:
    import torch

    base_model = Path(args.base_model).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    mmstar_path = Path(args.mmstar).resolve()
    mmlu_path = Path(args.mmlu_pro).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.model_label):
        raise ValueError("model label may contain only letters, digits, dot, dash, underscore")
    for path in (mmstar_path, mmlu_path):
        if not path.is_file():
            raise FileNotFoundError(f"benchmark parquet is missing: {path}")
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"fine-tuned checkpoint is missing: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{args.model_label}.predictions.jsonl"
    summary_path = output_dir / f"{args.model_label}.summary.json"
    manifest_path = output_dir / f"{args.model_label}.manifest.json"
    for path in (prediction_path, summary_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite evaluation artifact: {path}")

    mmstar = pd.read_parquet(mmstar_path)
    mmstar = mmstar.rename(columns={"index": "item_id", "l2_category": "sub_category"})
    if args.mmstar_limit > 0 and args.mmstar_limit < len(mmstar):
        per_group = max(1, args.mmstar_limit // mmstar["category"].nunique())
        mmstar = stratified_sample(
            mmstar,
            group_column="category",
            per_group=per_group,
            seed=args.seed,
            id_column="item_id",
        ).head(args.mmstar_limit)
    mmstar_rows = mmstar.to_dict(orient="records")

    mmlu = pd.read_parquet(mmlu_path).rename(columns={"question_id": "item_id"})
    mmlu = stratified_sample(
        mmlu,
        group_column="category",
        per_group=args.mmlu_per_category,
        seed=args.seed,
        id_column="item_id",
    )
    mmlu_rows = mmlu.to_dict(orient="records")

    manifest = {
        "schema_version": 1,
        "model_label": args.model_label,
        "base_model": str(base_model),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_size": checkpoint_path.stat().st_size if checkpoint_path else None,
        "mmstar": {
            "path": str(mmstar_path),
            "sha256": _sha256(mmstar_path),
            "sample_count": len(mmstar_rows),
        },
        "mmlu_pro": {
            "path": str(mmlu_path),
            "sha256": _sha256(mmlu_path),
            "sample_count": len(mmlu_rows),
            "per_category": args.mmlu_per_category,
        },
        "seed": args.seed,
        "device": args.device,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "scoring_method": "constrained_next_token_choice_over_direct_and_space_prefixed_letter_tokens",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    started = time.time()
    model, processor = _load_model(
        base_model,
        checkpoint_path,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    task_specs = (
        ("mmstar_visual", mmstar_rows, args.visual_batch_size),
        ("mmstar_text_only", mmstar_rows, args.text_batch_size),
        ("mmlu_pro", mmlu_rows, args.text_batch_size),
    )
    summaries: dict[str, Any] = {}
    with prediction_path.open("w", encoding="utf-8") as stream:
        for task, rows, batch_size in task_specs:
            task_started = time.time()
            task_records = []
            for record in _generate_batches(
                model=model,
                processor=processor,
                rows=rows,
                task=task,
                batch_size=batch_size,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            ):
                record["model_label"] = args.model_label
                task_records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            summaries[task] = {
                **_records_summary(task_records),
                "elapsed_seconds": time.time() - task_started,
            }
            print(
                f"[{task}] accuracy={summaries[task]['accuracy']:.6f} "
                f"valid={summaries[task]['valid_choice_rate']:.6f}",
                flush=True,
            )
    summaries["total_elapsed_seconds"] = time.time() - started
    _atomic_text(summary_path, json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(summary_path)


def _read_predictions(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    tasks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task = str(row["task"])
            item_id = str(row["item_id"])
            if item_id in tasks[task]:
                raise ValueError(f"duplicate prediction for {task}/{item_id}: {path}")
            tasks[task][item_id] = row
    return tasks


def _fmt_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def summarize_benchmarks(args: argparse.Namespace) -> None:
    baseline_path = Path(args.baseline_predictions).resolve()
    treatment_path = Path(args.treatment_predictions).resolve()
    baseline = _read_predictions(baseline_path)
    treatment = _read_predictions(treatment_path)
    task_results: dict[str, Any] = {}
    for task in ("mmstar_visual", "mmstar_text_only", "mmlu_pro"):
        base_rows = baseline[task]
        tuned_rows = treatment[task]
        base_correct = {item_id: bool(row["correct"]) for item_id, row in base_rows.items()}
        tuned_correct = {item_id: bool(row["correct"]) for item_id, row in tuned_rows.items()}
        paired = paired_statistics(
            base_correct,
            tuned_correct,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        categories: dict[str, Any] = {}
        all_categories = sorted(
            {str(row["category"]) for row in base_rows.values()}
            | {str(row["category"]) for row in tuned_rows.values()}
        )
        for category in all_categories:
            base_category = {
                item_id: bool(row["correct"])
                for item_id, row in base_rows.items()
                if str(row["category"]) == category
            }
            tuned_category = {
                item_id: bool(row["correct"])
                for item_id, row in tuned_rows.items()
                if str(row["category"]) == category
            }
            categories[category] = paired_statistics(
                base_category,
                tuned_category,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
        task_results[task] = {**paired, "by_category": categories}

    multimodal_gain: dict[str, Any] = {}
    for label, payload in (("baseline", baseline), ("treatment", treatment)):
        visual = {
            item_id: bool(row["correct"])
            for item_id, row in payload["mmstar_visual"].items()
        }
        text_only = {
            item_id: bool(row["correct"])
            for item_id, row in payload["mmstar_text_only"].items()
        }
        multimodal_gain[label] = paired_statistics(
            text_only,
            visual,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    multimodal_gain["delta_gain"] = (
        multimodal_gain["treatment"]["delta_accuracy"]
        - multimodal_gain["baseline"]["delta_accuracy"]
    )

    result = {
        "schema_version": 1,
        "baseline_predictions": str(baseline_path),
        "treatment_predictions": str(treatment_path),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "tasks": task_results,
        "mmstar_multimodal_gain": multimodal_gain,
    }
    output = Path(args.output).resolve()
    _atomic_text(output, json.dumps(result, indent=2, sort_keys=True) + "\n")

    report = [
        "# VLM retention comparison",
        "",
        "Accuracy deltas are treatment minus baseline on exactly paired items.",
        "",
        "| Task | N | Baseline | 100k | Delta | Paired bootstrap 95% CI | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "mmstar_visual": "MMStar with image",
        "mmstar_text_only": "MMStar without image",
        "mmlu_pro": "MMLU-Pro text-only",
    }
    for task, label in labels.items():
        row = task_results[task]
        ci = row["delta_bootstrap_95_ci"]
        report.append(
            f"| {label} | {row['sample_count']} | {_fmt_percent(row['baseline_accuracy'])} | "
            f"{_fmt_percent(row['treatment_accuracy'])} | {_fmt_percent(row['delta_accuracy'])} | "
            f"[{_fmt_percent(ci[0])}, {_fmt_percent(ci[1])}] | {row['mcnemar_exact_p']:.6g} |"
        )
    report.extend(
        [
            "",
            "## MMStar multimodal gain",
            "",
            f"- Baseline image gain over text-only: {_fmt_percent(multimodal_gain['baseline']['delta_accuracy'])}",
            f"- 100k image gain over text-only: {_fmt_percent(multimodal_gain['treatment']['delta_accuracy'])}",
            f"- Change in multimodal gain: {_fmt_percent(multimodal_gain['delta_gain'])}",
            "",
            "## Category deltas",
            "",
        ]
    )
    for task, label in labels.items():
        report.append(f"### {label}")
        report.append("")
        report.append("| Category | N | Baseline | 100k | Delta |")
        report.append("|---|---:|---:|---:|---:|")
        for category, row in task_results[task]["by_category"].items():
            report.append(
                f"| {category} | {row['sample_count']} | {_fmt_percent(row['baseline_accuracy'])} | "
                f"{_fmt_percent(row['treatment_accuracy'])} | {_fmt_percent(row['delta_accuracy'])} |"
            )
        report.append("")
    report_path = output.with_suffix(".md")
    _atomic_text(report_path, "\n".join(report) + "\n")
    print(output)
    print(report_path)


def _read_pdms_csv(path: Path, metrics: Sequence[str]) -> dict[str, dict[str, float]]:
    result = {metric: {} for metric in metrics}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            token = str(row["token"])
            if token == "average":
                continue
            for metric in metrics:
                result[metric][token] = float(row[metric])
    return result


def summarize_pdms(args: argparse.Namespace) -> None:
    baseline_path = Path(args.baseline_csv).resolve()
    treatment_path = Path(args.treatment_csv).resolve()
    metrics = tuple(args.metrics)
    baseline = _read_pdms_csv(baseline_path, metrics)
    treatment = _read_pdms_csv(treatment_path, metrics)
    comparisons = {
        metric: paired_continuous_statistics(
            baseline[metric],
            treatment[metric],
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        for metric in metrics
    }
    payload = {
        "schema_version": 1,
        "baseline_label": args.baseline_label,
        "treatment_label": args.treatment_label,
        "baseline_csv": str(baseline_path),
        "treatment_csv": str(treatment_path),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "metrics": comparisons,
    }
    output = Path(args.output).resolve()
    _atomic_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = [
        "# Paired NAVSIM PDMS comparison",
        "",
        f"Treatment is `{args.treatment_label}`; baseline is `{args.baseline_label}`.",
        "",
        "| Metric | N | Baseline | Treatment | Delta | Paired bootstrap 95% CI | Improved / same / regressed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, row in comparisons.items():
        ci = row["delta_bootstrap_95_ci"]
        report.append(
            f"| {metric} | {row['sample_count']} | {row['baseline_mean']:.6f} | "
            f"{row['treatment_mean']:.6f} | {row['delta_mean']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['improved_count']} / "
            f"{row['unchanged_count']} / {row['regressed_count']} |"
        )
    report_path = output.with_suffix(".md")
    _atomic_text(report_path, "\n".join(report) + "\n")
    print(output)
    print(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    drift = subparsers.add_parser("inspect-drift")
    drift.add_argument("--base-model", required=True)
    drift.add_argument("--checkpoint", required=True)
    drift.add_argument("--output", required=True)
    drift.add_argument("--chunk-size", type=int, default=2_000_000)
    drift.set_defaults(function=inspect_drift)

    benchmark = subparsers.add_parser("run")
    benchmark.add_argument("--base-model", required=True)
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--model-label", required=True)
    benchmark.add_argument("--mmstar", required=True)
    benchmark.add_argument("--mmlu-pro", required=True)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--seed", type=int, default=20260818)
    benchmark.add_argument("--mmstar-limit", type=int, default=0)
    benchmark.add_argument("--mmlu-per-category", type=int, default=100)
    benchmark.add_argument("--visual-batch-size", type=int, default=4)
    benchmark.add_argument("--text-batch-size", type=int, default=16)
    benchmark.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    benchmark.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    benchmark.add_argument("--max-new-tokens", type=int, default=8)
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.set_defaults(function=run_benchmark)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--baseline-predictions", required=True)
    summary.add_argument("--treatment-predictions", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--seed", type=int, default=20260818)
    summary.add_argument("--bootstrap-samples", type=int, default=10_000)
    summary.set_defaults(function=summarize_benchmarks)

    pdms = subparsers.add_parser("summarize-pdms")
    pdms.add_argument("--baseline-csv", required=True)
    pdms.add_argument("--treatment-csv", required=True)
    pdms.add_argument("--baseline-label", required=True)
    pdms.add_argument("--treatment-label", required=True)
    pdms.add_argument("--output", required=True)
    pdms.add_argument("--seed", type=int, default=20260818)
    pdms.add_argument("--bootstrap-samples", type=int, default=10_000)
    pdms.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "score",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ],
    )
    pdms.set_defaults(function=summarize_pdms)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
