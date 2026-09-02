#!/usr/bin/env python3
"""Prepare an explicitly tokenizer-aligned, otherwise raw Base InternVL VLM.

The Driving-VQA checkpoint adds eight tokenizer rows.  Formal paired training
cannot silently resize the Base model at runtime, so this one-time preparation
copies all public Base tensors and initializes only those eight new embedding
and lm-head rows to their respective per-dimension Base means.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import (  # noqa: E402
    audit_vlm_checkpoint,
    sha256_file,
)


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
RUNTIME_FILES = (
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "conversation.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "preprocessor_config.json",
)
EMBEDDING_KEYS = (
    "language_model.model.embed_tokens.weight",
    "language_model.lm_head.weight",
)


def _copy_if_present(source: Path, destination: Path, name: str) -> None:
    candidate = source / name
    if candidate.is_file():
        shutil.copy2(candidate, destination / name)


def _expand_rows(value: torch.Tensor, target_rows: int) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] >= target_rows:
        raise ValueError(
            f"Expected a 2-D embedding with fewer than {target_rows} rows, "
            f"got {tuple(value.shape)}"
        )
    additional = target_rows - value.shape[0]
    mean_row = value.mean(dim=0, dtype=torch.float32).to(dtype=value.dtype)
    return torch.cat((value, mean_row.unsqueeze(0).repeat(additional, 1)), dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tokenizer-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    tokenizer_source = Path(args.tokenizer_source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared Base VLM directory: {output}"
        )
    source_weight = base / "model.safetensors"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)

    from transformers import AutoTokenizer

    source_tokenizer = AutoTokenizer.from_pretrained(
        str(base), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    aligned_tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    old_vocab = source_tokenizer.get_vocab()
    new_vocab = aligned_tokenizer.get_vocab()
    changed_existing = {
        token: {"base": index, "aligned": new_vocab.get(token)}
        for token, index in old_vocab.items()
        if new_vocab.get(token) != index
    }
    if changed_existing:
        raise RuntimeError(
            "Tokenizer alignment would change existing public Base token IDs: "
            f"{dict(list(changed_existing.items())[:20])}"
        )
    added_tokens = {
        token: index for token, index in new_vocab.items() if token not in old_vocab
    }
    if len(added_tokens) != len(new_vocab) - len(old_vocab):
        raise RuntimeError("Tokenizer alignment contains an unexplained vocabulary change")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        for name in RUNTIME_FILES:
            _copy_if_present(base, temporary, name)
        for name in TOKENIZER_FILES:
            _copy_if_present(tokenizer_source, temporary, name)
        _copy_if_present(tokenizer_source, temporary, "generation_config.json")

        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        config["llm_config"]["vocab_size"] = len(aligned_tokenizer)
        config["planreg_formal_provenance"] = {
            "source_variant": "public_base_no_driving_vqa",
            "source_checkpoint": str(base),
            "source_checkpoint_sha256": sha256_file(source_weight),
            "tokenizer_alignment_source": str(tokenizer_source),
            "added_token_count": len(added_tokens),
            "added_token_initialization": "per_dimension_mean_of_public_base_rows",
            "runtime_resize_permitted": False,
        }
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        tensors = load_file(str(source_weight), device="cpu")
        missing = [key for key in EMBEDDING_KEYS if key not in tensors]
        if missing:
            raise RuntimeError(f"Base VLM lacks expected embedding tensors: {missing}")
        original_rows = int(tensors[EMBEDDING_KEYS[0]].shape[0])
        for key in EMBEDDING_KEYS:
            tensors[key] = _expand_rows(tensors[key], len(aligned_tokenizer))
        with safe_open(str(source_weight), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        prepared_weight = temporary / "model.safetensors"
        save_file(tensors, str(prepared_weight), metadata=metadata)
        del tensors

        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    audit = audit_vlm_checkpoint(str(output), variant="base")
    report = {
        "schema_version": 1,
        "operation": "explicit_base_tokenizer_alignment",
        "source_base_path": str(base),
        "source_base_checkpoint_sha256": sha256_file(source_weight),
        "tokenizer_source_path": str(tokenizer_source),
        "output_path": str(output),
        "output_checkpoint_sha256": audit["checkpoint_sha256"],
        "output_config_sha256": audit["config_sha256"],
        "output_tokenizer_sha256": audit["tokenizer_sha256"],
        "output_tokenizer_vocab_sha256": audit["tokenizer_vocab_sha256"],
        "original_vocab_size": original_rows,
        "output_vocab_size": audit["vocab_size"],
        "added_tokens": dict(sorted(added_tokens.items(), key=lambda item: item[1])),
        "unchanged_tensor_policy": "all non-embedding tensors copied byte-for-byte",
        "agent_checkpoint_loaded": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
