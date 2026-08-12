#!/usr/bin/env python3
"""Validate the local Qwen bundle used by the VGGT-query planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def load_tokens(path: Path, expected_count: int) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing VGGT token contract: {path}")
    tokens = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(tokens) != expected_count or len(set(tokens)) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} unique VGGT query tokens, found {len(tokens)}"
        )
    return tokens


def validate(model_dir: Path, tokens_file: Path, expected_count: int) -> dict[str, object]:
    required = ("config.json", "tokenizer.json", "added_custom_token_id_map.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    weight_files = tuple(model_dir.glob("*.safetensors")) + tuple(model_dir.glob("*.bin"))
    if missing or not weight_files:
        details = missing + ([] if weight_files else ["model weights (*.safetensors or *.bin)"])
        raise FileNotFoundError(f"Incomplete VGGT-token VLM {model_dir}: missing {details}")

    tokens = load_tokens(tokens_file, expected_count)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    vocabulary = tokenizer.get_vocab()
    missing_tokens = [token for token in tokens if token not in vocabulary]
    if missing_tokens:
        raise RuntimeError(f"VGGT-token VLM is missing tokens: {missing_tokens[:5]}")
    token_ids = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in tokens)
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(
            f"VGGT query tokens do not map to {expected_count} unique token IDs"
        )

    recorded = json.loads((model_dir / "added_custom_token_id_map.json").read_text(encoding="utf-8"))
    mismatched = [token for token, token_id in zip(tokens, token_ids) if recorded.get(token) != token_id]
    if mismatched:
        raise RuntimeError(f"Recorded VGGT token IDs disagree with the tokenizer: {mismatched[:5]}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    hidden_size = int(text_config.get("hidden_size", -1))
    if hidden_size != 2048:
        raise RuntimeError(f"Expected Qwen hidden size 2048, found {hidden_size}")

    return {
        "status": "PASS",
        "model_dir": str(model_dir.resolve()),
        "query_tokens": len(tokens),
        "unique_token_ids": len(set(token_ids)),
        "hidden_size": hidden_size,
        "weight_files": len(weight_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=15)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.model_dir, args.tokens_file, args.expected_count),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
