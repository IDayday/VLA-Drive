# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration

def _init_rows_by_token_ids(model, token_ids: List[int], init_strategy: str, ref_embed: torch.Tensor):
    """
    Initialize embedding rows for specific token_ids (even if they are in a reserved range).
    ref_embed: old_embed.weight used for avg reference, etc.
    """
    emb = model.get_input_embeddings().weight
    device = emb.device
    dtype = emb.dtype

    # filter valid ids
    token_ids = [tid for tid in token_ids if isinstance(tid, int) and 0 <= tid < emb.shape[0]]
    if not token_ids:
        return

    with torch.no_grad():
        if init_strategy == "avg":
            ref_vec = ref_embed.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
            emb[token_ids] = ref_vec.expand(len(token_ids), -1)
            # 强烈建议加噪声，防止精确相等/对称性
            emb[token_ids] += (1e-3 * torch.randn_like(emb[token_ids]))
        elif init_strategy == "zero":
            emb[token_ids].zero_()
            # 同样建议加噪声，否则全部一样
            emb[token_ids] += (1e-3 * torch.randn_like(emb[token_ids]))
        elif init_strategy == "normal":
            # 用 float32 init 再 cast 回去更稳
            tmp = torch.empty((len(token_ids), emb.shape[1]), device=device, dtype=torch.float32)
            torch.nn.init.normal_(tmp, mean=0.0, std=0.02)
            emb[token_ids] = tmp.to(dtype=dtype)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

def add_new_tokens(
    model,
    tokenizer,
    new_tokens: List[str],
    init_strategy: str = "avg",
    as_special: bool = True,
        ) -> Tuple[Dict[str, int], int, int, int]:
    """
    Add new tokens into the model and tokenizer (if they don't already exist).
    init_strategy: avg / normal / zero
    Returns:
      - mapping: token_id mapping for all target tokens
      - added_now: number of tokens actually added to the tokenizer this time
      - action_token_start_idx: start index of newly added embeddings (based on old embedding size)
      - action_token_end_idx: end index of newly added embeddings (if none added, equals start_idx - 1)
    Notes:
      - tokenizer.vocab_size is the base vocabulary size (excluding already added special/added tokens)
      - len(tokenizer) is the total vocabulary size (including added/special tokens)
      - The old embedding size of the model is model.get_input_embeddings().weight.shape[0]
    """
    # 1) Compute tokens to add (relative to current tokenizer vocab)
    vocab = tokenizer.get_vocab()  # includes existing special tokens
    to_add_tokens = [t for t in new_tokens if t not in vocab]

    # 2) Record current embedding size of the model (base size)
    old_embed = model.get_input_embeddings()
    old_embed_size = old_embed.weight.shape[0]  # includes Qwen reserved tokens

    # 3) If needed, add tokens into tokenizer first
    added_now = 0
    if to_add_tokens:
        if as_special:
            added_now = tokenizer.add_special_tokens({"additional_special_tokens": to_add_tokens})
        else:
            added_now = tokenizer.add_tokens(to_add_tokens)

    # 4) Target total size (base + newly added)
    # target_size = len(tokenizer) # total vocab --> whether to keep previously reserved empty tokens?
    target_size = old_embed_size + added_now
    # 5) If tokenizer total size exceeds model embedding size, resize and init new rows
    action_token_start_idx = old_embed_size  # no-reserve plan here
    action_token_end_idx = old_embed_size - 1  # default: "no additions"
    if target_size > old_embed_size:
        model.resize_token_embeddings(target_size)  # resizing to target size
        new_embed = model.get_input_embeddings()
        with torch.no_grad():
            if init_strategy == "avg":
                ref_vec = old_embed.weight.mean(dim=0, keepdim=True)
                for idx in range(old_embed_size, target_size):
                    new_embed.weight[idx].copy_(ref_vec[0])
            elif init_strategy == "zero":
                for idx in range(old_embed_size, target_size):
                    new_embed.weight[idx].zero_()
            elif init_strategy == "normal":
                for idx in range(old_embed_size, target_size):
                    nn.init.normal_(new_embed.weight[idx], mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init_strategy: {init_strategy}")

        action_token_end_idx = target_size - 1

    # 6) Build mapping (return ids for requested tokens)
    mapping = {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}

    # ✅ NEW: init the actual rows for tokens that were just added now (even if in reserved range)
    # 只初始化这次新增的那些 token（to_add_tokens），避免破坏原词表
    added_token_ids = [mapping[t] for t in to_add_tokens]
    _init_rows_by_token_ids(model, added_token_ids, init_strategy, old_embed.weight)

    # ✅ NEW: 额外防御：如果有人仍然等于 image token embedding，强制再 normal 一次
    img_id = getattr(model.config, "image_token_id", None)
    if img_id is not None:
        emb = model.get_input_embeddings().weight
        img_vec = emb[img_id]
        bad = [tid for tid in added_token_ids if (emb[tid] == img_vec).all().item()]
        if bad:
            print(f"[WARN] {len(bad)} tokens collide with <image> embedding. Re-init with normal: {bad[:10]}")
            _init_rows_by_token_ids(model, bad, "normal", old_embed.weight)
    
    return mapping, added_now, action_token_start_idx, action_token_end_idx

def save_bundle(model, tokenizer, mapping: Dict[str, int], save_dir: str, processor_src: str | None = None, padding_side: str | None = None):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "added_custom_token_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved to: {save_dir}")

    # Additionally save AutoProcessor (generate preprocessor_config.json) so AutoProcessor.from_pretrained(...) can load
    try:
        src = processor_src or save_dir
        processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        # Sync processor.tokenizer 
        processor.tokenizer = tokenizer
        processor.save_pretrained(save_dir)
        print(f"[OK] AutoProcessor saved to: {save_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save AutoProcessor: {e}")

def reload_and_check(save_dir: str, tokens: List[str]) -> bool:
    tok = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    vocab = tok.get_vocab()
    missing = [t for t in tokens if t not in vocab]
    if missing:
        print(f"[WARN] Still missing after reload: {missing}")
        return False
    print("[OK] Reload check passed, all tokens exist.")
    return True

def parse_tokens(args) -> List[str]:
    tokens: List[str] = []
    if args.tokens:
        tokens.extend([t.strip() for t in args.tokens.split(",") if t.strip()])
    if args.tokens_file:
        with open(args.tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.append(line)
    # De-duplicate while keeping order
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def main():
    parser = argparse.ArgumentParser(
        description="Add special tokens to Qwen2.5-VL model and save to local directory."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF Hub model ID or local path")
    parser.add_argument("--save-dir", required=True, help="Output directory to save")
    parser.add_argument("--tokens", default="", help="Comma-separated tokens, e.g., <loc_x>,<loc_y>")
    parser.add_argument("--tokens-file", help="Text file containing tokens to add (one per line)")
    parser.add_argument("--init-strategy", default="avg", choices=["avg", "normal", "zero"], help="Initialization strategy for newly added embeddings")
    parser.add_argument("--as-special", action="store_true", help="Whether to add as special tokens")
    parser.add_argument("--no-as-special", dest="as_special", action="store_false")
    parser.set_defaults(as_special=True)
    parser.add_argument("--padding-side", default="left", choices=["left", "right"])
    parser.add_argument("--device", default="cuda", help="cuda / cpu / mps / auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention backend. Use sdpa for the PPU CUDA-compatible runtime.",
    )
    args = parser.parse_args()


    tokens = parse_tokens(args)
    if not tokens:
        print("No tokens provided, use --tokens or --tokens-file")
        return

    print(f"[INFO] Tokens to process: {tokens}")

    print(f"[INFO] Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model_id,
    #     torch_dtype="auto",
    #     device_map="auto" if args.device == "auto" else None,
    #     trust_remote_code=True,
    # )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
        device_map="auto" if args.device == "auto" else args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"


    # Print sizes for diagnosis
    base_tok_size = tokenizer.vocab_size                  # base vocab size
    total_tok_size = len(tokenizer)                       # total vocab size
    model_embed_size = model.get_input_embeddings().weight.shape[0]  # current model embedding size
    print(f"[DEBUG] tokenizer.vocab_size(base) = {base_tok_size}")
    print(f"[DEBUG] len(tokenizer)(total)     = {total_tok_size}")
    print(f"[DEBUG] model.embed_size(before)  = {model_embed_size}")
    print(f"[DEBUG] added_in_tokenizer        = {total_tok_size - base_tok_size}")

    mapping, added, action_token_start_idx, action_token_end_idx = add_new_tokens(
        model=model,
        tokenizer=tokenizer,
        new_tokens=tokens,
        init_strategy=args.init_strategy,
        as_special=args.as_special,
    )
    new_model_embed_size = model.get_input_embeddings().weight.shape[0]

    emb = model.get_input_embeddings().weight
    img_id = model.config.image_token_id
    img_vec = emb[img_id]
    bad = []
    for t in tokens:  # or to_add_tokens
        tid = tokenizer.convert_tokens_to_ids(t)
        if (emb[tid] == img_vec).all().item():
            bad.append((t, tid))
    print("collide-with-image:", bad[:10], "num=", len(bad))
    assert len(bad) == 0


    save_bundle(model, tokenizer, mapping, args.save_dir, processor_src=args.model_id, padding_side=args.padding_side)

    # Re-validate
    reload_and_check(args.save_dir, tokens)

    print(f"[INFO] Newly added to tokenizer: {added}")
    # print(f"[INFO] Token mapping: {mapping}")
    print(f"[INFO] Action token idx range: [{action_token_start_idx}, {action_token_end_idx}]")
    print(f"[DEBUG] model.embed_size(after)   = {new_model_embed_size}")



def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    # start_debugpy_once()
    main()
