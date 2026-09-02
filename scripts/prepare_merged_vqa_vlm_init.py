#!/usr/bin/env python3
"""Normalize a dense Driving-VQA VLM or merge a PEFT adapter exactly once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import (  # noqa: E402
    audit_vlm_checkpoint,
    discover_weight_files,
    scan_forbidden_state_keys,
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
    "generation_config.json",
)
RUNTIME_FILES = (
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "conversation.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "preprocessor_config.json",
)


def classify_vqa_checkpoint(path: Path) -> str:
    """Return ``peft`` or ``dense`` and reject ambiguous/missing layouts."""
    path = Path(path)
    has_adapter = (path / "adapter_config.json").is_file()
    dense_weights = discover_weight_files(path)
    if has_adapter and dense_weights:
        raise RuntimeError(
            "Ambiguous VQA checkpoint contains both PEFT metadata and dense weights"
        )
    if has_adapter:
        return "peft"
    if dense_weights:
        return "dense"
    raise FileNotFoundError(
        f"VQA path is neither a dense VLM nor a PEFT adapter: {path}"
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _fixed_forward(model, tokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    image_size = int(model.config.vision_config.image_size)
    dtype = next(model.parameters()).dtype
    pixels = torch.linspace(
        -1.0,
        1.0,
        steps=3 * image_size * image_size,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, image_size, image_size).to(dtype=dtype)
    context_id = int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
    start_id = int(tokenizer.convert_tokens_to_ids("<img>"))
    end_id = int(tokenizer.convert_tokens_to_ids("</img>"))
    bos_id_value = tokenizer.bos_token_id
    if bos_id_value is None:
        bos_id_value = tokenizer.eos_token_id
    if bos_id_value is None:
        bos_id_value = 0
    bos_id = int(bos_id_value)
    input_ids = torch.tensor(
        [[bos_id, start_id] + [context_id] * int(model.num_image_token) + [end_id]],
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    image_flags = torch.ones((1, 1), device=device, dtype=torch.long)
    model.img_context_token_id = context_id
    with torch.no_grad():
        outputs = model(
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
        )
    return {
        "logits_left": outputs.logits[:, :, :64].float().cpu(),
        "logits_right": outputs.logits[:, :, -64:].float().cpu(),
        "hidden": outputs.hidden_states[-1][:, :, :64].float().cpu(),
    }


def _compare_forward(
    before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]
) -> Tuple[float, Dict[str, float]]:
    differences = {
        key: float((before[key] - after[key]).abs().max().item())
        for key in before
    }
    return max(differences.values(), default=0.0), differences


def _load_vlm(path: Path, device: torch.device):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    config.vision_config.use_flash_attn = False
    return AutoModel.from_pretrained(
        str(path),
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=str(device),
        local_files_only=True,
        use_flash_attn=False,
    )


def _copy_normalized_dense(
    source: Path, output: Path, runtime_source: Path
) -> None:
    for name in RUNTIME_FILES:
        candidate = runtime_source / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)
    for name in TOKENIZER_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["planreg_formal_provenance"] = {
        "source_variant": "driving_vqa",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source / "model.safetensors"),
        "adapter_merged": False,
        "dense_vlm_source": True,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for weight in discover_weight_files(source):
        _link_or_copy(weight, output / weight.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Corresponding Base InternVL")
    parser.add_argument("--vqa", required=True, help="Dense VQA VLM or PEFT directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--runtime-source",
        help="Canonical trust_remote_code source (defaults to --base)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-forward-parity", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    vqa = Path(args.vqa).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    runtime_source = Path(args.runtime_source or args.base).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite VQA initialization: {output}")

    is_peft = classify_vqa_checkpoint(vqa) == "peft"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    before_forward = None
    after_forward = None
    merged_module_count = 0
    adapter_sha = None
    try:
        if is_peft:
            from peft import PeftModel
            from transformers import AutoTokenizer

            adapter_weights = tuple(vqa.glob("adapter_model*.safetensors"))
            if not adapter_weights:
                raise FileNotFoundError("PEFT VQA directory lacks adapter safetensors")
            adapter_sha = sha256_file(adapter_weights[0])
            device = torch.device(args.device)
            base_model = _load_vlm(base, device)
            tokenizer_source = vqa if (vqa / "tokenizer_config.json").exists() else base
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_source),
                trust_remote_code=True,
                use_fast=False,
                local_files_only=True,
            )
            peft_model = PeftModel.from_pretrained(base_model, str(vqa), is_trainable=False)
            merged_module_count = sum(
                1 for _, child in peft_model.named_modules() if hasattr(child, "merge")
            )
            if not args.skip_forward_parity:
                before_forward = _fixed_forward(peft_model, tokenizer, device)
            merged = peft_model.merge_and_unload(safe_merge=True)
            merged.save_pretrained(str(temporary), safe_serialization=True)
            tokenizer.save_pretrained(str(temporary))
            for name in RUNTIME_FILES:
                candidate = runtime_source / name
                if candidate.is_file():
                    shutil.copy2(candidate, temporary / name)
            config_path = temporary / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["planreg_formal_provenance"] = {
                "source_variant": "driving_vqa",
                "base_checkpoint": str(base),
                "adapter_checkpoint": str(vqa),
                "adapter_merged": True,
                "merged_module_count": merged_module_count,
            }
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            del peft_model, base_model, merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            _copy_normalized_dense(vqa, temporary, runtime_source)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    audit = audit_vlm_checkpoint(str(output), variant="driving_vqa")
    if not args.skip_forward_parity:
        from transformers import AutoTokenizer

        device = torch.device(args.device)
        if before_forward is None:
            source_model = _load_vlm(vqa, device)
            source_tokenizer = AutoTokenizer.from_pretrained(
                str(vqa), trust_remote_code=True, use_fast=False, local_files_only=True
            )
            before_forward = _fixed_forward(source_model, source_tokenizer, device)
            del source_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        output_model = _load_vlm(output, device)
        output_tokenizer = AutoTokenizer.from_pretrained(
            str(output), trust_remote_code=True, use_fast=False, local_files_only=True
        )
        after_forward = _fixed_forward(output_model, output_tokenizer, device)
        del output_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        max_abs_diff, component_differences = _compare_forward(
            before_forward, after_forward
        )
        if max_abs_diff > 1e-5:
            raise RuntimeError(
                "VQA merge/normalization forward parity failed: "
                f"max_abs_diff={max_abs_diff}, components={component_differences}"
            )
    else:
        max_abs_diff = None
        component_differences = {}

    report = {
        "schema_version": 1,
        "source_vqa_path": str(vqa),
        "base_path": str(base),
        "output_path": str(output),
        "vqa_checkpoint_kind": "peft" if is_peft else "dense",
        "adapter_merge_performed": is_peft,
        "adapter_sha256": adapter_sha,
        "base_checkpoint_sha256": audit_vlm_checkpoint(
            str(base), variant="base"
        )["checkpoint_sha256"],
        "output_checkpoint_sha256": audit["checkpoint_sha256"],
        "merged_module_count": merged_module_count,
        "fixed_forward_max_abs_diff": max_abs_diff,
        "fixed_forward_component_max_abs_diff": component_differences,
        "forbidden_agent_state_detected": False,
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
