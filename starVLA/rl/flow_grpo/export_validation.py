"""Compare exported predictions against raw training weights in the original agent."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from .config import split_tokens
from .data import KeyedDataset


def verify_export(cfg, sft, checkpoint, exported, output):
    from infer import VLAAgent, set_inference_seed

    checkpoint, exported, output = map(Path, (checkpoint, exported, output))
    if not all((p / "COMPLETE").is_file() for p in (checkpoint, exported)):
        raise ValueError("require complete training and exported checkpoints")
    os.environ["BASE_VLM"] = cfg["paths"]["base_vlm"]
    os.environ["VLM_ATTN_IMPLEMENTATION"] = sft.framework.qwenvl.attn_implementation
    agent = VLAAgent(
        str(exported),
        device="cuda",
        qwen_forward_mode=(
            "optimized"
            if cfg["checkpoint_contract"]["policy_feature_output"] == "normalized"
            else "legacy"
        ),
    )
    _, tokens = split_tokens(cfg)
    sample = KeyedDataset(sft)[(tokens[0], cfg["runtime"]["seed"])]

    def prediction():
        set_inference_seed(cfg["runtime"]["seed"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return agent.predict([sample])["normalized_actions"]

    after = prediction()
    if (checkpoint / "pytorch_model.bin").is_file():
        raw = torch.load(
            checkpoint / "pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
    else:
        with torch.serialization.safe_globals([set]):
            raw = torch.load(
                next(checkpoint.glob("*/mp_rank_00_model_states.pt")),
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )["module"]
    raw = {key.removeprefix("policy."): value for key, value in raw.items()}
    flat = torch.load(
        exported / "pytorch_model.pt", weights_only=True, map_location="cpu", mmap=True
    )
    if raw.keys() != flat.keys():
        raise AssertionError("export changed model keys")
    for name in raw:
        if raw[name].dtype != flat[name].dtype or not torch.equal(
            raw[name], flat[name]
        ):
            raise AssertionError("export changed saved tensor: " + name)
    result = agent.model.load_state_dict(raw, strict=False)
    if result.missing_keys or any(
        not key.startswith("rgb_model.") for key in result.unexpected_keys
    ):
        raise AssertionError("original agent cannot consume raw actor state")
    before = prediction()
    np.testing.assert_array_equal(after, before)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="TESTED",
        checkpoint=str(checkpoint),
        exported=str(exported),
        original_evaluator="infer.VLAAgent",
        tensors_identical=len(raw),
        scene=tokens[0],
        output_max_abs=float(np.max(np.abs(after - before))),
        protocol="same original inference dtype, seed, observation, and ODE steps",
    )
    (output / "export_equivalence.json").write_text(json.dumps(report, indent=2))
    np.savez(output / "export_predictions.npz", raw_training=before, exported=after)
    print(json.dumps(report), flush=True)
    return report
