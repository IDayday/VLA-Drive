#!/usr/bin/env python3
"""Verify that skipping InternVL's lm_head preserves decoder hidden states."""

from __future__ import annotations

import argparse

import torch

from navsim.agents.EpisodeDrive.drivevla_backbone import DriveVLABackbone
from navsim.agents.EpisodeDrive.utils.internvl_tokenize import (
    build_internvl_model_inputs,
)
from navsim.agents.EpisodeDrive.utils.utils import build_drivevla_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Raw InternVL checkpoint directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(0)
    backbone = DriveVLABackbone(
        model_type="internvl",
        checkpoint_path=args.model,
        device=args.device,
        extra_token_count=8,
        target_vocab_size=151_682,
        use_flash_attn=True,
        initialize_from_config=False,
        skip_lm_head=False,
        gradient_checkpointing=False,
    ).eval()

    history = torch.zeros(1, 4, 3)
    command = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    questions = build_drivevla_questions(history, command)
    num_patches_list = [1]
    model_inputs = build_internvl_model_inputs(
        backbone.tokenizer,
        questions,
        num_patches_list,
        backbone.model.system_message,
        backbone.num_image_token,
    )
    pixel_values = torch.randn(
        1,
        3,
        448,
        448,
        device=args.device,
        dtype=torch.bfloat16,
    )

    lm_head_calls = 0

    def count_lm_head_calls(_module, _inputs, _output):
        nonlocal lm_head_calls
        lm_head_calls += 1

    hook = backbone.model.language_model.get_output_embeddings().register_forward_hook(
        count_lm_head_calls
    )
    with torch.inference_mode():
        reference = backbone(
            pixel_values,
            questions,
            num_patches_list,
            model_inputs=model_inputs,
        )
        reference_hidden_states = tuple(
            hidden.detach().clone() for hidden in reference.hidden_states
        )
        reference_lm_head_calls = lm_head_calls
        del reference

        lm_head_calls = 0
        backbone.skip_lm_head = True
        bypassed = backbone(
            pixel_values,
            questions,
            num_patches_list,
            model_inputs=model_inputs,
        )
        bypass_lm_head_calls = lm_head_calls

    hook.remove()

    if reference_lm_head_calls != 1:
        raise RuntimeError(
            f"Reference forward called lm_head {reference_lm_head_calls} times; expected 1"
        )
    if bypass_lm_head_calls != 0:
        raise RuntimeError(
            f"Bypassed forward called lm_head {bypass_lm_head_calls} times; expected 0"
        )
    if len(reference_hidden_states) != len(bypassed.hidden_states):
        raise RuntimeError(
            "Hidden-state layer count differs: "
            f"reference={len(reference_hidden_states)}, "
            f"bypass={len(bypassed.hidden_states)}"
        )

    maximum_error = 0.0
    for index, (reference_hidden, bypass_hidden) in enumerate(
        zip(reference_hidden_states, bypassed.hidden_states)
    ):
        if reference_hidden.shape != bypass_hidden.shape:
            raise RuntimeError(
                f"Layer {index} shape differs: "
                f"reference={tuple(reference_hidden.shape)}, "
                f"bypass={tuple(bypass_hidden.shape)}"
            )
        layer_error = float(
            (reference_hidden - bypass_hidden).abs().max().float().item()
        )
        maximum_error = max(maximum_error, layer_error)

    if maximum_error != 0.0:
        raise RuntimeError(
            f"lm_head bypass changed decoder hidden states; max_abs_error={maximum_error}"
        )

    print(
        "lm_head bypass verification: PASS\n"
        f"hidden_state_layers={len(reference_hidden_states)}\n"
        f"last_hidden_shape={tuple(reference_hidden_states[-1].shape)}\n"
        f"max_abs_error={maximum_error}\n"
        f"reference_lm_head_calls={reference_lm_head_calls}\n"
        f"bypass_lm_head_calls={bypass_lm_head_calls}"
    )


if __name__ == "__main__":
    main()
