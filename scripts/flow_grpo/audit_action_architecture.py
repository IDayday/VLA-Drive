"""Pre-download architecture check against previously extracted ZIP tensor metadata.

This does NOT verify tensor content or replace the strict real checkpoint gates.
"""
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.contracts import (
    inherit_freezing,
    optimizer_groups,
    parameter_manifest,
    apply_rl_freezes,
    check_manifest,
)
from starVLA.rl.flow_grpo.source_oracle import action_only_source
from starVLA.model.framework.QwenOFT import Qwenvl_OFT


def main():
    metadata = json.loads(
        Path("reports/ddp_flow_grpo/action_only_weight_metadata.json").read_text()
    )
    results = {}
    for variant in ("frozen_visual", "unfrozen_visual"):
        cfg, sft = resolve_config(f"configs/flow_grpo/action_only_{variant}.yaml")
        model = Qwenvl_OFT(sft)
        inherit_freezing(model, sft)
        source = parameter_manifest(model, optimizer_groups(model, sft))
        allowed = apply_rl_freezes(model, cfg)
        actor = parameter_manifest(model, optimizer_groups(model, sft))
        check_manifest(source, actor, allowed)
        name = next(
            n
            for n in metadata
            if n.startswith("action-only-" + variant.replace("_", "-") + "-")
        )
        actual = model.state_dict()
        assert actual.keys() == metadata[name].keys(), (
            actual.keys() - metadata[name].keys(),
            metadata[name].keys() - actual.keys(),
        )
        assert all(
            list(t.shape) == metadata[name][k]["shape"] for k, t in actual.items()
        )
        before = model._build_action_prompt_suffix()
        with action_only_source(model, cfg["checkpoint_contract"]):
            assert model._build_action_prompt_suffix() == before
        assert model._build_action_prompt_suffix() == before
        out = Path("reports/ddp_flow_grpo/architecture_" + variant)
        out.mkdir(exist_ok=True)
        (out / "source_manifest.json").write_text(json.dumps(source, indent=2))
        (out / "actor_manifest.json").write_text(json.dumps(actor, indent=2))
        results[variant] = dict(
            scope="architecture/metadata only; no action checkpoint tensors loaded",
            state_tensors=len(actual),
            total_numel=sum(p.numel() for p in model.parameters()),
            source_trainable=source["trainable_numel"],
            actor_trainable=actor["trainable_numel"],
            source_method_binding_valid=True,
        )
        print(variant, results[variant], flush=True)
        del model, actual
        torch.cuda.empty_cache()
    Path("reports/ddp_flow_grpo/action_architecture.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
