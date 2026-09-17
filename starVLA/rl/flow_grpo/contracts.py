"""Canonical, object-deduplicated SFT parameter and optimizer contracts."""
from collections import defaultdict
import hashlib
import json
import torch


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def inherit_freezing(model, config):
    """Retain constructor freezes, then apply exactly the checkpoint trainer paths."""
    for path in config.trainer.get("freeze_modules", "").split(","):
        if path.strip():
            try:
                module = model.get_submodule(path.strip())
            except AttributeError:
                # Same conditional construction as SFT (text_input=0 omits T5).
                if (
                    path.strip() == "rgb_model.text_encoder"
                    and not config.datasets.video_data.text_input
                ):
                    continue
                raise ValueError(f"SFT freeze module missing: {path.strip()}")
            module.requires_grad_(False)


def optimizer_groups(model, config, multiplier=1.0):
    # Importing trainer_tools has no Accelerator initialization. Never import train_starvla.
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    if "qwen_visual" in config.trainer.learning_rate:
        from .action_only_optimizer import build_param_lr_groups

    groups = build_param_lr_groups(model, config)
    for group in groups:
        group["lr"] = float(group["lr"]) * multiplier
        group["weight_decay"] = float(config.trainer.optimizer.weight_decay)
    check_optimizer(model, groups)
    return groups


def check_optimizer(model, groups):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    seen = []
    for group in groups:
        if float(group["lr"]) <= 0:
            raise ValueError(f'nonpositive learning rate in {group.get("name")}')
        seen.extend(id(p) for p in group["params"])
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate optimizer parameter (including tied aliases)")
    if set(seen) != expected:
        raise ValueError(
            f"optimizer coverage mismatch: missing={len(expected-set(seen))}, extra={len(set(seen)-expected)}"
        )


def parameter_manifest(
    model, groups, gradient_sources=None, gradient_graph_sources=None
):
    aliases = defaultdict(list)
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases[id(p)].append(name)
    group_ids = {
        id(p): g.get("name", str(i)) for i, g in enumerate(groups) for p in g["params"]
    }
    entries = []
    for name, p in model.named_parameters():
        source = (gradient_sources or {}).get(name, [])
        category = "frozen" if not p.requires_grad else "unverified_trainable"
        if gradient_sources is not None and p.requires_grad:
            category = (
                "policy"
                if "rl" in source
                else "auxiliary_only"
                if "auxiliary" in source
                else "sft_only"
                if "sft" in source
                else "zero_gradient_observed"
                if (gradient_graph_sources or {}).get(name)
                else "sft_dormant"
            )
        entries.append(
            dict(
                name=name,
                aliases=aliases[id(p)],
                shape=list(p.shape),
                dtype=str(p.dtype),
                numel=p.numel(),
                requires_grad=p.requires_grad,
                module=name.rsplit(".", 1)[0],
                optimizer_group=group_ids.get(id(p)),
                category=category,
                gradient_sources=source,
                gradient_graph_sources=(gradient_graph_sources or {}).get(name, []),
            )
        )
    return {
        "schema": 1,
        "parameters": entries,
        "trainable_numel": sum(e["numel"] for e in entries if e["requires_grad"]),
    }


def apply_rl_freezes(model, config):
    """Only the user's explicit post-SFT visual freeze is an allowed exception."""
    paths = config.get("rl_freeze_modules", [])
    if set(paths) - {"qwen_vl_interface.model.visual"}:
        raise ValueError(
            "only the explicitly authorized visual tower freeze is allowed"
        )
    parameter_ids = set()
    for path in paths:
        module = model.get_submodule(path)
        parameter_ids.update(id(parameter) for parameter in module.parameters())
        module.requires_grad_(False)
    # Qwen exposes .visual as a property; its canonical names contain .model.visual.
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in parameter_ids
    )


def check_manifest(sft, actor, allowed_freezes=()):
    keys = ("name", "aliases", "shape", "dtype", "requires_grad", "optimizer_group")
    if len(sft["parameters"]) != len(actor["parameters"]):
        raise ValueError("actor parameters differ from the checkpoint SFT contract")
    for original, current in zip(sft["parameters"], actor["parameters"]):
        permitted = original["name"] in allowed_freezes
        for key in keys:
            if key in ("requires_grad", "optimizer_group") and permitted:
                if current["requires_grad"] or current["optimizer_group"] is not None:
                    raise ValueError("authorized visual freeze was not applied")
                continue
            if original[key] != current[key]:
                raise ValueError(
                    f"actor parameters differ from the checkpoint SFT contract: {original['name']} {key}"
                )


def grad_summary(model):
    summary = {}
    for name, p in model.named_parameters():
        root = name.split(".")[0]
        entry = summary.setdefault(
            root,
            {
                "trainable": 0,
                "grad_tensors": 0,
                "nonzero_tensors": 0,
                "squared_norm": 0.0,
            },
        )
        entry["trainable"] += int(p.requires_grad)
        if p.grad is not None:
            g = p.grad.detach().float()
            if not torch.isfinite(g).all():
                raise FloatingPointError(f"nonfinite gradient: {name}")
            norm = float(g.square().sum())
            entry["grad_tensors"] += 1
            entry["nonzero_tensors"] += int(norm > 0)
            entry["squared_norm"] += norm
    return summary
