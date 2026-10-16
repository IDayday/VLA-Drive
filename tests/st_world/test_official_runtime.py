"""CPU behavioral tests; these do not grant CUDA/FA2/ZeRO qualification."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.st_world_policy import official_runtime as rt


def request(**overrides):
    return {"schema": rt.SCHEMA, "preset": rt.PRESET, "scope": "official_reference",
            "adaptation_reason": "", "declared_recipe_overrides": [], **overrides}


def ingredients():
    r = deepcopy(rt.REFERENCE_RECIPE)
    return ({k: r[k] for k in ("world_size", "micro_batch", "global_batch", "max_updates")},
            {k: r[k] for k in ("lr", "minimum_lr", "warmup", "betas", "eps", "weight_decay", "clip")})


def policy():
    e, o = ingredients()
    return rt.resolve_official_execution(request(), execution=e, optimizer=o, gradient_checkpointing=False)


def test_reference_exact_and_all_backend_fields_consumed():
    p = policy()
    assert p["reference_recipe"] == p["effective_recipe"]
    assert p["checkpointing_evidence"]["official_yaml_flag"] is True
    assert p["effective_recipe"]["gradient_checkpointing"] is False
    ds = rt.official_deepspeed_config(p, accumulation_owner="deepspeed")
    assert ds["bf16"] == {"enabled": True} and ds["fp16"] == {"enabled": False}
    assert ds["zero_optimization"] == {"stage": 2, "allgather_partitions": True,
        "allgather_bucket_size": 500000000, "reduce_scatter": True, "reduce_bucket_size": 500000000,
        "overlap_comm": True, "contiguous_gradients": True, "cpu_offload": False}
    assert "optimizer" not in ds  # DS must retain the caller's actual Torch optimizer.
    assert (ds["train_batch_size"], ds["train_micro_batch_size_per_gpu"], ds["gradient_accumulation_steps"]) == (32, 4, 1)
    assert p["gpu_qualified"] is False
    assert policy() == p


def test_twenty_k_budget_and_registered_checkpointing_are_explicit_adaptations():
    e, o = ingredients(); e["max_updates"] = 20000; o["warmup"] = 1000
    r = request(scope="research_stage", adaptation_reason="Fixed pre-existing ST stage budget",
                declared_recipe_overrides=["warmup", "max_updates", "gradient_checkpointing"])
    original = deepcopy((r, e, o))
    p = rt.resolve_official_execution(r, execution=e, optimizer=o, gradient_checkpointing=True)
    assert (r, e, o) == original
    assert p["effective_recipe"]["max_updates"] == 20000
    assert p["effective_recipe"]["warmup"] == 1000
    assert p["recipe_overrides"]["gradient_checkpointing"] == {"reference": False, "effective": True}
    assert p["reference_recipe"]["max_updates"] == 100000


@pytest.mark.parametrize("change", [
    {"unknown": 1}, {"preset": "sdpa_fallback"}, {"schema": "old"}, {"scope": "formal"},
    {"declared_recipe_overrides": ["lr"]}, {"declared_recipe_overrides": ["lr", "lr"]},
    {"scope": "research_stage", "adaptation_reason": ""},
])
def test_unknown_or_unjustified_policy_rejected(change):
    e, o = ingredients()
    with pytest.raises(ValueError):
        rt.resolve_official_execution(request(**change), execution=e, optimizer=o, gradient_checkpointing=False)


@pytest.mark.parametrize("key,value", [("lr", float("nan")), ("betas", [True, .95]),
    ("weight_decay", -1), ("warmup", 100000), ("minimum_lr", 1), ("eps", 0)])
def test_invalid_optimizer_recipe_rejected(key, value):
    e, o = ingredients(); o[key] = value
    with pytest.raises(ValueError):
        rt.resolve_official_execution(request(), execution=e, optimizer=o, gradient_checkpointing=False)


def test_unacknowledged_budget_or_backend_mutation_rejected():
    e, o = ingredients(); e["max_updates"] = 20000
    with pytest.raises(ValueError, match="overrides"):
        rt.resolve_official_execution(request(), execution=e, optimizer=o, gradient_checkpointing=False)
    p = policy(); p["backend"]["optimizer"] = "FusedAdam"
    with pytest.raises(ValueError, match="Modified"):
        rt.official_deepspeed_config(p, accumulation_owner="deepspeed")
    p = policy(); p["effective_recipe"]["max_updates"] = 200
    with pytest.raises(ValueError, match="Modified"):
        rt.official_deepspeed_config(p, accumulation_owner="deepspeed")


def test_manual_global_denominators_are_not_divided_twice():
    e, o = ingredients(); e["micro_batch"] = 1
    r = request(scope="research_stage", adaptation_reason="C global-count normalized microbatches",
                declared_recipe_overrides=["micro_batch"])
    p = rt.resolve_official_execution(r, execution=e, optimizer=o, gradient_checkpointing=False)
    ds = rt.official_deepspeed_config(p, accumulation_owner="caller_global_valid_counts")
    assert p["effective_recipe"]["global_batch"] == 32 and p["accumulation_steps"] == 4
    assert ds["gradient_accumulation_steps"] == 1 and ds["train_batch_size"] == 8
    automatic = rt.official_deepspeed_config(p, accumulation_owner="deepspeed")
    assert automatic["gradient_accumulation_steps"] == 4 and automatic["train_batch_size"] == 32
    with pytest.raises(ValueError, match="owner"):
        rt.official_deepspeed_config(p, accumulation_owner="automatic_guess")


def test_real_torch_adamw_preserves_groups_order_rates_and_updates():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    named = dict(model.named_parameters())
    groups = [{"params": [named["1.bias"], named["0.weight"]], "name": "second_first", "lr": 2e-5},
              {"params": [named["0.bias"], named["1.weight"]], "weight_decay": .02}]
    original_ids = [[id(p) for p in g["params"]] for g in groups]
    opt, receipt = rt.build_official_optimizer(model, groups, policy())
    assert type(opt) is torch.optim.AdamW
    assert [[id(p) for p in g["params"]] for g in opt.param_groups] == original_ids
    assert opt.param_groups[0]["lr"] == 2e-5 and opt.param_groups[1]["weight_decay"] == .02
    assert set(groups[0]) == {"params", "name", "lr"}
    assert receipt["groups"][0]["parameters"][0]["name"] == "1.bias"
    assert opt.defaults["fused"] is None and opt.defaults["foreach"] is None
    before = [p.detach().clone() for p in model.parameters()]
    model(torch.ones(4, 2)).square().mean().backward(); opt.step()
    assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))
    assert all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in opt.state.values())


def test_torch_optimizer_native_state_roundtrip_next_update_exact():
    torch.manual_seed(7); a = nn.Linear(3, 2); b = deepcopy(a)
    oa, _ = rt.build_official_optimizer(a, [{"params": list(a.parameters())}], policy())
    x = torch.arange(12, dtype=torch.float32).view(4, 3) / 8
    a(x).square().mean().backward(); oa.step(); oa.zero_grad()
    b.load_state_dict(a.state_dict(), strict=True)
    ob, _ = rt.build_official_optimizer(b, [{"params": list(b.parameters())}], policy())
    ob.load_state_dict(deepcopy(oa.state_dict()))
    for model, opt in [(a, oa), (b, ob)]:
        model(x).square().mean().backward(); opt.step()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    for p, q in zip(a.parameters(), b.parameters()):
        assert all(torch.equal(oa.state[p][k], ob.state[q][k]) for k in ["step", "exp_avg", "exp_avg_sq"])


@pytest.mark.parametrize("kind", ["duplicate", "missing", "frozen", "foreign", "empty", "bf16", "fused"])
def test_optimizer_misregistration_rejected(kind):
    model = nn.Linear(2, 1); ps = list(model.parameters()); groups = [{"params": ps.copy()}]
    if kind == "duplicate": groups[0]["params"].append(ps[0])
    if kind == "missing": groups[0]["params"].pop()
    if kind == "frozen": ps[0].requires_grad_(False)
    if kind == "foreign": groups[0]["params"][0] = nn.Parameter(torch.ones(1))
    if kind == "empty": groups[0]["params"] = []
    if kind == "bf16": model.bfloat16()
    if kind == "fused": groups[0]["fused"] = True
    with pytest.raises(ValueError): rt.build_official_optimizer(model, groups, policy())


class Language(nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(_attn_implementation="sdpa"); self.enabled = False
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def is_gradient_checkpointing(self): return self.enabled
    def gradient_checkpointing_enable(self): self.enabled = True
    def gradient_checkpointing_disable(self): self.enabled = False


class VLM(nn.Module):
    _supports_flash_attn = True
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.language = Language(); self.visual = Language(); self.visual.requires_grad_(False)

    def set_attn_implementation(self, backend):
        for m in self.modules():
            if hasattr(m, "config"): m.config._attn_implementation = backend


def test_attention_all_submodels_checkpoint_flags_and_weights_unchanged(monkeypatch):
    monkeypatch.setattr(rt, "check_flash_attention2_available", lambda: {"unit_test_fixture": True})
    vlm = VLM(); action = nn.Linear(1, 1); action.gradient_checkpointing = True
    before = {k: v.clone() for k, v in vlm.state_dict().items()}; trainable = [p.requires_grad for p in vlm.parameters()]
    receipt = rt.apply_official_model_runtime(vlm=vlm, language=vlm.language, action_transformer=action,
                                            policy=policy(), training_copy=True)
    assert set(receipt["attention_modules"]) == {"<root>", "language", "visual"}
    assert not vlm.language.is_gradient_checkpointing and not action.gradient_checkpointing
    assert receipt["gpu_qualified"] is False
    assert all(torch.equal(v, before[k]) for k, v in vlm.state_dict().items())
    assert [p.requires_grad for p in vlm.parameters()] == trainable


def test_attention_fallback_and_canonical_evaluation_mutation_rejected(monkeypatch):
    monkeypatch.setattr(rt, "check_flash_attention2_available", lambda: {})
    v = VLM(); a = SimpleNamespace(gradient_checkpointing=False)
    with pytest.raises(ValueError, match="training copy"):
        rt.apply_official_model_runtime(vlm=v, language=v.language, action_transformer=a,
                                        policy=policy(), training_copy=False)
    v.set_attn_implementation = lambda _: setattr(v.config, "_attn_implementation", "flash_attention_2")
    with pytest.raises(RuntimeError, match="fallback"):
        rt.apply_official_model_runtime(vlm=v, language=v.language, action_transformer=a,
                                        policy=policy(), training_copy=True)


@pytest.mark.parametrize("missing", ["visual", "language_config", "checkpoint_evidence"])
def test_absent_submodel_or_checkpointing_evidence_cannot_silently_pass(monkeypatch, missing):
    monkeypatch.setattr(rt, "check_flash_attention2_available", lambda: {})
    v = VLM(); a = SimpleNamespace(gradient_checkpointing=False)
    if missing == "visual": del v.visual
    if missing == "language_config": del v.language.config
    if missing == "checkpoint_evidence": v.language.enabled = None; v.language.gradient_checkpointing_disable = lambda: None
    with pytest.raises((ValueError, RuntimeError)):
        rt.apply_official_model_runtime(vlm=v, language=v.language, action_transformer=a,
                                        policy=policy(), training_copy=True)


def test_missing_fa2_extension_is_not_silent_sdpa(monkeypatch):
    monkeypatch.setattr(rt.importlib.metadata, "version", lambda name: rt.PINNED_PACKAGES[name])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    def absent(_): raise ImportError("fixture missing extension")
    monkeypatch.setattr(rt.importlib, "import_module", absent)
    with pytest.raises(RuntimeError, match="no SDPA fallback"): rt.check_flash_attention2_available()


def test_unsupported_dependency_or_device_rejected(monkeypatch):
    monkeypatch.setattr(rt.importlib.metadata, "version", lambda _: "0.0.0")
    with pytest.raises(RuntimeError, match="dependency"): rt.check_flash_attention2_available()
    monkeypatch.setattr(rt.importlib.metadata, "version", lambda name: rt.PINNED_PACKAGES[name])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"): rt.check_flash_attention2_available()


def test_engine_verifier_requires_real_torch_client_bf16_and_fp32_moments():
    model = nn.Linear(2, 1)
    opt, _ = rt.build_official_optimizer(model, [{"params": [p]} for p in model.parameters()], policy())
    model(torch.ones(1, 2)).sum().backward(); opt.step()
    masters = list(model.parameters())
    zero = SimpleNamespace(optimizer=opt, cpu_offload=False, dtype=torch.bfloat16,
        gradient_accumulation_dtype=torch.bfloat16, single_partition_of_fp32_groups=masters,
        reduce_bucket_size=500000000, allgather_bucket_size=500000000)
    engine = SimpleNamespace(optimizer=zero, module=deepcopy(model).bfloat16(),
        zero_optimization_stage=lambda: 2, bfloat16_enabled=lambda: True)
    receipt = rt.verify_official_engine(engine, opt, policy(), require_moments=True)
    assert receipt["moment_elements"] == 2 * receipt["master_elements"]
    zero.reduce_bucket_size = 50000000
    with pytest.raises(ValueError, match="buckets"):
        rt.verify_official_engine(engine, opt, policy())
    zero.reduce_bucket_size = 500000000
    zero.single_partition_of_fp32_groups = list(reversed(masters))
    with pytest.raises(ValueError, match="ordered"):
        rt.verify_official_engine(engine, opt, policy())
    zero.single_partition_of_fp32_groups = masters
    opt.state[masters[0]]["exp_avg"] = opt.state[masters[0]]["exp_avg"].bfloat16()
    with pytest.raises(ValueError, match="moment"):
        rt.verify_official_engine(engine, opt, policy(), require_moments=True)
    zero.cpu_offload = True
    with pytest.raises(ValueError, match="offload"):
        rt.verify_official_engine(engine, opt, policy())


def test_actual_cli_resolves_reference_without_launching(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "resolved.json"
    command = [sys.executable, "-m", "tools.st_world_policy.official_runtime", "--runtime-policy",
        str(repo / "configs/st_world_policy/official_execution_reference.yaml"), "--training-config",
        str(repo / "configs/st_world_policy/official_reference_recipe.yaml"), "--output", str(output)]
    subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    d = json.loads(output.read_text()); assert d["gpu_qualified"] is False
    assert d["effective_recipe"]["max_updates"] == 100000
    first = output.read_bytes()
    subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    assert output.read_bytes() == first
