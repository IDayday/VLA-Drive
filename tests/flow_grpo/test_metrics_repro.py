import copy
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from starVLA.rl.flow_grpo.metrics import Metrics
from starVLA.rl.flow_grpo.reproducibility import (
    processor_identity,
    assert_resume_identity,
    write_asset_manifest,
    verify_asset_manifest,
)
from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
from starVLA.rl.flow_grpo.comparison import compare_named


def _metrics_worker(rank, path, output):
    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2
    )
    metric = Metrics()
    batches = [
        [torch.tensor([0.8, 0.99], dtype=torch.float64), torch.tensor([1.0])],
        [torch.tensor([1.01]), torch.tensor([1.03, 1.2], dtype=torch.float64)],
    ][rank]
    for values in batches:
        metric.ratios("ratio", values, 0.02)
        metric.add("scenes", "count", 1)
    states = [None, None]
    dist.all_gather_object(states, metric.state)
    if rank == 0:
        result = Metrics()
        for state in states:
            result.merge(Metrics(state))
        torch.save(result.result(), output)
    dist.destroy_process_group()


def test_global_metrics_extrema_fraction_quantiles(tmp_path):
    out = str(tmp_path / "metrics.pt")
    mp.spawn(_metrics_worker, args=(str(tmp_path / "rdzv"), out), nprocs=2, join=True)
    result = torch.load(out, weights_only=True)
    assert result["ratio_min"] == 0.8 and result["ratio_max"] == 1.2
    assert result["ratio_clip_fraction"] == 0.5
    assert result["ratio_count"] == 6 and result["scenes"] == 4
    assert result["ratio_mean"] == pytest.approx(
        (0.8 + 0.99 + 1 + 1.01 + 1.03 + 1.2) / 6
    )
    assert result["ratio_quantiles"]["0.5"] == pytest.approx(1.005)


def test_processor_same_path_replacement_rejects_resume(tmp_path):
    p = tmp_path / "processor_config.json"
    p.write_text('{"size":1024}')
    old = {"processor": processor_identity(tmp_path)}
    p.write_text('{"size":1025}')
    with pytest.raises(ValueError, match="processor"):
        assert_resume_identity(old, {"processor": processor_identity(tmp_path)})


def test_publisher_refuses_missing_evidence_without_creating_release(tmp_path):
    from scripts.flow_grpo.publish_acceptance import publish

    destination = tmp_path / "never_ready.json"
    with pytest.raises(ValueError, match="missing gates"):
        publish({"runtime": {"acceptance_record": str(destination)}}, {}, [], destination)
    assert not destination.exists()


def test_numerics_and_dependency_change_reject_resume():
    old = {"numerics": {"tf32": False}, "dependencies": {"torch": "2.5.1"}}
    new = copy.deepcopy(old)
    new["numerics"]["tf32"] = True
    with pytest.raises(ValueError, match="numerics"):
        assert_resume_identity(old, new)
    new = copy.deepcopy(old)
    new["dependencies"]["torch"] = "changed"
    with pytest.raises(ValueError, match="dependencies"):
        assert_resume_identity(old, new)


def test_manifest_detects_same_size_timestamp_replacement(tmp_path):
    data = tmp_path / "cache.pkl"
    data.write_bytes(b"abcd")
    output = tmp_path / "manifest.json"
    manifest = write_asset_manifest([data], output)
    assert verify_asset_manifest(output) == manifest["identity"]
    st = data.stat()
    data.write_bytes(b"abce")
    os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ValueError, match="asset changed"):
        verify_asset_manifest(output)


def test_formal_training_fails_closed():
    cfg = {"runtime": {"max_updates": 2000}, "sampling": {"candidate_chunk_size": 1}}
    with pytest.raises(ValueError, match="bounded"):
        enforce_training_budget(cfg)
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(
        OmegaConf.load("configs/flow_grpo/action_only_frozen_visual.yaml")
    )
    cfg["sampling"]["num_steps"] = 10
    cfg["algorithm"]["inner_epochs"] = 2
    cfg["runtime"].update(
        run_mode="formal",
        max_updates=2000,
        accumulation_steps=16,
        numerical_profile="bf16_zero2_fp32_partition_v2",
    )
    with pytest.raises(ValueError, match="missing"):
        enforce_training_budget(cfg, context={})
    cfg["sampling"]["candidate_chunk_size"] = 2
    with pytest.raises(ValueError, match="experimental"):
        enforce_training_budget(cfg, context={})


def test_comparison_collects_every_failure_and_near_zero():
    a = {
        "action_model.a": torch.tensor([1.0, 0.0]),
        "qwen_vl_interface.b": torch.zeros(2),
    }
    b = {
        "action_model.a": torch.tensor([2.0, 0.0]),
        "qwen_vl_interface.b": torch.ones(2),
        "action_input_model.c": torch.ones(1),
    }
    report = compare_named(a, b)
    assert len(report["parameters"]) == 3
    assert report["modules"]["whole_model"]["failed_tensors"] == 3
    assert report["parameters"]["qwen_vl_interface.b"]["relative_l2"] is None


def test_dtype_inventory_handles_zero2_released_partition_groups():
    from types import SimpleNamespace
    from starVLA.rl.flow_grpo.monitor import dtype_inventory

    model = torch.nn.Linear(2, 1).bfloat16()
    optimizer = SimpleNamespace(
        averaged_gradients={0: None, 1: [None, torch.ones(2)]},
        single_partition_of_fp32_groups=[torch.ones(2)],
        gradient_accumulation_dtype=torch.float32,
        communication_data_type=torch.float32,
        optimizer=SimpleNamespace(state={}),
    )
    engine = SimpleNamespace(optimizer=optimizer)
    before = dtype_inventory(model, engine)
    assert before["partition_buffers"] == ["torch.float32"]
    optimizer.averaged_gradients[1] = None
    after = dtype_inventory(model, engine)
    assert after["partition_buffers"] == []
    assert before["partition_buffers"] == ["torch.float32"]
    assert after["communication_buffers"] == []  # never invent observed dtypes


def acceptance_fixture(tmp_path, monkeypatch):
    """Synthetic TEMP fixtures for validator control flow, never production receipts."""
    import json
    from starVLA.rl.flow_grpo.acceptance import GATES, GATE_REQUIREMENTS
    from starVLA.rl.flow_grpo.loading import file_sha
    from starVLA.rl.flow_grpo.config import resolve_config, config_hash
    from starVLA.rl.flow_grpo.reproducibility import numerical_profile
    from starVLA.rl.flow_grpo.contracts import digest

    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg, sft = resolve_config("configs/flow_grpo/paired_fp32_partition_frozen_visual.yaml")
    profile = numerical_profile(cfg, sft)
    evidence, receipt, inventory = [
        tmp_path / x for x in ("evidence.json", "release.json", "dtype.json")
    ]
    cfg["runtime"]["acceptance_record"] = str(receipt)
    context = {
        "executable_sha256": "c" * 64,
        "config_sha256": config_hash(cfg),
        "checkpoint_sha256": cfg["checkpoint_contract"]["sha256"],
        "resume_identity": {"numerics": profile, "assets": "TEMP validator fixture"},
        "world_size": 4,
    }
    row = {
        "device": {"type": "cuda", "name": "SYNTHETIC TEST FIXTURE"},
        "dtype": {
            "parameters": {"torch.bfloat16": 1},
            "accumulation_dtype": "torch.float32",
            "communication_dtype": "torch.float32",
            "communication_buffers": ["torch.float32"],
            "master_weights": ["torch.float32"],
            "optimizer_states": {"torch.float32": 2},
        },
        "dtype_before_step": {"partition_buffers": ["torch.float32"]},
        "activation_dtypes": {"language": ["torch.bfloat16"]},
    }
    inventory.write_text(
        json.dumps({"ranks": [{**row, "rank": rank} for rank in range(4)]})
    )
    tests = []
    for gate in GATES:
        category, checks = GATE_REQUIREMENTS[gate]
        production, cuda = (
            category == "production",
            category in {"production", "cuda_model"},
        )
        world = 4 if production else 2 if category == "tool_distributed" else 1
        tests.append(
            {
                "schema_version": 1,
                "test_id": gate,
                "status": "PASS",
                "execution": {
                    "executable_sha256": context["executable_sha256"],
                    "exit_code": 0,
                    "completed": True,
                    "command": ["synthetic schema fixture; not an executed GPU test"],
                },
                "checkpoint": None
                if category.startswith("tool")
                else {
                    "sha256": context["checkpoint_sha256"],
                    "variant": cfg["checkpoint_contract"]["variant"],
                },
                "scope": {
                    "kind": "model_independent"
                    if category.startswith("tool")
                    else "full_model",
                    "checks": sorted(checks),
                },
                "observed_profile": {
                    "device_type": "cuda" if cuda else "cpu",
                    "precision": "bfloat16" if cuda else "float32",
                    "backend": "deepspeed" if production else "pytorch",
                    "world_size": world,
                    "devices": ["fixture"] * world,
                    "zero_stage": 2,
                    **{
                        key: profile[key]
                        for key in (
                            "candidate_chunk",
                            "transition_chunk",
                            "activation_checkpointing",
                            "optimizer_offload",
                            "attention_backend",
                        )
                    },
                },
                "declared_profile": profile,
                "config_sha256": context["config_sha256"],
                "resume_identity_sha256": digest(context["resume_identity"]),
                "artifacts": {
                    "dtype_inventory": {
                        "path": str(inventory),
                        "sha256": file_sha(inventory),
                    }
                },
                "results": {
                    **{check: True for check in checks},
                    "chunks": [1, 2],
                    "groups": [2, 8],
                    "num_steps": 10,
                    "optimizer_updates": 2,
                    "official_nonzero_advantage_groups": 1,
                    "compared_world_sizes": [1, 4],
                },
            }
        )
    bundle = {"schema_version": 1, "tests": tests}

    def write():
        evidence.write_text(json.dumps(bundle))
        record = {
            "status": "READY_FOR_THIS_PROFILE",
            "context": context,
            "gates": {
                gate: {
                    "status": "PASS",
                    "test_id": gate,
                    "path": str(evidence),
                    "sha256": file_sha(evidence),
                }
                for gate in GATES
            },
        }
        receipt.write_text(json.dumps(record))

    write()
    return cfg, context, bundle, write, evidence, receipt


def test_acceptance_rejects_changed_code_config_assets_or_evidence(
    tmp_path, monkeypatch
):
    cfg, context, _, _, evidence, _ = acceptance_fixture(tmp_path, monkeypatch)
    enforce_training_budget(cfg, context)
    for key in context:
        with pytest.raises(ValueError, match="does not match"):
            enforce_training_budget(cfg, {**context, key: "changed"})
    evidence.write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match="evidence changed"):
        enforce_training_budget(cfg, context)


def test_publisher_validates_before_atomic_publication_and_is_idempotent(tmp_path, monkeypatch):
    from scripts.flow_grpo.publish_acceptance import publish

    cfg, context, _, _, evidence, receipt = acceptance_fixture(tmp_path, monkeypatch)
    receipt.unlink()  # TEMP synthetic validator fixture, never a production path
    publish(cfg, context, [evidence], receipt)
    before = receipt.stat().st_mtime_ns
    publish(cfg, context, [evidence], receipt)
    assert receipt.stat().st_mtime_ns == before


@pytest.mark.parametrize("damage", ["inner_fail", "recipe"])
def test_publisher_does_not_publish_invalid_record(tmp_path, monkeypatch, damage):
    import json
    from scripts.flow_grpo.publish_acceptance import publish

    cfg, context, bundle, _, evidence, receipt = acceptance_fixture(tmp_path, monkeypatch)
    receipt.unlink()
    if damage == "inner_fail":
        bundle["tests"][0]["status"] = "FAIL"
        evidence.write_text(json.dumps(bundle))
    else:
        cfg["runtime"]["seed"] = 99
    with pytest.raises(ValueError):
        publish(cfg, context, [evidence], receipt)
    assert not receipt.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "FAIL",
        "NOT_RUN",
        "BLOCKED",
        "cpu",
        "other_checkpoint",
        "test_id",
        "backend",
        "world_size",
        "devices",
        "generic",
        "dtype",
        "zero_advantage",
    ],
)
def test_semantic_gate_rejects_correct_sha_but_invalid_evidence(
    tmp_path, monkeypatch, damage
):
    cfg, context, bundle, write, evidence, receipt = acceptance_fixture(
        tmp_path, monkeypatch
    )
    entry = next(
        row for row in bundle["tests"] if row["test_id"] == "rl_sft_reference_gradients"
    )
    if damage in {"FAIL", "NOT_RUN", "BLOCKED"}:
        entry["status"] = damage
    elif damage == "cpu":
        entry["observed_profile"].update(device_type="cpu", precision="float32")
    elif damage == "other_checkpoint":
        entry["checkpoint"].update(sha256="U" * 64, variant="unfrozen_visual")
    elif damage == "test_id":
        entry["test_id"] = "wrong_test_id"
    elif damage in {"backend", "world_size", "devices"}:
        entry["observed_profile"].pop(damage)
    elif damage == "generic":
        entry["scope"]["kind"] = "model_independent"
    elif damage == "zero_advantage":
        entry["results"]["official_nonzero_advantage_groups"] = 0
    elif damage == "dtype":
        entry["artifacts"] = {}
    write()  # deliberately CORRECT SHA: the semantic validator must still refuse.
    with pytest.raises(ValueError):
        enforce_training_budget(cfg, context)


def test_generic_unit_pass_cannot_satisfy_all_gates(tmp_path, monkeypatch):
    import json
    from starVLA.rl.flow_grpo.loading import file_sha

    cfg, context, _, _, evidence, receipt = acceptance_fixture(tmp_path, monkeypatch)
    evidence.write_text('{"status":"PASS","scope":"unit test only"}')
    record = json.loads(receipt.read_text())
    for gate in record["gates"].values():
        gate["sha256"] = file_sha(evidence)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema"):
        enforce_training_budget(cfg, context)


def test_legal_cpu_math_evidence_and_complete_bundle(tmp_path, monkeypatch):
    from starVLA.rl.flow_grpo.acceptance import validate_gate_evidence
    import json

    cfg, context, bundle, _, _, receipt = acceptance_fixture(tmp_path, monkeypatch)
    math = next(
        row for row in bundle["tests"] if row["test_id"] == "fp32_chunk_mathematics"
    )
    assert math["observed_profile"]["device_type"] == "cpu"
    pointer = json.loads(receipt.read_text())["gates"]["fp32_chunk_mathematics"]
    validate_gate_evidence("fp32_chunk_mathematics", pointer, context, cfg)
    enforce_training_budget(cfg, context)


def test_adam_update_is_not_classified_as_zero_by_acceptance_atol():
    from starVLA.rl.flow_grpo.comparison import tensor_comparison

    result = tensor_comparison(torch.tensor([1e-6, -1e-6]), torch.tensor([2e-6, -1e-6]))
    assert not result["near_zero_reference"]
    assert result["relative_l2"] == pytest.approx(2**-0.5)


def test_collective_dtype_observer_preserves_arguments_and_restores_api():
    from types import SimpleNamespace
    from starVLA.rl.flow_grpo.monitor import CommunicationDtypeMonitor

    calls = []

    def original(value, group=None):
        calls.append((value, group))
        return value

    module = SimpleNamespace(all_reduce=original)
    engine = SimpleNamespace()
    observer = CommunicationDtypeMonitor(engine, module)
    tensor = torch.ones(4)
    assert module.all_reduce(tensor, group="fixture") is tensor
    assert calls == [(tensor, "fixture")]
    assert engine.flow_communication_dtypes == set()  # CPU isn't GPU evidence
    observer.close()
    assert module.all_reduce is original
