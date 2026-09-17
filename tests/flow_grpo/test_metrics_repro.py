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
        numerical_profile="bf16_zero2_fp32_accum_v1",
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


def test_acceptance_rejects_changed_code_config_assets_or_evidence(
    tmp_path, monkeypatch
):
    import json
    from starVLA.rl.flow_grpo.acceptance import GATES
    from starVLA.rl.flow_grpo.loading import file_sha
    from starVLA.rl.flow_grpo.config import resolve_config

    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg, _ = resolve_config("configs/flow_grpo/paired_frozen_visual.yaml")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"PASS","scope":"unit test only"}')
    receipt = tmp_path / "release.json"
    cfg["runtime"]["acceptance_record"] = str(receipt)
    context = {
        "code": "original",
        "config": "original",
        "assets": "original",
        "world_size": 4,
    }
    record = {
        "status": "READY_FOR_THIS_PROFILE",
        "context": context,
        "gates": {
            g: {"status": "PASS", "path": str(evidence), "sha256": file_sha(evidence)}
            for g in GATES
        },
    }
    receipt.write_text(json.dumps(record))
    enforce_training_budget(cfg, context)
    for key in context:
        with pytest.raises(ValueError, match="does not match"):
            enforce_training_budget(cfg, {**context, key: "changed"})
    evidence.write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match="evidence changed"):
        enforce_training_budget(cfg, context)


def test_adam_update_is_not_classified_as_zero_by_acceptance_atol():
    from starVLA.rl.flow_grpo.comparison import tensor_comparison

    result = tensor_comparison(torch.tensor([1e-6, -1e-6]), torch.tensor([2e-6, -1e-6]))
    assert not result["near_zero_reference"]
    assert result["relative_l2"] == pytest.approx(2**-0.5)
