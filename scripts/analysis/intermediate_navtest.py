"""One preselected saved checkpoint, full navtest and paired same-seed SFT scores.

Uses the original single-candidate evaluator and existing official v1 scorer.
No training commands, checkpoint selection, new predictions or reward changes.
"""
import argparse
import json
from pathlib import Path
import threading
import time

from scripts.cluster_flow_grpo.cluster import exclusive_controller, write_json, run, PYTHON
from scripts.cluster_flow_grpo.parallel_evaluation import evaluate_parallel
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.evaluation import paired_scores
from starVLA.rl.flow_grpo.evaluation_transaction import completed_evaluation
from starVLA.rl.flow_grpo.transactions import preserve_attempt


def validate_pair_identity(original, current):
    # Training implementation/ZeRO/checkpointing metadata can differ. The actual
    # original inference implementation is separately bound by source hashes.
    keys = ("schema_version", "data_root", "observation_assets_sha256", "metric_assets_sha256",
            "resolved_data_model_config_sha256", "tokens_sha256", "seed", "split",
            "metric_protocol", "processor", "dependencies")
    for key in keys:
        if key not in original or original[key] != current.get(key):
            raise ValueError("SFT/RL evaluation identity differs: " + key)
    numeric = ("model_dtype", "qwen_autocast", "action_history_projector_autocast",
               "attention_backend", "deterministic", "cudnn_deterministic", "cudnn_benchmark",
               "matmul_tf32", "cudnn_tf32", "bf16_reduced_precision_reduction",
               "float32_matmul_precision", "flash_deterministic", "cublas_workspace")
    for key in numeric:
        if key not in original["numerics"] or original["numerics"][key] != current["numerics"].get(key):
            raise ValueError("SFT/RL inference numerics differ: " + key)


def paired_result(baseline, trained, tokens):
    """The complete declared token set is required before a paired statistic."""
    if len(tokens) != len(set(tokens)):
        raise ValueError("duplicate declared tokens")
    for frame in (baseline, trained):
        if len(frame) != len(tokens) or set(frame.token) != set(tokens):
            raise ValueError("incomplete paired evaluation")
    rows, summary = paired_scores(baseline, trained)
    summary.update(sft_mean=float(rows.score_sft.mean()), rl_mean=float(rows.score_rl.mean()),
                   wins=int((rows.delta > 0).sum()), losses=int((rows.delta < 0).sum()),
                   ties=int((rows.delta == 0).sum()))
    return rows, summary


def execute(spec):
    import pandas as pd
    root = Path(spec["output"])
    common = spec["evaluation"]
    tokens = json.loads(Path(common["tokens"]).read_text())
    if len(tokens) != 12146 or spec["seed"] != 42:
        raise ValueError("this interim registration requires full navtest and fixed seed42")
    seed, label = spec["seed"], spec["label"]
    checkpoint = Path(spec["checkpoint"])
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if not (checkpoint / "COMPLETE").is_file() or state["update"] != spec["update"]:
        raise ValueError("checkpoint is incomplete or update differs")
    for name, expected in spec["inference_sources"].items():
        if file_sha(name) != expected:
            raise ValueError("baseline inference implementation changed: " + name)
    from starVLA.rl.flow_grpo.checkpoint import export_checkpoint
    exported = export_checkpoint(checkpoint, spec["exported"])
    cancel = threading.Event()
    started = time.time()

    def progress(phase):
        write_json(root / "progress.json", dict(status="RUNNING", phase=phase,
                   update=spec["update"], seed=seed, scenes=len(tokens), started=started))

    progress("original_ODE_and_official_v2")
    v2 = root / "navtest_seed42_v2"
    report = evaluate_parallel(spec["config"], str(exported), str(v2), "navtest",
                              common["tokens"], common["data_root"], common["metric_cache"],
                              seed, common["slots"], cancel_event=cancel)
    if report["status"] != "COMPLETE" or report["scene_count"] != len(tokens):
        raise ValueError("full v2 evaluation did not complete")
    progress("official_v1_PDMS_on_same_predictions")
    v1 = root / "navtest_seed42_pdms_v1"
    v1spec = dict(split="navtest", seed=seed, raw_logs=common["raw_logs"], maps=common["maps"],
                  cache_root=common["v1_cache"], scene_csv=str(v2 / "original_protocol_scores.csv"),
                  evaluations={label: str(v2)})
    v1spec_path = root / "v1_spec.json"
    write_json(v1spec_path, v1spec)
    if not (v1 / "COMPLETE").is_file():
        if v1.exists():
            preserve_attempt(v1)
        job = dict(job_id=label+"_v1", cpu_only=True,
                   nodes=[dict(host=common["slots"][0]["host"], devices=[], cpu_affinity=list(range(64,96)))],
                   direct_command=[PYTHON, "-m", "scripts.analysis.paired_dev_pdms_v1", "--spec",
                                   str(v1spec_path), "--workers", "16", "--output", str(v1)],
                   control_dir=str(root / f"v1_control_{time.time_ns()}"), timeout_seconds=14400,
                   environment={"CUDA_VISIBLE_DEVICES": ""})
        path = root / f"v1_job_{time.time_ns()}.json"
        write_json(path, job)
        run(path, cancel_event=cancel)
    saved_identity = json.loads((v1 / "identity.json").read_text())
    if (saved_identity["spec"] != v1spec or saved_identity["inputs"][label]["trajectory_sha256"]
            != file_sha(v2 / "trajectories.npz")):
        raise ValueError("v1 predictions/spec identity changed")
    progress("paired_comparison_to_own_SFT")
    baseline = spec["baseline"]
    original = completed_evaluation(baseline["v2"], spec["baseline_identity"], tokens)
    if original is None:
        raise ValueError("same-seed full SFT baseline incomplete")
    # Keep the same inputs and inference protocol, changing only the checkpoint.
    validate_pair_identity(original["identity"], report["identity"])
    results = {}
    for protocol, left, right in (("PDMS_v1", Path(baseline["v1"]), v1),
                                 ("EPDMS_v2", Path(baseline["v2"]), v2)):
        if protocol == "PDMS_v1":
            for directory, name in ((left, "sft"), (right, label)):
                seal = json.loads((directory / "COMPLETE").read_text())
                if (seal["summary_sha256"] != file_sha(directory / "summary.json")
                        or seal["identity_sha256"] != file_sha(directory / "identity.json")):
                    raise ValueError("v1 completion seal changed")
                summary = json.loads((directory / "summary.json").read_text())
                if summary["results"][name]["csv_sha256"] != file_sha(directory / (name + ".csv")):
                    raise ValueError("v1 result CSV changed")
            left_file, right_file = left / "sft.csv", right / (label + ".csv")
        else:
            left_file, right_file = left / "original_protocol_scores.csv", right / "original_protocol_scores.csv"
        frames = [pd.read_csv(p, dtype={"token": str, "log_name": str}) for p in (left_file, right_file)]
        rows, summary = paired_result(*frames, tokens)
        rows.to_csv(root / (protocol + "_paired.csv"), index=False)
        results[protocol] = summary
    result = dict(status="COMPLETE", checkpoint=str(checkpoint), update=spec["update"], seed=seed,
                  scene_count=len(tokens), scope="interim single-seed full-navtest; not final five-seed result",
                  checkpoint_sha256=report["identity"]["checkpoint_sha256"], results=results,
                  seconds=time.time()-started)
    write_json(root / "result.json", result)
    write_json(root / "progress.json", result)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--spec", required=True)
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    if spec["controller_sha256"] != file_sha(__file__):
        raise ValueError("intermediate evaluation controller changed")
    with exclusive_controller(spec["output"]):
        try:
            execute(spec)
        except BaseException as exc:
            write_json(Path(spec["output"]) / f"failure_{time.time_ns()}.json",
                       {"status": "FAIL", "error": repr(exc)})
            raise
