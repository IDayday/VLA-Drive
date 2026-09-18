"""Score saved paired ODE predictions with the vendored NAVSIM v1.1 evaluator.

This is a separate CPU-only development diagnostic, never an RL reward change.
It builds genuine v1 caches from raw logged scenes, not converted v2 caches.
No scene is silently omitted, substituted, resampled or assigned zero on error.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"navsim_v1.1/navsim"
sys.path.insert(0, str(V1))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def initialize(spec):
    global SPEC, PREDICTIONS
    import numpy as np
    import navsim
    assert Path(navsim.__file__).resolve().is_relative_to(V1.resolve())
    SPEC = spec
    PREDICTIONS = {}
    for label, directory in spec["evaluations"].items():
        with np.load(Path(directory)/"trajectories.npz", allow_pickle=False) as data:
            PREDICTIONS[label] = dict(zip(data["tokens"].tolist(), data["physical"]))


def score_chunk(job):
    import numpy as np
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    log, tokens = job
    loader = SceneLoader(Path(SPEC["raw_logs"]), None,
        SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                    log_names=[log], tokens=tokens), SensorConfig.build_no_sensors())
    if set(loader.tokens) != set(tokens):
        raise ValueError(f"raw scenes missing for {log}: {set(tokens)-set(loader.tokens)}")
    processor = MetricCacheProcessor(SPEC["cache_root"], False)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling)
    rows = []
    for token in tokens:
        scene = loader.get_scene_from_token(token)
        scenario = NavSimScenario(scene, map_root=SPEC["maps"], map_version="nuplan-maps-v1.0")
        entry = processor.compute_metric_cache(scenario)
        with lzma.open(entry.file_name, "rb") as stream:
            cache = pickle.load(stream)  # Only caches produced by this trusted v1 processor.
        if cache.ego_state.time_point != scenario.initial_ego_state.time_point:
            raise ValueError("v1 cache timestamp differs from the raw scene")
        for label, mapping in PREDICTIONS.items():
            poses = mapping[token]
            if poses.shape != (8, 3) or not np.isfinite(poses).all():
                raise ValueError(f"invalid saved prediction: {label}/{token}")
            result = asdict(pdm_score(cache, Trajectory(poses,
                TrajectorySampling(num_poses=8, interval_length=.5)), sampling, simulator, scorer))
            if not all(np.isfinite(v) for v in result.values()):
                raise ValueError(f"nonfinite official v1 score: {label}/{token}")
            rows.append({"token": token, "log_name": log, "label": label, "valid": True, **result})
        # Check scorer reuse does not change the first policy's reference pool.
        if token == tokens[0]:
            label = next(iter(PREDICTIONS))
            repeat = asdict(pdm_score(cache, Trajectory(PREDICTIONS[label][token],
                TrajectorySampling(num_poses=8, interval_length=.5)), sampling, simulator, scorer))
            first = next(r for r in rows if r["token"] == token and r["label"] == label)
            if any(repeat[k] != first[k] for k in repeat):
                raise ValueError("serial policy scoring changed the reference/score")
    return rows


def main():
    import numpy as np
    import pandas as pd
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        p.error("CPU workers must be in [1,16]")
    spec = json.loads(Path(a.spec).read_text())
    seed = int(spec.get("seed", 42))
    split = spec.get("split", "rl_dev")
    if split not in ("rl_dev", "navtest"):
        raise ValueError("explicit evaluation split required")
    os.environ["NUPLAN_MAPS_ROOT"] = spec["maps"]
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(spec["scene_csv"], dtype={"token":str,"log_name":str})
    tokens = table.token.tolist()
    if not tokens or table.token.duplicated().any() or table.log_name.isna().any():
        raise ValueError("unique explicit scenes and log identities required")
    inputs = {}
    for label, directory in spec["evaluations"].items():
        root = Path(directory)
        report = json.loads((root/"evaluation.json").read_text())
        if (not (root/"COMPLETE").is_file() or report["status"] != "COMPLETE"
                or report["seed"] != seed or report.get("split", report.get("identity",{}).get("split")) != split):
            raise ValueError("requires completed ODE predictions with matching seed/split")
        with np.load(root/"trajectories.npz", allow_pickle=False) as data:
            found = data["tokens"].tolist()
            if len(found) != len(set(found)) or set(found) != set(tokens):
                raise ValueError("predictions differ from the fixed dev token set")
        inputs[label] = {"path":str(root), "trajectory_sha256":sha(root/"trajectories.npz"),
                         "checkpoint_sha256":report["identity"]["checkpoint_sha256"]}
    identity = {"protocol":"NAVSIM_v1.1_PDMScorer_default", "scope":"fixed_"+split,
        "scene_count":len(tokens), "logs":int(table.log_name.nunique()), "seed":seed,
        "inputs":inputs,"spec":spec,"script_sha256":sha(__file__),
        "scorer_sources":{str(f.relative_to(V1)):sha(f) for f in sorted((V1/"navsim").rglob("*.py"))},
        "sampling":{"scored_poses":40,"scored_interval":.1,"prediction_poses":8,"prediction_interval":.5},
        "cpu_workers":a.workers,"new_gpu_inference":False}
    publish(output/"identity.json", identity)
    jobs = [(log, part.token.tolist()[i:i+32]) for log, part in table.groupby("log_name", sort=True)
            for i in range(0, len(part), 32)]
    started = time.time();rows=[]
    try:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=(spec,)) as pool:
            futures = [pool.submit(score_chunk, job) for job in jobs]
            for future in as_completed(futures):
                rows.extend(future.result())
                publish(output/"progress.json", {"status":"RUNNING","scenes":len(rows)//len(inputs),
                    "total":len(tokens),"seconds":time.time()-started})
        frame = pd.DataFrame(rows)
        summaries = {}
        for label, part in frame.groupby("label", sort=True):
            if len(part) != len(tokens) or part.token.duplicated().any() or set(part.token) != set(tokens):
                raise ValueError("incomplete or duplicate result rows")
            path = output/(label+".csv")
            part.sort_values("token").to_csv(path,index=False)
            summaries[label] = {"scenes":len(part),"valid":int(part.valid.sum()),
                "means":{c:float(part[c].mean()) for c in part if c not in {"token","log_name","label","valid"}},
                "csv_sha256":sha(path)}
        publish(output/"summary.json", {"status":"COMPLETE", "protocol":identity["protocol"],
            "scope":identity["scope"], "results":summaries, "seconds":time.time()-started})
        publish(output/"COMPLETE", {"summary_sha256":sha(output/"summary.json"), "identity_sha256":sha(output/"identity.json")})
        print(json.dumps(summaries,indent=2))
    except BaseException as exc:
        publish(output/"failure.json", {"status":"FAIL","error":repr(exc),"scored_rows":len(rows)})
        raise


if __name__ == "__main__":
    main()
