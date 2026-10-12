from dataclasses import asdict
from pathlib import Path
import os
import subprocess
from iqe.contracts import require
from iqe.io import atomic_json, read_json, digest
from iqe.data.manifests import load_scenes
from .features import JointCache
from .bank import cache_candidates, score_candidates, protocol


def audit(config):
    cache = JointCache(config)
    scene = next(s for s in load_scenes(config.root/"scenes.json") if s.split_role == "incremental_fit" and cache.path(s).exists())
    candidates = cache_candidates(config, [scene], "base")+cache_candidates(config, [scene], "gt")
    results = score_candidates(config, [scene]*2, candidates, "reference_equivalence_audit")
    proto = protocol(config)
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo), proto["navsim_root"], proto["nuplan_root"])),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    folder = config.root/"reference_equivalence"
    layouts = [[0], [0,1], [1,0], [1], [0,1,0]]
    evidence = []
    for layout_index, layout in enumerate(layouts):
        for position, index in enumerate(layout):
            candidate = candidates[index]
            request = {"candidate": asdict(candidate), "scene": asdict(scene), "protocol_hash": proto["protocol_hash"],
                       "navsim_root": proto["navsim_root"], "source_hash": proto["navsim_source_hash"], "config": proto["config"],
                       "seed": int(digest({"scene_id": scene.scene_id,"seed":42,"repetition":0})[:8],16), "eval_seed":42,"repetition":0}
            req, response = folder/f"{layout_index}_{position}_request.json", folder/f"{layout_index}_{position}_response.json"
            atomic_json(req, request, immutable=True)
            job = subprocess.run([config.scoring_python,"-m","iqe.scoring.worker",str(req),str(response)], env=env, cwd=repo,
                                 capture_output=True,text=True,timeout=300)
            require(job.returncode == 0, job.stderr[-2000:])
            actual = read_json(response)
            require(actual == asdict(results[index]), "fresh official single entry differs from persistent-import bank")
            evidence.append({"layout":layout,"position":position,"candidate_key":candidate.key,"exact_equal":True})
    result = {"status":"PASS","scene_id":scene.scene_id,"fresh_official_calls":len(evidence),
              "persistent_import_calls":2,"evidence":evidence,"protocol":proto,
              "state_isolation":"fresh metric cache, observation, simulator, scorer for every trajectory",
              "GT_is_deployment_candidate":False}
    atomic_json(config.root/"SCORE_REFERENCE_EQUIVALENCE.json",result,immutable=True)
    return result


if __name__ == "__main__":
    import argparse,json
    from .config import load_config
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    print(json.dumps(audit(load_config(a.config))))
