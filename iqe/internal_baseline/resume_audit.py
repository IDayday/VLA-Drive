"""Short real-feature/real-label resume proof; never supplies main-run weights."""
from copy import deepcopy
import json
import os
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, file_hash, digest
from iqe.data.manifests import load_scenes
from iqe.training.trainer import train
from .training import load_bank, make_scorer, training_config, score_context, training_device
from .features import JointCache
from .sampling import TwoStreamSampler
from .objectives import terms, denominators, comparison_weights


def equal_tree(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def audit(config, device="cuda:0", deterministic=False):
    if deterministic:
        require(not torch.cuda.is_initialized(), 'deterministic audit must configure the backend before CUDA initialization')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = training_device(device)
    torch.set_num_threads(2)
    manifest, data = load_bank(config)
    candidates = [i for i, r in enumerate(manifest["rows"]) if r["role"] == "incremental_fit"
                  and abs(data['scores'][i, 1]-data['scores'][i, 0]) > .01]
    require(candidates, 'SCORER_COMPARISON_SIGNAL_ABSENT: real two-stream resume probe requires a valid pair')
    candidates.sort(key=lambda i: digest([config.selection_seed, "resume_audit", manifest["rows"][i]["scene_id"]]))
    ix = candidates[:64]
    ids = [manifest["rows"][i]["scene_id"] for i in ix]
    by_id = {s.scene_id: s for s in load_scenes(config.root/"scenes.json")}
    position = {name: i for i, name in enumerate(ids)}
    cache = JointCache(config)
    frozen = cache.batch([by_id[i] for i in ids], device)
    truth = torch.as_tensor(data["scores"][ix], device=device)
    components = torch.as_tensor(data["components"][ix], device=device)
    trajectories = torch.as_tensor(data["trajectories"][ix], device=device)
    weights = comparison_weights(truth, torch.ones_like(truth, dtype=torch.bool)).cpu().numpy()
    def new():
        module = make_scorer(config)
        # Switch after one update only in this explicitly separate resume probe.
        sampler = TwoStreamSampler(ids, weights, natural_seed=config.natural_seed, comparison_seed=config.comparison_seed,
                                   pointwise_updates=1, bank_hash=file_hash(config.root/"PAIR_BANK.json"))
        return module, sampler
    def fetch(entries, device):
        indices = [position[e["scene_id"]] for e in entries]
        x = {k: v[indices] for k, v in frozen.items()}
        return {"context": score_context(config, x), "trajectories": trajectories[indices], "scores": truth[indices],
                "valid": torch.ones(len(indices),2,dtype=torch.bool,device=device), "components": components[indices],
                "component_valid": torch.ones_like(components[indices],dtype=torch.bool),
                "natural": torch.tensor([e["natural"] for e in entries],device=device),
                "comparison": torch.tensor([e["comparison"] for e in entries],device=device)}
    def loss(module,batch):return terms(module(batch["context"],batch["trajectories"]),batch)
    root=config.root/("real_resume_deterministic" if deterministic else "real_resume_audit")
    cfg=training_config(config,"scorer") # original 2,000-step LR timeline, bounded with stop_after
    deps={"purpose":"real_resume_only","base_binding":file_hash(config.root/"BASE_BINDING.json"),
          "bank":file_hash(config.root/"PAIR_BANK.json"),"scene_ids":ids}
    options=dict(device=device,weights={"value":1.,"components":.2,"ranking":.2},seed=config.training_rng_seed,
                 denominator_function=denominators)
    def run_to(folder, target):
        from iqe.training.resume import inspect_checkpoint
        checkpoints=sorted(folder.glob('step_*.pt.COMPLETE.json'))
        latest=0
        if checkpoints:
            latest=inspect_checkpoint(str(checkpoints[-1]).removesuffix('.COMPLETE.json'), deps, cfg)['optimizer_step']
        if latest >= target: return
        m,s=new()
        train(m,loss,fetch,s,cfg,deps,folder,resume='auto' if latest else None,stop_after=target,**options)
    run_to(root/"continuous",4)
    run_to(root/"split",2)
    run_to(root/"split",4)
    a=torch.load(root/"continuous/step_000004.pt",weights_only=False,map_location="cpu")
    b=torch.load(root/"split/step_000004.pt",weights_only=False,map_location="cpu")
    checks={k:equal_tree(a[k],b[k]) for k in ("module","optimizer","scheduler","rng_by_rank","sampler","consumed_sample_ids")}
    la=[json.loads(x) for x in (root/"continuous/steps.jsonl").read_text().splitlines()]
    lb=[json.loads(x) for x in (root/"split/steps.jsonl").read_text().splitlines()]
    checks["losses"]=[x["loss"] for x in la]==[x["loss"] for x in lb]
    checks["gradient_norms"]=[x["gradient_norm"] for x in la]==[x["gradient_norm"] for x in lb]
    result={"status":"PASS" if all(checks.values()) else "FAIL", "checks":checks,"real_scenes":len(ids),
            "continuous_steps":4,"split_steps":[2,2],"normal_dropout":config.scorer_dropout,
            "same_2000_step_schedule":True,"VLM_calls":0,"main_scorer_initialized_from_probe":False,
            "environment":a["environment"],"next_batch_restoration":read_json(root/"split/RESUME_RESOLUTION.json"),
            "deterministic_audit_only":deterministic,"main_training_backend_changed":False}
    atomic_json(config.root/("REAL_RESUME_DETERMINISTIC.json" if deterministic else "REAL_RESUME_AUDIT.json"),result)
    require(all(checks.values()),"real same-runtime resume mismatch; evidence preserved")
    return result


if __name__=='__main__':
    import argparse
    from .config import load_config
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--deterministic',action='store_true',help='separate deterministic-backend proof; preserves the production-backend evidence')
    a=p.parse_args();print(json.dumps(audit(load_config(a.config),a.device,a.deterministic)))
