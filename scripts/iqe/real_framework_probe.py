"""Bounded real-image interface probe. Not a trained S0 or two-round smoke."""
from pathlib import Path
import argparse
import time
import torch
from omegaconf import OmegaConf
from iqe.config import load_config
from iqe.io import read_json, atomic_json, digest, atomic_torch
from iqe.data.manifests import load_scenes
from iqe.data.sources import load_observation, load_target
from iqe.query_base import build_query_framework
from iqe.s0_adapter import S0Adapter
from iqe.model import IQEModel
from iqe.losses import reduce_terms
from iqe.evaluation.retention import frozen_snapshot
from iqe.resources import qualify


def run(config_path, output, device='cpu'):
    config = load_config(config_path)
    qualify(device)
    torch.set_num_threads(2)
    c = read_json(config["s0"]["contract"])
    scene = next(s for s in load_scenes(Path(config["output_root"]) / "scenes.json") if s.split_role == "incremental_fit")
    start = time.perf_counter()
    framework = build_query_framework(OmegaConf.create(c["source_config"]), c["query_architecture"], c["source_root"], c["source_commit"]).float().to(device).eval()
    framework.strip_auxiliary_heads()
    observation = load_observation(scene, c)
    model = IQEModel(S0Adapter(framework, c)).eval()
    with torch.no_grad():
        # Actual original entry and wrapped K1 use the same preprocessed real images.
        original = framework.predict_action([observation])
        candidates = model.forward_candidates([observation])
    torch.testing.assert_close(original, candidates.trajectories[:,0], atol=1e-5, rtol=1e-6)
    f = candidates.features
    model.append_expert("expert_1")
    with torch.no_grad():
        clone = model.forward_candidates(None,features=f)
    torch.testing.assert_close(clone.raw[:,0],clone.raw[:,1],atol=1e-6,rtol=1e-6)
    before = frozen_snapshot(model, "expert_1")
    model.set_trainable_stage("expert_train","expert_1");model.train()
    optimizer = torch.optim.AdamW(model.optimizer_parameters(),lr=3e-4)
    target = load_target(scene,model.experts["expert_0"].trajectory_contract)[None].to(device)
    prediction = model.experts["expert_1"](f)
    loss, _ = reduce_terms(model.adapter.compute_expert_il_loss(prediction,target))
    loss.backward()
    gradients = {name: {"finite": bool(torch.isfinite(p.grad).all()), "norm": float(p.grad.norm())} for name,p in model.experts["expert_1"].named_parameters() if p.grad is not None}
    optimizer.step()
    assert frozen_snapshot(model,"expert_1") == before
    model.eval()
    with torch.no_grad():
        updated = model.experts["expert_1"](f)
    atomic_torch(Path(output)/"features.pt", {"scene":f.scene,"ego":f.ego,"valid_tokens":f.valid_tokens,"contract_hash":f.contract_hash,"scene_ids":f.scene_ids,"conditions":f.conditions})
    atomic_torch(Path(output)/"expert_step_000001.pt",model.experts["expert_1"].state_dict())
    result = {"status":"PASS_REAL_IMAGE_INTERFACE_PROBE", "scene_id":scene.scene_id,"observation_ref":scene.observation_ref,
        "real_generic_Qwen":c["generic_vlm"],"source_commit":c["source_commit"],"learned_driving_weights_loaded":False,
        "original_vs_wrapped_max_abs":float((original-candidates.trajectories[:,0]).abs().max()),
        "clone_raw_max_abs":float((clone.raw[:,0]-clone.raw[:,1]).abs().max()),"IL_loss":float(loss),
        "private_expert_update_raw_max_abs":float((updated.raw-clone.raw[:,1]).abs().max()),"gradients":gradients,
        "frozen_hashes_unchanged":True,"scene_memory_shape":list(f.scene.shape),"ego_shape":list(f.ego.shape),
        "sequence_lengths":framework.last_sequence_lengths.tolist(),"elapsed_seconds":time.perf_counter()-start,
        "device":device,"threads":2,"science":"UNTESTED","not_a_trained_S0":True,"not_two_round_smoke":True}
    atomic_json(Path(output)/"RESULT.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();run(a.config,a.output,a.device)
