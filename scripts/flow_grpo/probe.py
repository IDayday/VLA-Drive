"""Real checkpoint/data preflight, with strict model loading."""
from pathlib import Path
import json
import torch
from infer import deal_action_1225
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.contracts import optimizer_groups, parameter_manifest
from starVLA.rl.flow_grpo.data import KeyedDataset, to_device
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reward import RewardService

cfg, sft = resolve_config("configs/flow_grpo/full_sft.yaml")
train, val = split_tokens(cfg)
dataset = KeyedDataset(sft)
print("SCENE", train[0], flush=True)
sample = dataset[(train[0], 42)]
print("DATA", sample.keys(), [im.size for im in sample["image"]], flush=True)
model = load_policy(cfg, sft).cuda().eval()
groups = optimizer_groups(model, sft)
manifest = parameter_manifest(model, groups)
Path("reports/ddp_flow_grpo/sft_parameter_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)
print(
    "MODEL",
    sum(p.numel() for p in model.parameters()),
    manifest["trainable_numel"],
    flush=True,
)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    hidden, _ = model.encode_policy_features([sample])
    print("HIDDEN", hidden.dtype, flush=True)
    condition = model.encode_policy_condition(prepare_policy_observation([sample]))
    print("CONDITION", condition.shape, condition.dtype, flush=True)
    out = model.predict_action_infer_1d_legacy([sample])
    print("ODE", out, flush=True)
    losses = model([to_device(sample, "cuda")])
    print("SFT", {k: float(v) for k, v in losses.items()}, flush=True)
print("ALLOCATED", torch.cuda.max_memory_allocated() / 2**30, flush=True)

reward = RewardService(
    "navsim", cfg["paths"]["metric_cache"], "train", [train[0]], workers=0
)
print(
    "REWARD",
    reward.score(
        [train[0]], deal_action_1225(out["normalized_actions"], act_norm=1)[:, None]
    ),
    flush=True,
)
reward.close()
