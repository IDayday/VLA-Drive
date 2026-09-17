import torch, json
from dataclasses import replace
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.loading import load_policy, enable_checkpointing
from starVLA.rl.flow_grpo.contracts import apply_rl_freezes
from starVLA.rl.flow_grpo.rollout import (
    sample_chain,
    SamplingSpec,
    evaluate_transitions,
)
from starVLA.rl.flow_grpo.math import reduce_dimensions, clipped_surrogate

torch.use_deterministic_algorithms(True)
cfg, sft = resolve_config("configs/flow_grpo/action_only_frozen_visual.yaml")
train, _ = split_tokens(cfg)
obs = prepare_policy_observation([KeyedDataset(sft)[(train[0], 42)]])
p = load_policy(cfg, sft).cuda().eval().bfloat16()
apply_rl_freezes(p, cfg)
enable_checkpointing(p, True)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    r = sample_chain(p, obs, SamplingSpec(group_size=2), 0, 42, {})
orig = p.encode_policy_condition
conds = []


def encode(*a, **kw):
    c = orig(*a, **kw)
    c.retain_grad()
    conds.append(c)
    return c


p.encode_policy_condition = encode
snapshots = []
for chunk in (1, 2):
    p.zero_grad(set_to_none=True)
    r.spec = replace(r.spec, candidate_chunk_size=chunk)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        stats = evaluate_transitions(p, obs, r, checkpoint=True)
        lp = reduce_dimensions(stats["elementwise_logprob"])
        loss = clipped_surrogate(
            lp, r.old_logprob, torch.tensor([-1.0, 1.0], device="cuda")[None, :, None]
        )[0].mean()
    loss.backward()
    snapshot = {
        n: v.grad.detach().float().cpu().clone()
        for n, v in p.named_parameters()
        if v.grad is not None
    }
    snapshot["condition"] = conds[-1].grad.detach().float().cpu()
    snapshot["mean"] = stats["mean"].detach().cpu()
    snapshot["logp"] = lp.detach().cpu()
    print(
        "CHUNK",
        chunk,
        "condition_dtype",
        conds[-1].dtype,
        "loss",
        float(loss),
        flush=True,
    )
    if not snapshots:
        snapshots.append(snapshot)
    else:
        out = {}
        for n, x in snapshot.items():
            y = snapshots[0][n]
            d = x - y
            out[n] = {
                "maxabs": float(d.abs().max()),
                "norm": float(y.norm()),
                "rel_l2": float(d.norm() / y.norm().clamp_min(1e-30)),
                "mismatch": int((d.abs() > (2e-6 + 2e-3 * y.abs())).sum()),
            }
        open("reports/ddp_flow_grpo/chunk_gradient_probe.json", "w").write(
            json.dumps(out, indent=2)
        )
        print(
            {
                n: v
                for n, v in out.items()
                if n in ["condition", "mean", "logp"] or n.endswith("qwen_proj.weight")
            },
            flush=True,
        )
    del stats, lp, loss, conds[:]
