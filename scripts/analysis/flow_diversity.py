"""Fixed real-scene exploration audit; no policy update and no best-of-N inference.

Compare original ODE with independent initial noise against the exact training
SDE at several explicitly declared strengths. Save every candidate and full SDE
chain. The first eight of the same sixteen candidates isolate group-size effects
without resampling or selecting easy scenes. Only original F-SFT is supported.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens
from starVLA.rl.flow_grpo.loading import load_policy, file_sha
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.diversity import trajectory_diversity
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.rollout import SamplingSpec, sample_chain, evaluate_transitions
from starVLA.rl.flow_grpo.math import reduce_dimensions
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def publish(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--noise-levels", nargs="+", type=float, default=[.05, .1, .2, .3])
    p.add_argument("--temporal-correlations", nargs="+", type=float, default=[0.0])
    p.add_argument("--transition-modes", nargs="+", choices=["flow_sde", "euler_gaussian"], default=["flow_sde"])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if any(n <= 0 for n in a.noise_levels) or len(a.noise_levels) != len(set(a.noise_levels)):
        p.error("positive unique SDE noise levels required")
    from starVLA.rl.flow_grpo.temporal_noise import validate_correlation
    for rho in a.temporal_correlations:
        validate_correlation(rho)
    if len(a.temporal_correlations) != len(set(a.temporal_correlations)):
        p.error("unique temporal correlations required")
    if len(a.transition_modes) != len(set(a.transition_modes)):
        p.error("unique transition modes required")
    cfg, sft = resolve_config(a.config)
    if cfg["checkpoint_contract"]["variant"] != "frozen_visual":
        p.error("this experiment is F-only")
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    tokens = json.loads(Path(a.tokens).read_text())
    train, dev = split_tokens(cfg)
    if not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= set(train):
        raise ValueError("unique, explicit training-only diagnostic scenes required")
    configure_numerics()
    from starVLA.rl.flow_grpo.audit import source_fingerprints
    from starVLA.rl.flow_grpo.reward import RewardService, reward_metadata
    from infer import deal_action_1225
    identity = {"scope": "F-SFT exploration diagnostic; no training, no dev/navtest selection",
        "checkpoint_sha256": cfg["checkpoint_contract"]["sha256"], "seed": a.seed,
        "tokens": tokens, "tokens_sha256": file_sha(a.tokens), "source": source_fingerprints(),
        "script_sha256": file_sha(__file__), "group_size": 16, "noise_levels": a.noise_levels,
        "temporal_correlations": a.temporal_correlations,
        "transition_modes": a.transition_modes,
        "reward_protocol": reward_metadata(Path("navsim").resolve()), "candidate_chunk_size": 1,
        "dtype": "BF16 model; inherited FP32 action kernel", "num_steps": cfg["sampling"]["num_steps"]}
    publish(root/"identity.json", identity)
    records = []
    started = time.time()
    service = None
    try:
        policy = load_policy(cfg, sft).cuda().to(dtype=torch.bfloat16).eval()
        dataset = KeyedDataset(sft)
        service = RewardService(Path("navsim").resolve(), cfg["paths"]["metric_cache"],
            "train", tokens, workers=2)
        for token in tokens:
            seed = int.from_bytes(hashlib.sha256(f"{a.seed}:{token}".encode()).digest()[:8], "little") % (2**63-1)
            sample = dataset[(token, a.seed)]
            observation = prepare_policy_observation([sample])
            artifact = {}
            settings = {}
            initial = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for noise, rho, mode in [(n, r, m) for n in a.noise_levels
                                        for r in a.temporal_correlations for m in a.transition_modes]:
                    spec = SamplingSpec(group_size=16, num_steps=cfg["sampling"]["num_steps"], noise_level=noise,
                                        temporal_noise_correlation=rho, transition_mode=mode)
                    rollout = sample_chain(policy, observation, spec, 0, seed, {"diversity_only": True})
                    if initial is None:
                        initial = rollout.chain[:, :, 0].clone()
                    elif not torch.equal(initial, rollout.chain[:, :, 0]):
                        raise AssertionError("initial noises differ across SDE settings")
                    # Every shard checks a real fixed-chain ratio, not synthetic probabilities.
                    max_ratio_error = None
                    if token == tokens[0]:
                        current = evaluate_transitions(policy, observation, rollout)
                        drift = reduce_dimensions(current["elementwise_logprob"])-rollout.old_logprob
                        max_ratio_error = float(drift.abs().max())
                        if max_ratio_error > 1e-5:
                            raise AssertionError("no-update logratio mismatch")
                        del current
                    physical = deal_action_1225(rollout.raw_final_action.cpu().numpy(),
                                                act_norm=int(sft.datasets.vla_data.act_norm))[0]
                    scores = service.score([token], physical[None])[0]
                    chain = rollout.chain[0].cpu().numpy()
                    name = (f"sde_{noise}" if mode == "flow_sde" else f"{mode}_{noise}") + (f"_rho{rho}" if rho else "")
                    settings[name] = {"scores": [asdict(s) for s in scores], "logratio_error": max_ratio_error,
                        "temporal_noise_correlation": rho,
                        "transition_mode": mode,
                        "g16": trajectory_diversity(physical, [s.score for s in scores], chain),
                        "g8_prefix": trajectory_diversity(physical[:8], [s.score for s in scores[:8]], chain[:8])}
                    artifact[name] = physical
                    artifact[name+"_chain"] = chain
                    del rollout
                # Original Euler solver, identical initial noise per candidate and chunk=1.
                condition = policy.encode_policy_condition(observation)
                with torch.autocast("cuda", dtype=torch.float32):
                    ode = torch.stack([policy.action_model._euler_sample(initial[:, g], condition)
                                       for g in range(16)], dim=1)
                physical = deal_action_1225(ode.cpu().numpy(), act_norm=int(sft.datasets.vla_data.act_norm))[0]
                scores = service.score([token], physical[None])[0]
                settings["ode"] = {"scores": [asdict(s) for s in scores],
                    "g16": trajectory_diversity(physical, [s.score for s in scores]),
                    "g8_prefix": trajectory_diversity(physical[:8], [s.score for s in scores[:8]])}
                artifact["ode"] = physical
                artifact["initial_noise"] = initial.cpu().numpy()[0]
            np.savez(root/(token+".npz"), **artifact)
            records.append({"token": token, "noise_key": seed, "settings": settings})
            publish(root/"results.json", {"identity": identity, "scenes": records})
            print(token, {k: {"ade_median": v["g16"]["pair_ade_m"]["0.5"],
                              "reward_std": v["g16"]["reward_std"]} for k, v in settings.items()}, flush=True)
        publish(root/"COMPLETE", {"scenes": len(records), "seconds": time.time()-started,
                                  "results_sha256": file_sha(root/"results.json")})
    except BaseException as exc:
        publish(root/"failure.json", {"status": "FAIL", "error": repr(exc), "completed_scenes": len(records)})
        raise
    finally:
        if service is not None:
            service.close()


if __name__ == "__main__":
    main()
