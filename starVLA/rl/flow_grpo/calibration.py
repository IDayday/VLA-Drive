"""Fixed train calibration only; reports all noise settings without selecting seeds."""
from pathlib import Path
import json
import numpy as np
import torch
from .config import split_tokens
from .data import KeyedDataset
from .loading import load_policy
from .rollout import SamplingSpec, sample_chain, evaluate_transitions
from .observation import prepare_policy_observation
from .reward import RewardService
from .math import reduce_dimensions


def calibrate(cfg, sft, output):
    from infer import deal_action_1225, set_inference_seed

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    train, _ = split_tokens(cfg)
    tokens = train[: cfg["runtime"]["calibration_scenes"]]
    dataset = KeyedDataset(sft)
    policy = load_policy(cfg, sft).cuda().eval()
    if cfg["runtime"]["deepspeed_stage"]:
        policy.to(dtype=torch.bfloat16)
    service = RewardService(
        Path("navsim").resolve(),
        cfg["paths"]["metric_cache"],
        "train",
        tokens,
        workers=cfg["runtime"]["reward_workers"],
    )
    records = []
    try:
        for i, token in enumerate(tokens):
            sample = dataset[(token, cfg["runtime"]["seed"])]
            observation = prepare_policy_observation([sample])
            seed = cfg["runtime"]["seed"] + i
            set_inference_seed(seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ode = policy.predict_action_infer_1d([sample])["normalized_actions"]
            physical_ode = deal_action_1225(
                ode, act_norm=int(sft.datasets.vla_data.act_norm)
            )[:, None]
            ode_score = service.score([token], physical_ode)[0][0]
            record = {
                "token": token,
                "seed": seed,
                "sft_ode": vars(ode_score),
                "settings": [],
            }
            artifact = {
                "gt_normalized": sample["action"],
                "sft_ode": physical_ode[0, 0],
            }
            for noise in [0.05, 0.1, 0.2]:
                spec = SamplingSpec(
                    group_size=cfg["sampling"]["group_size"],
                    num_steps=cfg["sampling"]["num_steps"],
                    noise_level=noise,
                    temporal_noise_correlation=cfg["sampling"].get("temporal_noise_correlation", 0.0),
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    rollout = sample_chain(
                        policy,
                        observation,
                        spec,
                        0,
                        seed,
                        {"calibration_train_only": True},
                    )
                    current = evaluate_transitions(policy, observation, rollout)
                drift = (
                    reduce_dimensions(current["elementwise_logprob"])
                    - rollout.old_logprob
                )
                if drift.abs().max() > 1e-5:
                    raise AssertionError("no-update log-ratio mismatch")
                physical = deal_action_1225(
                    rollout.raw_final_action.cpu().numpy(),
                    act_norm=int(sft.datasets.vla_data.act_norm),
                )
                scores = service.score([token], physical)[0]
                record["settings"].append(
                    {
                        "noise_level": noise,
                        "scores": [vars(s) for s in scores],
                        "group_mean": float(np.mean([s.score for s in scores])),
                        "diagnostic_oracle": max(s.score for s in scores),
                        "xy_std": float(physical[0, :, :, :2].std(axis=0).mean()),
                        "max_abs_logratio": float(drift.abs().max()),
                    }
                )
                artifact[f"sde_{noise}"] = physical[0]
            np.savez(root / f"{token}.npz", **artifact)
            records.append(record)
            (root / "calibration.json").write_text(
                json.dumps(
                    {
                        "split": "train",
                        "network_dtype": str(
                            next(policy.action_model.parameters()).dtype
                        ),
                        "ode_scope": "original ODE at training runtime precision; deployment baseline is evaluated separately",
                        "parameter_selection": "none; retain task default 0.1",
                        "scenes": records,
                    },
                    indent=2,
                )
            )
            print(
                token,
                "ODE",
                ode_score.score,
                "SDE",
                [(r["noise_level"], r["group_mean"]) for r in record["settings"]],
                flush=True,
            )
    finally:
        service.close()
