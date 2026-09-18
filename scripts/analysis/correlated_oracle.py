"""Real CUDA source oracle and correlated-transition numerical checks; no update."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from starVLA.rl.flow_grpo.audit import capture_source_environment
from starVLA.rl.flow_grpo.checkpoint import capture_rng, restore_rng
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.data import KeyedDataset, to_device
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.math import conditional_kl, reduce_dimensions
from starVLA.rl.flow_grpo.model import make_reference
from starVLA.rl.flow_grpo.numerical_checks import precision_gate
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import SamplingSpec, sample_chain, evaluate_transitions
from starVLA.rl.flow_grpo.source_oracle import action_only_source


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg, sft = resolve_config(a.config)
    configure_numerics()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    capture_source_environment(output, cfg)
    token = json.loads(Path(a.manifest).read_text())["tokens"][0]
    sample = KeyedDataset(sft)[(token, 42)]
    observation = prepare_policy_observation([sample])
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    policy._inference_qwen_forward_mode = "optimized"
    report = {"scope": "real F-SFT CUDA oracle; no optimizer update", "tests": {},
              "token": token, "checkpoint_sha256": policy._flow_source_sha256,
              "views": [im.size for im in sample["image"]]}

    def record(name, function):
        try:
            report["tests"][name] = {"status": "PASS", "result": function()}
        except Exception as error:
            traceback.print_exc()
            report["tests"][name] = {"status": "FAIL", "error": repr(error)}
        (output / "oracle.json").write_text(json.dumps(report, indent=2))
        print(name, report["tests"][name], flush=True)

    def source_oracle():
        state = capture_rng(policy)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with action_only_source(policy, cfg["checkpoint_contract"]):
                original = policy.predict_action_infer_1d([sample])["normalized_actions"]
            restore_rng(state, policy)
            current = policy.predict_action_infer_1d([sample])["normalized_actions"]
        np.testing.assert_allclose(current, original, atol=2e-5, rtol=2e-5)
        state = capture_rng(policy)
        batch = [to_device(sample, "cuda")]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with action_only_source(policy, cfg["checkpoint_contract"]):
                old = policy(batch)
            restore_rng(state, policy)
            new = policy.compute_sft_losses(batch)
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=2e-6, atol=2e-6)
        return {"ode_max_abs": float(np.abs(current-original).max()),
                "sft_old": {k: float(v) for k, v in old.items()},
                "sft_new": {k: float(v) for k, v in new.items()}}

    record("source_ode_sft_cuda", source_oracle)
    reference = make_reference(policy)
    seed = int.from_bytes(hashlib.sha256(f"42:{token}".encode()).digest()[:8], "little") % (2**63-1)
    for rho in (0., .8):
        for noise in (.1, .2):
            spec = SamplingSpec(group_size=16, noise_level=noise, temporal_noise_correlation=rho)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rollout = sample_chain(policy, observation, spec, 0, seed, {})

            def checks():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    current = evaluate_transitions(policy, observation, rollout)
                    ref = evaluate_transitions(reference, observation, rollout)
                logratio = reduce_dimensions(current["elementwise_logprob"]) - rollout.old_logprob
                torch.testing.assert_close(logratio, torch.zeros_like(logratio), rtol=0, atol=1e-5)
                kl = conditional_kl(current["mean"], current["std"], ref["mean"], ref["std"], rho)
                torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=1e-7)
                precision = precision_gate(policy, observation, rollout)
                return {"logratio_max_abs": float(logratio.abs().max()), "kl_max_abs": float(kl.abs().max()),
                        "independent_fp64_probability": precision}

            record(f"rho{rho}_noise{noise}", checks)
    report["status"] = "PASS" if all(t["status"] == "PASS" for t in report["tests"].values()) else "FAIL"
    (output / "oracle.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise AssertionError("real oracle failures retained")


if __name__ == "__main__":
    main()
