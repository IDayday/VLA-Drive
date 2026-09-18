"""Merge sufficient statistics, never means of extrema or rank quantiles."""
import math
import torch


class Metrics:
    def __init__(self, state=None):
        self.state = state or {}

    def add(self, name, kind, numerator, denominator=1):
        if kind not in {"min", "max", "count", "mean", "fraction", "quantile"}:
            raise ValueError(f"unknown metric reduction: {kind}")
        if kind == "quantile":
            value = [float(x) for x in numerator]
        else:
            value = [float(numerator), float(denominator)]
        entry = {"kind": kind, "value": value}
        if name in self.state:
            entry = self._merge(self.state[name], entry)
        self.state[name] = entry

    @staticmethod
    def _merge(a, b):
        kind = a["kind"]
        if kind != b["kind"]:
            raise ValueError("metric schema changed")
        x, y = a["value"], b["value"]
        if kind == "quantile":
            value = x + y
        elif kind in {"min", "max"}:
            value = [(min if kind == "min" else max)(x[0], y[0]), 1]
        else:
            value = [x[0] + y[0], x[1] + y[1]]
        return {"kind": kind, "value": value}

    def merge(self, other):
        for name, entry in other.state.items():
            self.state[name] = (
                self._merge(self.state[name], entry) if name in self.state else entry
            )
        return self

    def result(self):
        result = {}
        for name, entry in self.state.items():
            kind, value = entry["kind"], entry["value"]
            if kind == "quantile":
                # All actual ratio samples, deterministic order-independent quantiles.
                values = torch.tensor(sorted(value), dtype=torch.float64)
                result[name] = {
                    str(q): float(torch.quantile(values, q))
                    for q in (0.0, 0.01, 0.5, 0.99, 1.0)
                }
            elif kind in {"mean", "fraction"}:
                result[name] = value[0] / value[1] if value[1] else None
            else:
                result[name] = value[0]
        return result

    def ratios(self, prefix, ratio, clip):
        values = ratio.detach().double().flatten().cpu()
        if not torch.isfinite(values).all() or not len(values):
            raise ValueError("empty/nonfinite ratios")
        self.add(prefix + "_min", "min", values.min())
        self.add(prefix + "_max", "max", values.max())
        self.add(prefix + "_mean", "mean", values.sum(), len(values))
        self.add(
            prefix + "_clip_fraction",
            "fraction",
            ((values - 1).abs() > clip).sum(),
            len(values),
        )
        self.add(prefix + "_count", "count", len(values))
        self.add(prefix + "_quantiles", "quantile", values.tolist())

    def update_scene(self, result, rollout, clip):
        valid = rollout.transition_mask
        self.ratios("pre_update_ratio", result["ratio"][valid], clip)
        for name in ("loss", "grpo", "reference", "sft"):
            self.add(name, "mean", float(result[name].detach()))
        self.add(
            "old_policy_drift",
            "mean",
            float(result["logratio"][valid].double().square().sum() / 2),
            int(valid.sum()),
        )
        reward = rollout.rewards.detach().double()
        self.add("reward", "mean", reward.sum(), reward.numel())
        self.add(
            "reward_zero_fraction", "fraction", (reward == 0).sum(), reward.numel()
        )
        self.add("all_zero_groups", "fraction", (reward == 0).all(1).sum(), len(reward))
        self.add(
            "all_equal_groups",
            "fraction",
            (reward.std(1, unbiased=False) == 0).sum(),
            len(reward),
        )
        self.add("scene_count", "count", len(reward))
        advantage = rollout.advantages.detach().double()
        self.add("reward_group_std_quantiles", "quantile", reward.std(1, unbiased=False).tolist())
        self.add("advantage_absolute_mean", "mean", advantage.abs().sum(), advantage.numel())
        self.add("advantage_second_moment", "mean", advantage.square().sum(), advantage.numel())
        self.add("advantage_group_mean_abs_max", "max", advantage.mean(1).abs().max())
        self.add("advantage_nonzero_group_fraction", "fraction",
                 (advantage.abs().max(1).values > 0).sum(), len(advantage))
        statistics = getattr(rollout, "advantage_statistics", None) or {}
        if "global_reward_std" in statistics:
            self.add("behavior_global_reward_std", "mean", statistics["global_reward_std"])
            self.add("behavior_global_candidate_count", "mean", statistics["global_candidate_count"])
        if getattr(rollout, "physical_trajectories", None) is not None:
            from .diversity import trajectory_diversity

            for physical, scores in zip(rollout.physical_trajectories, reward.cpu().numpy()):
                d = trajectory_diversity(physical, scores)
                self.add("candidate_pair_ade_scene_median_m", "mean", d["pair_ade_m"]["0.5"])
                self.add("candidate_pair_fde_scene_median_m", "mean", d["pair_fde_m"]["0.5"])
                self.add("candidate_xy_effective_rank", "mean", d["xy_covariance_effective_rank"])
                self.add("reward_group_span_quantiles", "quantile", [d["reward_span"]])
        self.add("candidate_count", "count", reward.numel())
        self.add("replay_scene_count", "count", 1)
        for key, value in result["components"].items():
            self.add("sft_component/" + key, "mean", float(value.detach()))
        for row in rollout.score_records:
            for record in row:
                for key, value in record.metrics.items():
                    if value is not None and math.isfinite(value):
                        self.add("reward_component/" + key, "mean", value)
