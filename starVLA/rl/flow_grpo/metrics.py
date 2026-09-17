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
        self.add("candidate_count", "count", reward.numel())
        self.add("replay_scene_count", "count", 1)
        for key, value in result["components"].items():
            self.add("sft_component/" + key, "mean", float(value.detach()))
        for row in rollout.score_records:
            for record in row:
                for key, value in record.metrics.items():
                    if value is not None and math.isfinite(value):
                        self.add("reward_component/" + key, "mean", value)
