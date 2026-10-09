"""A frozen global sampling plan; consumed cursor, never DataLoader prefetch, is authoritative."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random
from ..contracts import require
from ..io import digest

BUCKETS = ("hard_original", "new_labeled", "local_support", "global_anchor")


def mixture(buckets, ratios, *, explicit_anchor_zero=False):
    ratios = dict(ratios)
    require(set(ratios) == set(BUCKETS) and abs(sum(ratios.values()) - 1) < 1e-9 and min(ratios.values()) >= 0, "bucket proportions")
    require(ratios["global_anchor"] > 0 or explicit_anchor_zero, "anchor=0 needs explicit ablation")
    if ratios["global_anchor"] > 0:
        require(buckets["global_anchor"], "missing required global_anchor")
    transfers = {}
    for k in ("new_labeled", "local_support"):
        if not buckets[k]:
            transfers[k] = ratios[k]
            ratios["hard_original"] += ratios[k]
            ratios[k] = 0
    require(any(buckets[k] and ratios[k] > 0 for k in ("hard_original", "new_labeled")), "NO_ELIGIBLE_COVERAGE_GAP")
    transfers_to_new = 0.0
    if not buckets["hard_original"] and buckets["new_labeled"]:
        transfers_to_new = ratios["hard_original"]
        ratios["new_labeled"] += ratios["hard_original"]
        ratios["hard_original"] = 0.0
    for k in BUCKETS:
        require(buckets[k] or ratios[k] == 0, f"empty nonzero bucket {k}")
    return ratios, {"transfers_to_hard": transfers, "missing_hard_to_audited_new": transfers_to_new,
                    "experiment_kind": "new_observations" if any(s.source_kind in {"new_real", "synthetic"} for s in buckets["new_labeled"]) else "resampling_only"}


def sampling_plan(buckets, ratios, draws, seed=42, max_fraction=.10, explicit_anchor_zero=False):
    ratios, metadata = mixture(buckets, ratios, explicit_anchor_zero=explicit_anchor_zero)
    require(draws > 0 and 0 < max_fraction <= 1, "sampling budget")
    rng = random.Random(seed)
    counts = {k: math.floor(draws * ratios[k]) for k in BUCKETS}
    remainder = draws - sum(counts.values())
    priority = sorted(BUCKETS, key=lambda k: (-(draws * ratios[k] - counts[k]), BUCKETS.index(k)))
    for k in priority[:remainder]:
        counts[k] += 1
    plan = []
    metadata["buckets"] = {}
    for bucket in BUCKETS:
        rows = buckets[bucket]
        groups = defaultdict(list)
        for s in rows:
            s.eligible_input(external_targets=True)
            groups[s.source_group_id].append(s)
        group_ids = sorted(groups)
        achieved_cap = max(max_fraction, 1 / len(groups)) if groups else max_fraction
        metadata["buckets"][bucket] = {"records": len(rows), "unique_groups": len(groups), "draws": counts[bucket],
                                      "sampling_with_replacement": True, "group_fraction_cap": achieved_cap,
                                      "uniform_group_degradation": bool(groups) and 1 / len(groups) > max_fraction}
        # Balance the complete plan. Integer draws can exceed the requested
        # fractional cap in very short probes; record that unavoidable rounding.
        exposure = Counter()
        for i in range(counts[bucket]):
            least = min(exposure[g] for g in group_ids)
            group = rng.choice([g for g in group_ids if exposure[g] == least])
            scene = rng.choice(sorted(groups[group], key=lambda s: s.scene_id))
            exposure[group] += 1
            plan.append({"scene_id": scene.scene_id, "source_group_id": group, "bucket": bucket,
                         "target_id": scene.target_id, "augmentation_seed": rng.randrange(2**32)})
        metadata["buckets"][bucket]["exposures_by_group"] = dict(exposure)
        actual_max = max(exposure.values(), default=0)
        quota = math.ceil(counts[bucket] * achieved_cap)
        require(actual_max <= quota, "cumulative source group quota exceeded")
        metadata["buckets"][bucket].update(integer_group_quota=quota,
            actual_max_group_fraction=actual_max / counts[bucket] if counts[bucket] else 0.,
            fractional_cap_rounding=bool(counts[bucket] and actual_max / counts[bucket] > achieved_cap + 1e-12),
            minimum_feasible_integer_fraction=math.ceil(counts[bucket] / len(groups)) / counts[bucket] if counts[bucket] else 0.)
        drawn = [p["scene_id"] for p in plan if p["bucket"] == bucket]
        metadata["buckets"][bucket]["replacement"] = len(set(drawn)) < len(drawn)
    rng.shuffle(plan)
    metadata["draws"] = len(plan)
    metadata["unique_observations"] = len({s.observation_hash for k in BUCKETS for s in buckets[k]})
    metadata["source_kinds"] = {kind: {"records": sum(s.source_kind == kind for rows in buckets.values() for s in rows),
        "unique_observations": len({s.observation_hash for rows in buckets.values() for s in rows if s.source_kind == kind})}
        for kind in ("original", "retrieved", "new_real", "synthetic", "alternative_target")}
    metadata["actual_proportions"] = {k: counts[k] / draws for k in BUCKETS}
    metadata["exposures_by_scene"] = dict(Counter(p["scene_id"] for p in plan))
    return {"entries": plan, "metadata": metadata, "plan_hash": digest(plan)}


class ConsumedSampler:
    def __init__(self, plan, cursor=0):
        require(plan["plan_hash"] == digest(plan["entries"]), "sampler plan corrupted")
        self.plan = plan
        self.cursor = cursor
        require(0 <= cursor <= len(plan["entries"]), "sampler cursor invalid")

    def peek(self, n):
        return self.plan["entries"][self.cursor:self.cursor + n]

    def consume(self, n):
        require(n >= 0 and self.cursor + n <= len(self.plan["entries"]), "sampler over-consumption")
        self.cursor += n

    def state_dict(self):
        return {"plan_hash": self.plan["plan_hash"], "consumed_cursor": self.cursor}

    def load_state_dict(self, state):
        require(state["plan_hash"] == self.plan["plan_hash"], "sampler version conflict")
        require(0 <= state["consumed_cursor"] <= len(self.plan["entries"]), "sampler cursor invalid")
        self.cursor = state["consumed_cursor"]
