"""Log/source-parent union grouping prevents window and synthetic-parent leakage."""
from __future__ import annotations
from dataclasses import replace
from collections import Counter
from ..contracts import require
from ..io import digest


def source_components(scenes):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for s in scenes:
        union("log:" + s.source_log_id, "group:" + s.source_group_id)
        if s.parent_group_id:
            union("group:" + s.source_group_id, "group:" + s.parent_group_id)
    return {s.scene_id: find("group:" + s.source_group_id) for s in scenes}


def validate_isolation(scenes):
    roots = source_components(scenes)
    groups = {}
    observations = {}
    for s in scenes:
        groups.setdefault(roots[s.scene_id], set()).add(s.split_role)
        observations.setdefault(s.observation_hash, set()).add(s.split_role)
    require(all(len(v) == 1 for v in groups.values()), "log/source-parent group split leakage")
    require(all(len(v) == 1 for v in observations.values()), "observation hash split leakage")
    return {"roles": dict(Counter(s.split_role for s in scenes)),
            "logs": {r: len({s.source_log_id for s in scenes if s.split_role == r}) for r in {s.split_role for s in scenes}},
            "source_groups": len(groups), "unique_observations": len(observations), "isolation": "PASS"}


def split_training_logs(scenes, ratios=(.9, .05, .05), seed=42):
    require(len(ratios) == 3 and abs(sum(ratios) - 1) < 1e-9 and min(ratios) > 0, "split ratios")
    roots = source_components(scenes)
    roles = ("incremental_fit", "stage_val", "selector_cal")
    locked = {}
    for s in scenes:
        if s.split_role in {"dev_report", "final_test"}:
            locked.setdefault(roots[s.scene_id], set()).add(s.split_role)
    require(all(len(v) == 1 for v in locked.values()), "dev/test source overlap")
    result = []
    for s in scenes:
        key = roots[s.scene_id]
        if key in locked:
            require(s.split_role in locked[key], "allowed train group overlaps dev/test")
            result.append(s)
            continue
        value = int(digest({"group": key, "seed": seed})[:16], 16) / 2**64
        role = roles[0] if value < ratios[0] else roles[1] if value < ratios[0] + ratios[1] else roles[2]
        result.append(replace(s, split_role=role))
    validate_isolation(result)
    return result
