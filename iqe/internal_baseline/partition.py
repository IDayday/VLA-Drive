"""Immutable expert/scorer subroles, grouped before looking at any scores."""
from collections import Counter
from iqe.contracts import require
from iqe.data.splits import source_components
from iqe.io import atomic_json, read_json, digest


def partition_fit(config, scenes):
    fit = [s for s in scenes if s.split_role == "incremental_fit"]
    roots = source_components(scenes)
    groups = sorted({roots[s.scene_id] for s in fit}, key=lambda g: digest([config.partition_seed, g]))
    require(len(groups) >= 2, "need at least two independent fit log/source components")
    boundary = min(len(groups)-1, max(1, round(.8*len(groups))))
    assignments = {g: "expert_fit" if i < boundary else "scorer_fit" for i, g in enumerate(groups)}
    chosen, roles = [], {}
    for role, cap in (("expert_fit", config.expert_fit_limit), ("scorer_fit", config.scorer_fit_limit)):
        available = [s for s in fit if assignments[roots[s.scene_id]] == role]
        ordered = sorted(available, key=lambda s: digest([config.selection_seed, role, s.scene_id]))
        rows = ordered if config.fit_limit == "all_fit" else ordered[:cap]
        require(rows, "empty assigned subrole: "+role)
        chosen.extend(rows)
        roles[role] = {"scene_ids": [s.scene_id for s in rows], "scene_ids_hash": digest([s.scene_id for s in rows]),
                       "source_logs": sorted({s.source_log_id for s in rows}),
                       "source_groups": sorted({s.source_group_id for s in rows}),
                       "assigned_components": [g for g in groups if assignments[g] == role],
                       "available_scenes": len(available), "selected_scenes": len(rows)}
    for field in ("source_logs", "source_groups", "assigned_components"):
        require(not set(roles["expert_fit"][field]) & set(roles["scorer_fit"][field]), "D_E/D_S group leakage")
    expert_ids, scorer_ids = set(roles["expert_fit"]["scene_ids"]), set(roles["scorer_fit"]["scene_ids"])
    require(not ({s.observation_hash for s in chosen if s.scene_id in expert_ids} &
                 {s.observation_hash for s in chosen if s.scene_id in scorer_ids}), "D_E/D_S observation leakage")
    report = {"schema": "two_candidate_partition_v1", "protocol": config.research_protocol,
              "partition_seed": config.partition_seed, "selection_seed": config.selection_seed,
              "grouping": "source_log_and_source_parent_union", "group_split": [boundary, len(groups)-boundary],
              "roles": roles, "all_assignments": assignments, "selected_before_scores": True,
              "scorer_fit_scope": "held out from NEW expert specialization; not necessarily unseen by Base"}
    report["hash"] = digest(report)
    atomic_json(config.root/"DATA_PARTITION.json", report, immutable=True)
    return chosen


def subrole_ids(config, role):
    require(role in {"expert_fit", "scorer_fit"}, "unknown incremental subrole")
    p = read_json(config.root/"DATA_PARTITION.json")
    require(p["hash"] == digest({k:v for k,v in p.items() if k != "hash"}), "partition changed")
    return set(p["roles"][role]["scene_ids"])


def training_ids(config, scenes, role):
    if config.research_protocol != "two_candidate_v1":
        return {s.scene_id for s in scenes if s.split_role == "incremental_fit"}
    selected = subrole_ids(config, role)
    lookup = {s.scene_id: s for s in scenes}
    require(all(i in lookup and lookup[i].split_role == "incremental_fit" for i in selected), "subrole scene/role mismatch")
    return selected


def finalize_partition(config, scenes):
    p = read_json(config.root/"DATA_PARTITION.json")
    evidence = {}
    for role in ("expert_fit", "scorer_fit", "stage_val", "selector_cal", "dev_report"):
        ids = subrole_ids(config, role) if role.endswith("_fit") else {s.scene_id for s in scenes if s.split_role == role}
        rows = [s for s in scenes if s.scene_id in ids]
        evidence[role] = {"scenes": len(rows), "logs": len({s.source_log_id for s in rows}),
                          "groups": len({s.source_group_id for s in rows}), "base_seen": dict(Counter(s.s0_seen for s in rows)),
                          "scene_ids": sorted(ids), "scene_hash": digest(sorted(ids))}
    atomic_json(config.root/"DATA_PARTITION_EXPOSURE.json", {"partition_hash": p["hash"], "roles": evidence,
                "final_test_accessed": False, "navtest_accessed": False}, immutable=True)
