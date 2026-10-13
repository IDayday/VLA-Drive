"""An explicit pre-training budget amendment without rebinding Base or data.

The original Config and BASE_BINDING remain immutable. Only the optimizer
budgets and the pointwise/comparison boundary can change here, before either
head has consumed any updates. All training checkpoints bind this receipt.
"""
from datetime import datetime, timezone
from iqe.contracts import require
from iqe.io import atomic_json, read_json, file_hash, lock


FIELDS = ("expert_updates", "scorer_updates", "pointwise_updates")
FILE = "TRAINING_BUDGET_AMENDMENT.json"
SCHEMA = "two_candidate_training_budget_amendment_v1"


def original_budget(config):
    return {name: getattr(config, name) for name in FIELDS}


def retained_recipe(config):
    return {"global_batch": 128, "lr": 1e-4, "weight_decay": .01,
            "expert_warmup": 50, "scorer_warmup": 100,
            "post_warmup_schedule": "constant", "expert_hard_fraction": .8,
            "scorer_loss_weights": {"value": 1., "components": .2, "ranking": .2},
            "validation_interval": config.validation_interval, "clip_norm": 1.,
            "module_init_seed": config.module_init_seed, "training_rng_seed": config.training_rng_seed,
            "natural_seed": config.natural_seed, "comparison_seed": config.comparison_seed,
            "selection_seed": config.selection_seed, "partition_seed": config.partition_seed,
            "scorer_dropout": config.scorer_dropout,
            "expert_checkpoint_selection": "stage_val_oracle_then_hard_val_gain_then_earlier",
            "scorer_checkpoint_selection": "stage_val_raw_argmax_then_safety_then_earlier",
            "dev_used_for_budget_selection": False, "planned_pause_resume": False}


def _validate_target(target, initial):
    require(set(target) == set(FIELDS), "only three training-budget fields may be amended")
    require(all(type(value) is int for value in target.values()), "integer update budgets required")
    require(initial["expert_updates"] <= target["expert_updates"] <= 10000,
            "expert budget must be an explicit extension within 10000")
    require(initial["scorer_updates"] <= target["scorer_updates"] <= 20000,
            "scorer budget must be an explicit extension within 20000")
    require(target["scorer_updates"] == 4*target["pointwise_updates"],
            "preserve the declared 1:3 pointwise/comparison phase ratio")


def training_budget(config):
    initial = original_budget(config)
    path = config.root/FILE
    if not path.exists():
        return {**initial, "amendment_hash": None}
    require(config.research_protocol == "two_candidate_v1", "budget amendment requires two_candidate_v1")
    receipt = read_json(path)
    require(set(receipt) == {"schema", "experiment_id", "base_config_hash", "base_binding_hash",
            "partition_hash", "from", "to", "retained_recipe", "reason", "recorded_utc",
            "optimizer_updates_at_amendment", "source"}, "unexpected amendment fields")
    require(receipt["schema"] == SCHEMA and receipt["experiment_id"] == config.experiment_id,
            "budget amendment experiment mismatch")
    require(receipt["base_config_hash"] == config.identity and receipt["from"] == initial,
            "original immutable config changed")
    require(receipt["base_binding_hash"] == file_hash(config.root/"BASE_BINDING.json") and
            receipt["partition_hash"] == file_hash(config.root/"DATA_PARTITION.json"),
            "budget amendment Base/data binding mismatch")
    require(receipt["retained_recipe"] == retained_recipe(config), "budget-only amendment changed the training recipe")
    require(receipt["optimizer_updates_at_amendment"] == {"expert": 0, "scorer": 0},
            "this amendment requires unused head training lineages")
    _validate_target(receipt["to"], initial)
    return {**receipt["to"], "amendment_hash": file_hash(path)}


def amend_training_budget(config, target, reason):
    require(config.research_protocol == "two_candidate_v1", "amendment only applies to this research baseline")
    require(isinstance(reason, str) and reason.strip(), "explicit budget amendment reason required")
    initial = original_budget(config)
    _validate_target(target, initial)
    with lock(config.root/"training_budget.lock"):
        if (config.root/FILE).exists():
            active = training_budget(config)
            require({k: active[k] for k in FIELDS} == target, "immutable training budget already differs")
            return active
        for kind in ("expert", "scorer"):
            root = config.root/kind
            steps = root/"steps.jsonl"
            require(not (steps.exists() and steps.stat().st_size) and
                    not list(root.glob("step_*.pt*")) and not (root/"result.json").exists() and
                    not (root/"status.json").exists() and not (root/"TRAINING_BUDGET.json").exists(),
                    "cannot amend after a head optimizer has started")
        atomic_json(config.root/FILE, {
            "schema": SCHEMA, "experiment_id": config.experiment_id,
            "base_config_hash": config.identity,
            "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"),
            "partition_hash": file_hash(config.root/"DATA_PARTITION.json"),
            "from": initial, "to": target, "retained_recipe": retained_recipe(config),
            "reason": reason.strip(), "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "optimizer_updates_at_amendment": {"expert": 0, "scorer": 0},
            "source": "explicit_current_user_training_budget_revision",
        }, immutable=True)
        return training_budget(config)


def training_dependencies(config):
    value = training_budget(config)["amendment_hash"]
    return {"training_budget_amendment": value} if value else {}


def bind_training_budget(config, kind):
    require(kind in {"expert", "scorer"}, "unknown head training kind")
    with lock(config.root/"training_budget.lock"):
        budget = training_budget(config)
        atomic_json(config.root/kind/"TRAINING_BUDGET.json", {
            "kind": kind, "budget": budget, "original_config_hash": config.identity,
            "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"),
        }, immutable=True)
    return budget
