import copy
import json

import pytest

from tools.st_world_policy.base_binding import immutable_json, validate_parent


def parent():
    identity = {"scope": "formal", "config": {"framework": {
        "name": "DDPActionVideoForesight", "action_model": {
            "action_dim": 4, "action_horizon": 8, "num_inference_timesteps": 10,
            "repeated_diffusion_steps": 8}}, "foresight": {
                "planner_condition_mode": "action_only", "num_queries": 144}}}
    checkpoint = {"completed": 100000,
                  "model_class": "starVLA.model.framework.ddp_action_video_foresight.DDPActionVideoForesight",
                  "files": {"mp_rank_00_model_states.pt": "unit fixture", "rng_rank0.pt": "unit fixture"}}
    return identity, checkpoint


def test_diagnostic_query_and_changed_sampling_are_rejected():
    identity, checkpoint = parent()
    validate_parent(identity, checkpoint)
    for field, value in (("name", "QueryBase"), ("num_inference_timesteps", 5),
                         ("repeated_diffusion_steps", 1)):
        mutated = copy.deepcopy(identity)
        target = mutated["config"]["framework"]
        if field != "name":
            target = target["action_model"]
        target[field] = value
        with pytest.raises(ValueError):
            validate_parent(mutated, checkpoint)
    identity["scope"] = "profile"
    with pytest.raises(ValueError, match="Diagnostic"):
        validate_parent(identity, checkpoint)


def test_immutable_binding_is_idempotent_and_never_reselects(tmp_path):
    path = tmp_path / "BASE_BINDING.json"
    value = {"checkpoint": "a", "experimental_id": "fixed"}
    immutable_json(path, value)
    inode = path.stat().st_ino
    immutable_json(path, value)
    assert path.stat().st_ino == inode
    with pytest.raises(ValueError, match="Immutable"):
        immutable_json(path, {**value, "checkpoint": "b"})
    assert json.loads(path.read_text()) == value
    assert not list(tmp_path.glob("*.tmp.*"))
