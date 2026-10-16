"""Independent negative tests for deployment identity and native state input."""
import subprocess

import pytest

from tools.st_world_policy.checkpoints import inference_source_contract
from tools.st_world_policy.training_state import checkpoint_receipt, verify_checkpoint


def test_native_reader_rejects_unreceipted_model_shard(tmp_path):
    """zero_to_fp32 globs model shards, so receipt-only inventory is insufficient."""
    folder = tmp_path / "step_000004"
    folder.mkdir()
    for name in ("mp_rank_00_model_states.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
                 "rng_rank0.pt"):
        (folder / name).write_bytes(name.encode())
    state = {"identity": "review", "completed": 4, "epoch": 0, "offset": 4,
             "tag": folder.name, "termination_reason": "RUNNING_CHECKPOINT"}
    checkpoint_receipt(folder, state, world_size=1)
    verify_checkpoint(folder, identity="review", world_size=1)
    # This can be a stale copied model shard. It sorts before mp_rank_00 and is
    # consumed by DeepSpeed despite never being included in COMPLETE.json.
    (folder / "a_model_states.pt").write_bytes(b"unregistered frozen fragments")
    with pytest.raises(ValueError, match="[Ii]nventory|[Ff]oreign|[Uu]nexpected|[Ss]hard"):
        verify_checkpoint(folder, identity="review", world_size=1)


def test_deployment_source_contract_rejects_changed_module_global(tmp_path):
    """Identical method AST is not sufficient when their module globals change."""
    files = ["starVLA/model/framework/st_world_model.py", "tools/st_world_policy/world_expansion.py",
             "starVLA/model/framework/ddp_action_video_foresight.py", "starVLA/model/modules/st_world/config.py",
             "starVLA/dataloader/st_world_schema.py"]
    model = "INPUT_SCALE = 1\nclass SpaceTimeStudent:\n" + "\n".join(
        "    def " + name + "(self):\n        return INPUT_SCALE\n" for name in
        ("__init__", "train", "encode_inputs", "predict_raw", "predict_action", "strip_auxiliary_heads", "forward"))
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model if name.endswith("st_world_model.py") else "CONTRACT = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "independent deployment global fixture"], cwd=tmp_path, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    inference_source_contract(tmp_path, sha, history=False)
    (tmp_path / files[0]).write_text(model.replace("INPUT_SCALE = 1", "INPUT_SCALE = 2"))
    with pytest.raises(ValueError, match="compatibility review"):
        inference_source_contract(tmp_path, sha, history=False)
