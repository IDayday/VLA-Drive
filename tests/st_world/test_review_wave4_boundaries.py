"""Independent rejection tests; no reviewed primary implementation is changed."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest
import torch
from torch import nn

from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.st_world_policy import checkpoints as checkpoint_reader
from tools.st_world_policy.training_state import checkpoint_receipt, verify_checkpoint


def test_history_deployment_protects_called_pixel_and_causal_reader(tmp_path):
    """The adapter AST is unchanged when its actual image loader changes."""
    repo = Path(__file__).resolve().parents[2]
    files = ["starVLA/model/framework/st_world_model.py", "tools/st_world_policy/world_expansion.py",
             "starVLA/model/framework/ddp_action_video_foresight.py", "starVLA/model/modules/st_world/config.py",
             "starVLA/dataloader/st_world_schema.py", "starVLA/model/modules/st_world/history_adapter.py",
             "starVLA/dataloader/st_world_history.py"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / name).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=IndependentReview", "-c", "user.email=review@example.invalid",
                    "commit", "-qm", "actual history source dependency fixture"], cwd=tmp_path, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    checkpoint_reader.inference_source_contract(tmp_path, sha, history=True)
    helper = tmp_path / "starVLA/dataloader/st_world_history.py"
    original = helper.read_text()
    assert "Image.Resampling.LANCZOS" in original
    helper.write_text(original.replace("Image.Resampling.LANCZOS", "Image.Resampling.BILINEAR"))
    with pytest.raises(ValueError, match="compatibility review"):
        checkpoint_reader.inference_source_contract(tmp_path, sha, history=True)


def test_st_native_rechecks_after_model_construction(tmp_path, monkeypatch):
    """Real installed DS must not consume a new stale shard after first check."""
    folder = tmp_path / "run/checkpoints/step_000004"
    folder.mkdir(parents=True)
    state = {"identity": "review-wave4", "completed": 4, "epoch": 0, "offset": 4,
             "tag": folder.name, "termination_reason": "RUNNING_CHECKPOINT"}
    native = {"module": {"frozen_visual": torch.tensor([2.])}, "buffer_names": [],
              "param_shapes": [OrderedDict([("trainable", torch.Size([2]))])], "shared_params": {},
              "frozen_param_shapes": {"frozen_visual": torch.Size([1])},
              "frozen_param_fragments": {"frozen_visual": torch.tensor([2.])}, "ds_version": "0.16.9"}
    torch.save(native, folder / "mp_rank_00_model_states.pt")
    torch.save({"optimizer_state_dict": {"zero_stage": 2, "partition_count": 1,
               "single_partition_of_fp32_groups": [torch.tensor([1., 3.])], "optimizer_state_dict": {}}},
               folder / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save({}, folder / "rng_rank0.pt")
    receipt = checkpoint_receipt(folder, state, world_size=1)
    record = {"run": str(folder.parent.parent), "tag": folder.name, "run_identity": state["identity"],
              "receipt_identity": receipt["receipt_identity"],
              "native_receipt_sha256": file_sha256(folder / "COMPLETE.json"),
              "native_checkpoint_sha256": identity_hash({k:v["sha256"] for k,v in receipt["files"].items()})}
    cfg = {"execution": {"seed": 42, "world_size": 1}, "space_time": {}}
    binding = {"parent_identity": {"config": {"framework": {"qwenvl": {}}}}}
    def checked_identity(*args):
        verify_checkpoint(folder, identity=state["identity"], world_size=1)
        return cfg, binding, record
    monkeypatch.setattr(checkpoint_reader, "checkpoint_identity", checked_identity)
    import starVLA.model.framework
    import starVLA.model.framework.st_world_model as wrapper
    from starVLA.model.modules.st_world.config import SpaceTimeConfig
    monkeypatch.setattr(SpaceTimeConfig, "from_dict", lambda *_: SimpleNamespace(history=False))
    def build_parent(*args):
        # A stale/copying process adds a shard during the expensive parent build.
        stale = deepcopy(native)
        stale["frozen_param_fragments"]["frozen_visual"].fill_(9.)
        torch.save(stale, folder / "a_model_states.pt")
        return object()
    monkeypatch.setattr(starVLA.model.framework, "build_framework", build_parent)
    class StrictFixture(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.trainable = nn.Parameter(torch.zeros(2))
            self.frozen_visual = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.base = SimpleNamespace(inference_fp32=False)
        def strip_auxiliary_heads(self):
            return self
    monkeypatch.setattr(wrapper, "SpaceTimeStudent", StrictFixture)
    with pytest.raises(ValueError, match="inventory|changed during|changed after"):
        checkpoint_reader.load_world_checkpoint("fixture_registration.json", folder.name, device="cpu")
