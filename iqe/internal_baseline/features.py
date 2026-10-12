"""Read-only W/A extraction, sharing exactly one original encode_current call."""
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
import time
import torch
from iqe.contracts import FeatureBundle, require
from iqe.io import atomic_json, atomic_torch, read_json, digest, file_hash, complete
from iqe.data.manifests import load_scenes
from iqe.data.sources import load_observation, load_target


@dataclass
class ScoreContextBundle:
    world: torch.Tensor
    action: torch.Tensor
    current_ego: torch.Tensor
    world_valid: torch.Tensor
    action_valid: torch.Tensor
    scene_ids: tuple[str, ...]
    identity: str

    def __post_init__(self):
        require(self.world.ndim == self.action.ndim == 3 and len(self.scene_ids) == len(self.action), "score context dimensions")
        require(self.current_ego.shape == (len(self.action), 4), "legal original current ego [B,4] required")
        for x, mask in ((self.world, self.world_valid), (self.action, self.action_valid)):
            require(mask.dtype == torch.bool and mask.shape == x.shape[:2], "score context mask")
            require(not x.requires_grad and torch.isfinite(x).all().item(), "frozen finite score context required")

    def tensors(self):
        return {k: getattr(self, k) for k in ("world", "action", "current_ego", "world_valid", "action_valid")}


@torch.no_grad()
def encode_joint(adapter, observations):
    # Never pass training labels, targets or cached hidden tensors to the framework.
    allowed = {"image", "lang", "state", "token"}
    current = [{k: row[k] for k in allowed} for row in observations]
    encoded = adapter.framework.encode_current(current)
    require(set(("W", "action_queries")) <= set(encoded), "current framework lacks the required W/A interface")
    planner = adapter.framework.action_model.encode_features(
        adapter.framework.build_planner_condition(encoded), adapter.framework.action_model.ego_state,
        [r["token"] for r in current], adapter.contract_hash).detached()
    w, a = encoded["W"].detach(), encoded["action_queries"].detach()
    context = ScoreContextBundle(w, a, adapter.framework.action_model.ego_state.detach().reshape(len(a), 4),
                                torch.ones(w.shape[:2], device=w.device, dtype=torch.bool),
                                torch.ones(a.shape[:2], device=a.device, dtype=torch.bool),
                                planner.scene_ids, adapter.contract_hash)
    return planner, context


def cache_key(config, scene):
    b = read_json(config.root / "BASE_BINDING.json")
    return digest({"schema": "joint_planner_W_A_current_ego_v1", "base": b["checkpoint_hash"],
                   "runtime": b["inference_runtime"], "contract": b["contract_hash"],
                   "observation": scene.observation_hash, "scene": scene.scene_id,
                   "feature_code": file_hash(__file__), "masks": "boolean_true_valid"})


class JointCache:
    def __init__(self, config, capacity=128):
        self.config, self.capacity = config, capacity
        self.loaded = OrderedDict()

    def path(self, scene):
        return self.config.root / "features" / (scene.scene_id + ".pt")

    def valid(self, scene):
        path = self.path(scene)
        receipt = Path(str(path) + ".COMPLETE.json")
        if not receipt.exists():
            return False
        r = read_json(receipt)
        require(r["key"] == cache_key(self.config, scene), "stale joint feature cache")
        require(file_hash(path) == r["checksum"], "corrupt joint feature cache")
        return True

    def write(self, scene, values):
        path = self.path(scene)
        atomic_torch(path, {k: v.detach().cpu().contiguous().clone() for k, v in values.items()}, immutable=True)
        atomic_json(str(path) + ".COMPLETE.json", {"key": cache_key(self.config, scene), "checksum": file_hash(path),
                    "scene_id": scene.scene_id, "shapes": {k: list(v.shape) for k, v in values.items()},
                    "dtypes": {k: str(v.dtype) for k, v in values.items()}}, immutable=True)

    def read(self, scene):
        if scene.scene_id in self.loaded:
            self.loaded.move_to_end(scene.scene_id)
            return self.loaded[scene.scene_id]
        require(self.valid(scene), "missing frozen joint features: " + scene.scene_id)
        value = torch.load(self.path(scene), weights_only=True, map_location="cpu")
        self.loaded[scene.scene_id] = value
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return value

    def batch(self, scenes, device):
        rows = [self.read(s) for s in scenes]
        # Tokens are explicit and constant for this Base (W and action slots),
        # not the variable-length language sequence. Unknown shapes are rejected.
        return {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


def planner_bundle(config, tensors, ids):
    return FeatureBundle(tensors["planner_scene"], tensors["planner_ego"], tensors["planner_mask"],
                         read_json(config.root / "BASE_BINDING.json")["contract_hash"], tuple(ids))


def build_cache(config, shard=0, shards=1, device="cuda:0", max_samples=None):
    from .binding import binding
    from iqe.s0_adapter import S0Adapter
    require(0 <= shard < shards <= 8, "cache worker/shard budget")
    binding(config)
    scenes = load_scenes(config.root / "scenes.json")[shard::shards]
    if max_samples is not None:
        require(max_samples > 0, "sample budget")
        scenes = scenes[:max_samples]
    cache = JointCache(config)
    pending = [s for s in scenes if not cache.valid(s)]
    if not pending:
        return {"status": "COMPLETE", "scenes": len(scenes), "reused": True}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = S0Adapter.load_and_validate_s0(read_json(config.root / "OFFLINE_BASE_CONTRACT.json"), device)
    require(not any(p.requires_grad for p in adapter.parameters()), "shared model must be fully frozen")
    trajectory = adapter.framework.action_model.trajectory_contract
    started = time.perf_counter()
    count = 0
    for at in range(0, len(pending), config.cache_batch):
        subset = pending[at:at + config.cache_batch]
        observations = [load_observation(s, adapter.contract) for s in subset]
        planner, context = encode_joint(adapter, observations)
        require(context.world.shape[1] > 0 or config.score_context == "action_ego", "W absent: explicit action_ego ablation required")
        with torch.no_grad():
            pred = adapter.predict_s0(planner)
        targets = torch.stack([load_target(s, trajectory) for s in subset])
        values = context.tensors() | {"planner_scene": planner.scene, "planner_ego": planner.ego,
                  "planner_mask": planner.valid_tokens, "base_raw": pred.raw, "base_physical": pred.physical,
                  "target_raw": targets, "target_physical": trajectory.physical(targets)}
        # Targets are stored in a separate namespace; ScoreContextBundle and
        # scorer.forward accept only the explicit deployment tensor whitelist.
        for i, s in enumerate(subset):
            cache.write(s, {k: v[i] for k, v in values.items()})
        count += len(subset)
        atomic_json(config.root / f"cache_worker_{shard}.json", {"status": "RUNNING", "completed": count,
                    "pending_at_start": len(pending), "assigned": len(scenes), "elapsed": time.perf_counter()-started,
                    "W_shape": list(context.world.shape), "A_shape": list(context.action.shape),
                    "planner_shape": list(planner.scene.shape), "cuda_peak": torch.cuda.max_memory_allocated()})
    result = {"status": "COMPLETE", "scenes": len(scenes), "new": count, "seconds": time.perf_counter()-started}
    atomic_json(config.root / f"cache_worker_{shard}.json", result)
    return result
