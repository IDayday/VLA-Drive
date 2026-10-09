"""Original-trajectory S0 framework with a post-VLM singleton Query action head.

The user's correction authorizes architecture migration, not learned S0 weights.
Generic Qwen initialization and the original auxiliary objectives are retained.
The original DiT/flow-matching action objective is explicitly replaced by Query IL.
No additional VLM token is inserted for any incremental expert.
"""
from __future__ import annotations
import importlib
from pathlib import Path
import sys
import threading
import torch
from torch import nn
from .contracts import FeatureBundle, TrajectoryContract, require
from .expert import IndependentExpert
from .losses import expert_il_terms, reduce_terms
from .io import file_hash

_CONSTRUCTION_LOCK = threading.Lock()


def bind_source(root, expected_commit):
    import subprocess
    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(actual == expected_commit, "original S0 framework source commit mismatch")
    require(not subprocess.check_output(["git", "diff", "--name-only"], cwd=root).strip(), "original S0 framework source dirty")
    loaded = sys.modules.get("starVLA")
    if loaded is not None:
        require(Path(loaded.__file__).resolve().is_relative_to(root), "another starVLA source already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def source_trajectory_contract(horizon=8, dt=0.5):
    return TrajectoryContract(horizon, dt, "normalized_xy_sincos", "ego_relative", "rear_axle", "radian",
                              (10.172484, .360762, 0., 0.), (8.805105, 2.277741, 1., 1.),
                              "original version1225 training constants; no dev/test fitted statistics")


class QueryActionBridge(nn.Module):
    def __init__(self, source_config, options):
        super().__init__()
        from omegaconf import OmegaConf
        from navsim.agents.EpisodeDrive.layers.q_former.q_former import VisionOnlyQFormer
        from navsim.agents.EpisodeDrive.transformer_decoder import TransformerDecoder
        from navsim.agents.EpisodeDrive.layers.utils.mlp import MLP
        width, ffn, layers = options["width"], options["ffn"], options["layers"]
        self.trajectory_contract = source_trajectory_contract(int(source_config.framework.action_model.action_horizon), options["dt"])
        source_dim = int(source_config.framework.qwenvl.vl_hidden_dim)
        self.q_former = VisionOnlyQFormer(source_dim, width, options["scene_tokens"], options["compressor_layers"], options["compressor_heads"])
        self.scene_queries = nn.Parameter(torch.randn(1, options["scene_tokens"], width) * 1e-6)
        self.ego_encoder = nn.Linear(4, width)
        config = OmegaConf.create({"ref_num": layers, "tf_d_model": width, "refiner_num_heads": options["heads"], "refiner_ls_values": 0.0})
        query = nn.Embedding(1, width)
        decoder = TransformerDecoder(options["dropout"], options["drop_path"], config)
        heads = nn.ModuleList([MLP(width, ffn, self.trajectory_contract.horizon * self.trajectory_contract.raw_dim) for _ in range(layers + 1)])
        self.expert = IndependentExpert(query, decoder, heads, self.trajectory_contract)
        # Original S0 constructor toggles this marker for its action model. It does not run a DiT.
        self.model = nn.Identity()
        self.ego_state = None
        self.sample_valid = None
        self.prev_weight = options["prev_weight"]

    def encode_features(self, action_memory, ego_state, scene_ids=None, contract_hash="base_training"):
        scene = self.q_former(self.scene_queries, action_memory.float())
        ego = self.ego_encoder(ego_state.float()).reshape(len(scene), 1, -1)
        return FeatureBundle(scene, ego, torch.ones(scene.shape[:2], device=scene.device, dtype=torch.bool),
                             contract_hash, tuple(scene_ids or map(str, range(len(scene)))))

    def forward(self, action_memory, target, **unused_flow_arguments):
        require(self.ego_state is not None, "Query head requires actual current ego state")
        repeat = len(action_memory) // len(self.ego_state)
        require(repeat * len(self.ego_state) == len(action_memory), "source repeated-target batch mismatch")
        f = self.encode_features(action_memory, self.ego_state.repeat(repeat, 1, 1))
        prediction = self.expert(f)
        # Original framework applies optimizer-batch global_counts outside this head.
        # Returning the local mean preserves that original DDP/accumulation reduction.
        masks = None if self.sample_valid is None else self.sample_valid.repeat(repeat)[:, None].expand(-1, target.shape[1])
        term = expert_il_terms(prediction, target, prev_weight=self.prev_weight, masks=masks)["il"]
        # The source scales this local padded-batch mean by local batch size and
        # the actual global valid count. Padding contributes a graph-connected zero.
        return term.numerator / max(1, len(target))

    def predict_action(self, action_memory, **unused_noise):
        return self.expert(self.encode_features(action_memory, self.ego_state)).raw


def build_query_framework(config, options, source_root, source_commit):
    """Instantiate the real framework, replacing DiT construction without allocating its weights."""
    bind_source(source_root, source_commit)
    from starVLA.model.framework.ddp_action_video_foresight import DDPActionVideoForesight
    module = importlib.import_module("starVLA.model.framework.QwenOFT")

    class QueryFramework(DDPActionVideoForesight):
        def encode_current(self, observations):
            encoded = super().encode_current(observations)
            import numpy as np
            self.action_model.ego_state = torch.as_tensor(np.asarray([e["state"] for e in observations]),
                                                        device=encoded["action_queries"].device, dtype=torch.float32)
            return encoded

        def forward(self, *args, **kwargs):
            target = args[1] if len(args) > 1 else kwargs.get("targets", kwargs.get("training_targets"))
            self.action_model.sample_valid = target.get("iqe_sample_valid")
            out = super().forward(*args, **kwargs)
            out["losses"]["ego_il"] = out["losses"].pop("ego_fm")
            return out

        def predict_action(self, observations, *, sampling_seed=42, initial_noise=None):
            # Query inference is deterministic. Keep the newly defined FP32 action path
            # identical to S0Adapter; the shared Qwen encoder retains its own autocast.
            encoded = self.encode_current(observations)
            action = self.build_planner_condition(encoded)
            with torch.autocast(action.device.type, enabled=False):
                features = self.action_model.encode_features(action, self.action_model.ego_state)
                return self.action_model.expert(features).physical

    # Scoped factory replacement only changes the head requested by the original constructor.
    # Source files, generic source validation, tokenizer, prompts and auxiliary paths are untouched.
    with _CONSTRUCTION_LOCK:
        original = module.get_action_model
        module.get_action_model = lambda config: QueryActionBridge(config, options)
        try:
            model = QueryFramework(config)
        finally:
            module.get_action_model = original
    require(isinstance(model.action_model, QueryActionBridge), "head migration did not bind")
    return model
