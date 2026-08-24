"""Geometry-Preserving Scene-Conditioned 3D-Mix framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.vggt_query.centered_action_geometry_reader import (
    CenteredActionGeometryReader,
)
from starVLA.model.modules.vggt_query.gp_geometry_adapter import GeometryMemoryAdapter
from starVLA.model.modules.vggt_query.gp_geometry_gate import (
    SceneConditionedGeometryGate,
)
from starVLA.model.modules.vggt_query.gp_slot_stats import (
    load_gp_slot_stats,
    sha256_file,
)
from starVLA.model.modules.vggt_query.scene_summary import MaskedSceneSummary
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_geometry_per_view,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


FUSION_MODES = {"disabled", "projected_residual", "gated_residual"}
INTERVENTION_MODES = {"real", "zero", "shuffled", "slot_mean"}


def _required_file_sha(path: str | Path | None, name: str) -> str:
    if not path:
        raise ValueError(f"Non-disabled GP-SQ3D-Mix requires {name}")
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"GP-SQ3D-Mix {name} is missing: {candidate}")
    return sha256_file(candidate)


@FRAMEWORK_REGISTRY.register("QwenOFT_GPSQ3DMix")
class QwenOFT_GPSQ3DMix(Qwenvl_OFT):
    """Inject a bounded, centered geometry residual into action queries."""

    _ACTION_ONLY_MISSING_PREFIXES = (
        "geometry_memory_adapter.",
        "scene_conditioned_geometry_gate.",
        "centered_geometry_reader.",
        "pooled_feature_slot_mean",
    )

    def __init__(self, config=None, accelerator=None, infer_not_load_wan=0, **kwargs):
        super().__init__(
            config=config,
            accelerator=accelerator,
            infer_not_load_wan=infer_not_load_wan,
            **kwargs,
        )
        gp_cfg = OmegaConf.select(config, "framework.gp_sq_3d_mix", default=None)
        if gp_cfg is None:
            raise ValueError("QwenOFT_GPSQ3DMix requires framework.gp_sq_3d_mix")
        self._validate_action_only_contract(config)
        self.gp_mode = str(
            OmegaConf.select(gp_cfg, "mode", default="gated_residual")
        ).strip().lower()
        if self.gp_mode not in FUSION_MODES:
            raise ValueError(f"GP mode must be one of {sorted(FUSION_MODES)}")
        self.gp_intervention = str(
            OmegaConf.select(gp_cfg, "intervention.mode", default="real")
        ).strip().lower()
        if self.gp_intervention not in INTERVENTION_MODES:
            raise ValueError(
                f"GP intervention must be one of {sorted(INTERVENTION_MODES)}"
            )
        self.training_stage = str(
            OmegaConf.select(gp_cfg, "training.stage", default="stage_a")
        ).strip().lower()
        if self.training_stage not in {"stage_a", "stage_b", "inference"}:
            raise ValueError("training.stage must be stage_a, stage_b, or inference")
        self.rank_margin_ratio = float(
            OmegaConf.select(gp_cfg, "training.rank_margin_ratio", default=0.05)
        )
        self.fidelity_tolerance = float(
            OmegaConf.select(gp_cfg, "training.fidelity_tolerance", default=0.02)
        )
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        if hidden_dim != 2048:
            raise ValueError("GP-SQ3D-Mix requires Qwen hidden dimension 2048")
        self.scene_summary = MaskedSceneSummary(2048, 8)
        self.geometry_memory_adapter = GeometryMemoryAdapter(2048, 512, 3)
        self.scene_conditioned_geometry_gate = SceneConditionedGeometryGate(
            scene_dim=2048,
            geometry_dim=512,
            minimum_retention=float(
                OmegaConf.select(gp_cfg, "gate.minimum_retention", default=0.05)
            ),
            maximum_retention=float(
                OmegaConf.select(gp_cfg, "gate.maximum_retention", default=0.50)
            ),
            initial_retention=float(
                OmegaConf.select(gp_cfg, "gate.initial_retention", default=0.10)
            ),
        )
        self.centered_geometry_reader = CenteredActionGeometryReader(
            action_dim=2048,
            geometry_dim=512,
            num_heads=8,
            alpha_min=float(OmegaConf.select(gp_cfg, "reader.alpha_min", default=0.05)),
            alpha_max=float(OmegaConf.select(gp_cfg, "reader.alpha_max", default=0.30)),
            alpha_initial=float(
                OmegaConf.select(gp_cfg, "reader.alpha_initial", default=0.10)
            ),
        )
        self.register_buffer(
            "pooled_feature_slot_mean",
            torch.zeros((180, 2048), dtype=torch.float32),
            persistent=True,
        )
        if self.gp_mode != "disabled":
            stats_root = str(OmegaConf.select(gp_cfg, "stats.root", default="")).strip()
            if not stats_root:
                raise ValueError("Non-disabled GP-SQ3D-Mix requires stats.root")
            source_cache_manifest = str(
                OmegaConf.select(
                    gp_cfg, "stats.source_cache_manifest", default=""
                )
            ).strip()
            datalist = str(
                OmegaConf.select(gp_cfg, "stats.source_datalist", default="")
            ).strip()
            mean, manifest = load_gp_slot_stats(
                stats_root,
                expected_source_cache_manifest_sha256=_required_file_sha(
                    source_cache_manifest, "stats.source_cache_manifest"
                ),
                expected_datalist_sha256=_required_file_sha(
                    datalist, "stats.source_datalist"
                ),
            )
            self.pooled_feature_slot_mean.copy_(mean)
            self._gp_stats_manifest = manifest
        else:
            self._gp_stats_manifest = None
        self._use_named_loss_contract = True
        self._configure_stage_trainability()

    def _validate_action_only_contract(self, config) -> None:
        if self.action_prompt_mode != "minimal" or self.mlp_head != 0:
            raise ValueError("GP-SQ3D-Mix requires minimal prompts and Flow Matching")
        if len(self.act_query_tokens) != 8 or int(self.act_tok) != 8:
            raise ValueError("GP-SQ3D-Mix requires exactly eight action queries")
        if int(OmegaConf.select(config, "datasets.vla_data.load_act_data", default=0)) != 1:
            raise ValueError("GP-SQ3D-Mix requires action data")
        forbidden = (
            "datasets.video_data.load_2d_data",
            "datasets.gs_data.load_3d_data",
            "datasets.reward_data.load_reward_data",
            "w_depth",
        )
        if any(bool(OmegaConf.select(config, path, default=False)) for path in forbidden):
            raise ValueError("GP-SQ3D-Mix is an action-only framework")

    def _configure_stage_trainability(self) -> None:
        # Fail closed: no inherited auxiliary head may accidentally enter the
        # optimizer.  Each stage opts in only the parameters allowed by its
        # contract.
        self.requires_grad_(False)
        if self.training_stage == "stage_a":
            if self.gp_mode == "disabled":
                raise ValueError("Stage A requires an active GP residual mode")
            self.geometry_memory_adapter.requires_grad_(True)
            self.centered_geometry_reader.requires_grad_(True)
            if self.gp_mode == "gated_residual":
                self.scene_conditioned_geometry_gate.requires_grad_(True)
        elif self.training_stage == "stage_b":
            self.action_model.requires_grad_(True)
            # The matched continuation control uses this framework in
            # disabled mode so model construction and the action path stay
            # matched.  Keep its unused GP parameters frozen; otherwise DDP
            # may wait for gradients that the disabled route cannot produce.
            if self.gp_mode != "disabled":
                self.geometry_memory_adapter.requires_grad_(True)
                self.centered_geometry_reader.requires_grad_(True)
                if self.gp_mode == "gated_residual":
                    self.scene_conditioned_geometry_gate.requires_grad_(True)
        self._enforce_frozen_module_modes()

    def _enforce_frozen_module_modes(self) -> None:
        self.qwen_vl_interface.eval()
        self.action_input_model.eval()
        if self.training_stage != "stage_b":
            self.action_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._enforce_frozen_module_modes()
        return self

    @staticmethod
    def _payloads_from_examples(
        examples: Sequence[Mapping[str, object]] | None,
    ) -> list[Mapping[str, Tensor]]:
        if examples is None:
            raise RuntimeError("GP-SQ3D-Mix requires cache-backed examples")
        payloads = [example.get("vggt_dense_feature_cache") for example in examples]
        if not all(payload is not None for payload in payloads):
            missing = [index for index, value in enumerate(payloads) if value is None]
            raise RuntimeError(f"Missing dense VGGT payloads at batch indices {missing}")
        return payloads

    def set_gp_intervention(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in INTERVENTION_MODES:
            raise ValueError(f"Unknown GP intervention: {mode}")
        self.gp_intervention = normalized

    @staticmethod
    def _nearest_target_shuffle(examples, device: torch.device) -> Tensor | None:
        if len(examples) < 2:
            return None
        targets = torch.stack(
            [torch.as_tensor(example["action"], dtype=torch.float32) for example in examples]
        ).to(device=device)
        flat = targets.flatten(1)
        distances = torch.cdist(flat, flat, p=2)
        distances.fill_diagonal_(float("inf"))
        return distances.argmin(dim=1)

    def _apply_intervention(self, pooled, examples, mode: str):
        selected = {key: value for key, value in pooled.items()}
        metrics = {}
        if mode == "real":
            return selected, metrics
        if mode == "zero":
            selected["features"] = torch.zeros_like(pooled["features"])
            return selected, metrics
        if mode == "slot_mean":
            selected["features"] = self.pooled_feature_slot_mean.to(
                device=pooled["features"].device, dtype=pooled["features"].dtype
            ).unsqueeze(0).expand(pooled["features"].shape[0], -1, -1)
            return selected, metrics
        if examples and all(
            example.get("vggt_dense_pre_shuffled", False) for example in examples
        ):
            metrics["gp_sq3dmix/topology_independent_shuffle"] = torch.ones(
                (), device=pooled["features"].device
            )
            return selected, metrics
        indices = self._nearest_target_shuffle(examples, pooled["features"].device)
        if indices is None:
            metrics["gp_sq3dmix/intervention_skipped"] = torch.ones(
                (), device=pooled["features"].device
            )
            return selected, metrics
        return {key: value.index_select(0, indices) for key, value in pooled.items()}, metrics

    def _memory_pair(self, pooled, examples, scene, mode: str):
        selected, diagnostics = self._apply_intervention(pooled, examples, mode)
        reference_features = self.pooled_feature_slot_mean.to(
            device=selected["features"].device,
            dtype=selected["features"].dtype,
        ).unsqueeze(0).expand(selected["features"].shape[0], -1, -1)
        real_memory = self.geometry_memory_adapter(
            selected["features"],
            selected["view_ids"],
            selected["uv_coords"],
            selected["ray_features"],
        )
        reference_memory = self.geometry_memory_adapter(
            reference_features,
            selected["view_ids"],
            selected["uv_coords"],
            selected["ray_features"],
        )
        if self.gp_mode == "gated_residual":
            real_memory, real_gate = self.scene_conditioned_geometry_gate(
                scene, real_memory
            )
            reference_memory, reference_gate = self.scene_conditioned_geometry_gate(
                scene, reference_memory
            )
            diagnostics.update(real_gate)
            diagnostics.update(
                {
                    key.replace("gp_sq3dmix/", "gp_sq3dmix/reference_"): value
                    for key, value in reference_gate.items()
                }
            )
        return real_memory, reference_memory, diagnostics

    def _enhance(self, action_queries, extension, mode: str):
        real_memory, reference_memory, diagnostics = self._memory_pair(
            extension["pooled_geometry"],
            extension["examples"],
            extension["scene_summary"],
            mode,
        )
        enhanced, reader_metrics = self.centered_geometry_reader(
            action_queries, real_memory, reference_memory
        )
        diagnostics.update(reader_metrics)
        return enhanced, diagnostics

    def _compute_query_extension(
        self,
        last_hidden,
        token_positions,
        examples,
        *,
        input_ids=None,
        attention_mask=None,
        image_grid_thw=None,
    ):
        del input_ids, image_grid_thw
        if self.gp_mode == "disabled":
            return {"action_context": None, "losses": {}, "metrics": {}}
        scene, diagnostics = self.scene_summary(
            last_hidden, attention_mask, token_positions["action"]
        )
        pooled = pool_dense_vggt_geometry_per_view(
            self._payloads_from_examples(examples),
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )
        return {
            "action_context": None,
            "losses": {},
            "metrics": diagnostics,
            "scene_summary": scene,
            "pooled_geometry": pooled,
            "examples": examples,
        }

    def _condition_action_queries(self, action_queries, extension):
        if self.gp_mode == "disabled":
            return action_queries, None, {}
        extension["baseline_action_queries"] = action_queries
        real_queries, real_metrics = self._enhance(action_queries, extension, "real")
        shuffled_queries, shuffled_metrics = self._enhance(
            action_queries, extension, "shuffled"
        )
        extension["real_action_queries"] = real_queries
        extension["shuffled_action_queries"] = shuffled_queries
        slot_mean_queries, _ = self._enhance(action_queries, extension, "slot_mean")
        extension["metrics"]["gp_sq3dmix/slot_mean_identity_max_abs"] = (
            slot_mean_queries.detach() - action_queries.detach()
        ).abs().max()
        selected = real_queries
        metrics = real_metrics
        if self.gp_intervention not in {"real", "shuffled"}:
            selected, metrics = self._enhance(
                action_queries, extension, self.gp_intervention
            )
        elif self.gp_intervention == "shuffled":
            selected, metrics = shuffled_queries, shuffled_metrics
        extension["metrics"].update(metrics)
        return selected, None, metrics

    def _condition_inference_action_queries(
        self,
        last_hidden,
        input_ids,
        action_queries,
        *,
        attention_mask=None,
        image_grid_thw=None,
        examples=None,
    ):
        del image_grid_thw
        if self.gp_mode == "disabled":
            return action_queries, None, {}
        positions = self._find_token_positions(
            input_ids, self._special_token_ids["action"]
        )
        extension = self._compute_query_extension(
            last_hidden,
            {"action": positions},
            examples,
            attention_mask=attention_mask,
        )
        enhanced, diagnostics = self._enhance(
            action_queries, extension, self.gp_intervention
        )
        return enhanced, None, diagnostics

    def _compute_action_loss(
        self,
        action_queries,
        actions,
        video_token,
        action_context,
        extension,
    ):
        if self.gp_mode == "disabled" or self.training_stage == "inference":
            return super()._compute_action_loss(
                action_queries, actions, video_token, action_context, extension
            )
        if video_token is not None or action_context is not None:
            raise RuntimeError("GP-SQ3D-Mix must call Action DiT with extra_context=None")
        repeats = int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 1)
        )
        repeat_actions = actions.repeat(repeats, 1, 1)
        baseline = extension["baseline_action_queries"].repeat(repeats, 1, 1)
        real = extension["real_action_queries"].repeat(repeats, 1, 1)
        shuffled = extension["shuffled_action_queries"].repeat(repeats, 1, 1)
        flow_state = self.action_model.sample_flow_state(repeat_actions)
        rng_devices = (
            [repeat_actions.device.index]
            if repeat_actions.is_cuda and repeat_actions.device.index is not None
            else []
        )
        cpu_rng_state = torch.random.get_rng_state()
        device_rng_state = (
            torch.cuda.get_rng_state(rng_devices[0]) if rng_devices else None
        )

        # Let the baseline call advance the global stream exactly once, as the
        # matched action-only control does.  Replays start from its entry state
        # inside fork_rng and restore the post-baseline stream on exit.  Thus
        # all three conditions share a dropout mask without shifting the next
        # training step's FlowMatchingState relative to the control run.
        base_loss = self.action_model.loss_from_flow_state(
            baseline,
            repeat_actions,
            flow_state,
            extra_context=None,
            reduction="none",
        )

        def replay_matched_loss(queries):
            with torch.random.fork_rng(devices=rng_devices):
                torch.random.set_rng_state(cpu_rng_state)
                if device_rng_state is not None:
                    torch.cuda.set_rng_state(device_rng_state, rng_devices[0])
                return self.action_model.loss_from_flow_state(
                    queries,
                    repeat_actions,
                    flow_state,
                    extra_context=None,
                    reduction="none",
                )

        real_loss = replay_matched_loss(real)
        shuffled_loss = replay_matched_loss(shuffled)
        margin = self.rank_margin_ratio * base_loss.detach().clamp_min(1e-4)
        rank_loss = F.relu(margin + real_loss - shuffled_loss).mean()
        action_loss = real_loss.mean()
        extension["losses"]["geometry_rank"] = rank_loss
        if self.training_stage == "stage_a":
            fidelity_loss = F.relu(
                real_loss - (1.0 + self.fidelity_tolerance) * base_loss.detach()
            ).mean()
            extension["losses"]["baseline_fidelity"] = fidelity_loss
        extension["metrics"].update(
            {
                "gp_sq3dmix/base_flow_loss": base_loss.detach().mean(),
                "gp_sq3dmix/real_flow_loss": real_loss.detach().mean(),
                "gp_sq3dmix/shuffled_flow_loss": shuffled_loss.detach().mean(),
                "gp_sq3dmix/relative_shuffled_real_gap": (
                    (shuffled_loss.detach() - real_loss.detach()).mean()
                    / real_loss.detach().mean().clamp_min(1e-8)
                ),
            }
        )
        self._last_gp_gate_samples = {
            "base_loss": base_loss.detach(),
            "real_loss": real_loss.detach(),
            "shuffled_loss": shuffled_loss.detach(),
        }
        return action_loss

    def get_planning_usage_metrics(self) -> dict[str, Tensor]:
        parameters = {
            "gp_sq3dmix/adapter_grad_norm": self.geometry_memory_adapter.feature_projection.weight,
            "gp_sq3dmix/gate_grad_norm": self.scene_conditioned_geometry_gate.gate_projection.weight,
            "gp_sq3dmix/reader_grad_norm": self.centered_geometry_reader.up_projection.weight,
        }
        return {
            name: parameter.grad.detach().float().norm()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }

    def load_gp_modules_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load only Stage-A GP parameters into a fresh action-only base."""

        modules = {
            "geometry_memory_adapter": self.geometry_memory_adapter,
            "scene_conditioned_geometry_gate": self.scene_conditioned_geometry_gate,
            "centered_geometry_reader": self.centered_geometry_reader,
        }
        for prefix, module in modules.items():
            selected = {
                key[len(prefix) + 1 :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix + ".")
            }
            if not selected:
                raise RuntimeError(f"Stage-A checkpoint has no {prefix} parameters")
            module.load_state_dict(selected, strict=True)

    def load_action_only_state_dict(
        self, state_dict: Mapping[str, Tensor], *, assign: bool = False
    ):
        """Strictly warm-start from an action-only checkpoint.

        The only permitted missing entries are the newly introduced GP
        modules/buffer, and a checkpoint that already contains GP parameters
        is rejected so Stage A cannot silently warm-start from the wrong run.
        """

        if any(
            key.startswith(self._ACTION_ONLY_MISSING_PREFIXES)
            for key in state_dict
        ):
            raise RuntimeError("Warm-start checkpoint is not action-only")
        incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self._ACTION_ONLY_MISSING_PREFIXES)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if invalid_missing or unexpected:
            raise RuntimeError(
                "QwenOFT_GPSQ3DMix checkpoint mismatch: "
                f"missing={invalid_missing[:20]} unexpected={unexpected[:20]}"
            )
        return incompatible

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Full GP checkpoints use ordinary PyTorch strictness.  Action-only
        # warm-start is intentionally a separate, explicit API above.
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


# Keep the repository's historical Qwenvl_* naming available to callers.
Qwenvl_OFT_GPSQ3DMix = QwenOFT_GPSQ3DMix
