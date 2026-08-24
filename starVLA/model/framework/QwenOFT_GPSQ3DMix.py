"""Geometry-Preserving Scene-Conditioned 3D-Mix framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from starVLA.gp_sq3dmix_v2 import spatial_shuffle_pooled_geometry
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
INTERVENTION_MODES = {
    "real",
    "zero",
    "slot_mean",
    "hard_shuffled",
    "spatial_shuffled",
}


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
        self.spatial_margin_ratio = float(
            OmegaConf.select(gp_cfg, "training.spatial_margin_ratio", default=0.02)
        )
        self.fidelity_tolerance = float(
            OmegaConf.select(gp_cfg, "training.fidelity_tolerance", default=0.02)
        )
        self.spatial_shuffle_seed = int(
            OmegaConf.select(gp_cfg, "intervention.seed", default=20260824)
        )
        self.scene_shuffle_diagnostic = bool(
            OmegaConf.select(
                gp_cfg, "evaluation.scene_shuffle_diagnostic", default=False
            )
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
        self._install_gradient_probes()

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

    def _install_gradient_probes(self) -> None:
        """Capture route gradients before ZeRO can release ``Parameter.grad``.

        DeepSpeed ZeRO-2 may partition or clear a parameter's public ``grad``
        immediately after its autograd hook has run. Reading ``parameter.grad``
        after ``accelerator.backward`` can consequently miss active routes.
        Hooks retain detached scalar norms only and never modify the gradient.
        """

        self._gp_gradient_norms: dict[str, Tensor] = {}
        self._gp_gradient_hook_handles = []
        parameters = {
            "gp_sq3dmix/adapter_grad_norm": (
                self.geometry_memory_adapter.feature_projection.weight
            ),
            "gp_sq3dmix/gate_grad_norm": (
                self.scene_conditioned_geometry_gate.gate_projection.weight
            ),
            "gp_sq3dmix/reader_grad_norm": (
                self.centered_geometry_reader.up_projection.weight
            ),
        }
        for name, parameter in parameters.items():
            if not parameter.requires_grad:
                continue

            def capture(gradient: Tensor, metric_name: str = name) -> Tensor:
                self._gp_gradient_norms[metric_name] = (
                    gradient.detach().float().norm()
                )
                return gradient

            self._gp_gradient_hook_handles.append(parameter.register_hook(capture))

    @staticmethod
    def _payloads_from_examples(
        examples: Sequence[Mapping[str, object]] | None,
        key: str = "vggt_dense_feature_cache",
    ) -> list[Mapping[str, Tensor]]:
        if examples is None:
            raise RuntimeError("GP-SQ3D-Mix requires cache-backed examples")
        payloads = [example.get(key) for example in examples]
        if not all(payload is not None for payload in payloads):
            missing = [index for index, value in enumerate(payloads) if value is None]
            raise RuntimeError(
                f"Missing dense VGGT payload {key!r} at batch indices {missing}"
            )
        return payloads

    def set_gp_intervention(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in INTERVENTION_MODES:
            raise ValueError(f"Unknown GP intervention: {mode}")
        self.gp_intervention = normalized

    def _apply_intervention(self, extension, mode: str):
        pooled = extension["pooled_geometry"]
        selected = {key: value for key, value in pooled.items()}
        device = pooled["features"].device
        diagnostics: dict[str, Tensor] = {}
        if mode == "real":
            return selected, diagnostics
        if mode == "zero":
            selected["features"] = torch.zeros_like(pooled["features"])
            return selected, diagnostics
        if mode == "slot_mean":
            selected["features"] = self.pooled_feature_slot_mean.to(
                device=device, dtype=pooled["features"].dtype
            ).unsqueeze(0).expand(pooled["features"].shape[0], -1, -1)
            return selected, diagnostics
        if mode == "hard_shuffled":
            donor = extension.get("pooled_hard_geometry")
            if donor is None:
                raise RuntimeError(
                    "hard_shuffled requires vggt_dense_hard_negative_cache"
                )
            metadata = [
                example.get("gp_hard_negative_metadata")
                for example in extension["examples"]
            ]
            if not all(isinstance(value, Mapping) for value in metadata):
                raise RuntimeError("hard_shuffled requires fixed donor metadata")
            for name, key in (
                ("hard_donor_action_distance", "action_distance"),
                ("hard_donor_geometry_distance", "geometry_cosine_distance"),
                ("hard_donor_fallback_level", "fallback_level"),
            ):
                diagnostics[f"gp_sq3dmix/{name}"] = torch.tensor(
                    [float(value[key]) for value in metadata],
                    device=device,
                    dtype=torch.float32,
                ).mean()
            return {key: value for key, value in donor.items()}, diagnostics
        if mode == "spatial_shuffled":
            selected, fixed_points = spatial_shuffle_pooled_geometry(
                pooled, extension["tokens"], self.spatial_shuffle_seed
            )
            diagnostics[
                "gp_sq3dmix/spatial_permutation_fixed_point_count"
            ] = fixed_points.float()
            return selected, diagnostics
        raise ValueError(f"Unknown GP intervention mode: {mode}")

    @staticmethod
    def _retention_sample_statistics(retention: Tensor, lower: float, upper: float):
        span = upper - lower
        tolerance = 0.01 * span
        flat = retention.float().flatten(1)
        return {
            "mean": flat.mean(dim=1),
            "std": flat.std(dim=1, unbiased=False),
            "min": flat.min(dim=1).values,
            "max": flat.max(dim=1).values,
            "near_lower_fraction": (flat <= lower + tolerance).float().mean(dim=1),
            "near_upper_fraction": (flat >= upper - tolerance).float().mean(dim=1),
            "_tensor": retention,
        }

    def _memory_pair(self, extension, mode: str, scene_override: Tensor | None = None):
        selected, diagnostics = self._apply_intervention(extension, mode)
        scene = (
            extension["scene_summary"] if scene_override is None else scene_override
        )
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
            real_retention = real_gate.pop("_retention")
            reference_gate.pop("_retention")
            diagnostics.update(real_gate)
            diagnostics.update(
                {
                    key.replace("gp_sq3dmix/", "gp_sq3dmix/reference_"): value
                    for key, value in reference_gate.items()
                }
            )
            retention_samples = self._retention_sample_statistics(
                real_retention,
                self.scene_conditioned_geometry_gate.minimum_retention,
                self.scene_conditioned_geometry_gate.maximum_retention,
            )
        else:
            retention_samples = None
        return real_memory, reference_memory, diagnostics, retention_samples

    def _enhance(
        self,
        action_queries,
        extension,
        mode: str,
        *,
        scene_override: Tensor | None = None,
    ):
        (
            real_memory,
            reference_memory,
            diagnostics,
            retention_samples,
        ) = self._memory_pair(
            extension, mode, scene_override=scene_override
        )
        enhanced, reader_metrics = self.centered_geometry_reader(
            action_queries, real_memory, reference_memory
        )
        centered_readout = reader_metrics.pop("_centered_readout")
        residual_ratios = reader_metrics.pop(
            "_residual_action_ratio_per_horizon"
        )
        diagnostics.update(reader_metrics)
        return enhanced, diagnostics, {
            "centered_readout": centered_readout,
            "residual_action_ratio_per_horizon": residual_ratios,
            "retention": retention_samples,
        }

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
        pooled_hard = pool_dense_vggt_geometry_per_view(
            self._payloads_from_examples(
                examples, "vggt_dense_hard_negative_cache"
            ),
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )
        return {
            "action_context": None,
            "losses": {},
            "metrics": diagnostics,
            "scene_summary": scene,
            "pooled_geometry": pooled,
            "pooled_hard_geometry": pooled_hard,
            "examples": examples,
            "tokens": [str(example["token"]) for example in examples],
        }

    def _condition_action_queries(self, action_queries, extension):
        if self.gp_mode == "disabled":
            return action_queries, None, {}
        extension["baseline_action_queries"] = action_queries
        real_queries, real_metrics, real_samples = self._enhance(
            action_queries, extension, "real"
        )
        hard_queries, hard_metrics, hard_samples = self._enhance(
            action_queries, extension, "hard_shuffled"
        )
        spatial_queries, spatial_metrics, spatial_samples = self._enhance(
            action_queries, extension, "spatial_shuffled"
        )
        extension["real_action_queries"] = real_queries
        extension["hard_shuffled_action_queries"] = hard_queries
        extension["spatial_shuffled_action_queries"] = spatial_queries
        slot_mean_queries, _, _ = self._enhance(
            action_queries, extension, "slot_mean"
        )
        extension["metrics"]["gp_sq3dmix/slot_mean_identity_max_abs"] = (
            slot_mean_queries.detach() - action_queries.detach()
        ).abs().max()
        real_residual = real_queries.detach().float() - action_queries.detach().float()
        hard_residual = hard_queries.detach().float() - action_queries.detach().float()
        spatial_residual = (
            spatial_queries.detach().float() - action_queries.detach().float()
        )
        extension["metrics"].update(
            {
                "gp_sq3dmix/hard_residual_norm": hard_residual.norm(dim=-1).mean(),
                "gp_sq3dmix/spatial_residual_norm": spatial_residual.norm(dim=-1).mean(),
                "gp_sq3dmix/hard_real_readout_cosine": F.cosine_similarity(
                    hard_samples["centered_readout"].float(),
                    real_samples["centered_readout"].float(),
                    dim=-1,
                ).mean(),
                "gp_sq3dmix/spatial_real_readout_cosine": F.cosine_similarity(
                    spatial_samples["centered_readout"].float(),
                    real_samples["centered_readout"].float(),
                    dim=-1,
                ).mean(),
            }
        )
        extension["metrics"].update(hard_metrics)
        extension["metrics"].update(spatial_metrics)
        scene_vectors = F.normalize(
            extension["scene_summary"].detach().float().squeeze(1),
            dim=-1,
            eps=1e-12,
        )
        if scene_vectors.shape[0] > 1:
            cosine = scene_vectors @ scene_vectors.transpose(0, 1)
            off_diagonal = ~torch.eye(
                scene_vectors.shape[0],
                device=scene_vectors.device,
                dtype=torch.bool,
            )
            extension["metrics"][
                "gp_sq3dmix/scene_summary_cross_scene_pairwise_cosine"
            ] = cosine[off_diagonal].mean()
        self._last_gp_query_samples = {
            "real_residual_action_ratio_per_horizon": real_samples[
                "residual_action_ratio_per_horizon"
            ],
            "hard_residual_action_ratio_per_horizon": hard_samples[
                "residual_action_ratio_per_horizon"
            ],
            "spatial_residual_action_ratio_per_horizon": spatial_samples[
                "residual_action_ratio_per_horizon"
            ],
            "real_retention": real_samples["retention"],
            "scene_summary": extension["scene_summary"].detach(),
            "real_residual": real_residual,
            "hard_residual": hard_residual,
            "spatial_residual": spatial_residual,
        }
        if self.scene_shuffle_diagnostic and self.gp_mode == "gated_residual":
            batch_size = action_queries.shape[0]
            if batch_size > 1:
                ordered = sorted(
                    range(batch_size), key=lambda index: extension["tokens"][index]
                )
                permutation = torch.empty(
                    batch_size, device=action_queries.device, dtype=torch.long
                )
                for offset, target_index in enumerate(ordered):
                    permutation[target_index] = ordered[(offset + 1) % batch_size]
                (
                    scene_shuffled_queries,
                    _,
                    scene_shuffled_samples,
                ) = self.evaluate_scene_shuffled_queries(
                    action_queries, extension, permutation
                )
                extension["scene_shuffled_action_queries"] = scene_shuffled_queries
                scene_residual = (
                    scene_shuffled_queries.detach().float()
                    - action_queries.detach().float()
                )
                real_retention = real_samples["retention"]["_tensor"].float()
                shuffled_retention = scene_shuffled_samples["retention"][
                    "_tensor"
                ].float()
                extension["metrics"].update(
                    {
                        "gp_sq3dmix/scene_shuffled_retention_l2": (
                            shuffled_retention - real_retention
                        ).flatten(1).norm(dim=1).mean(),
                        "gp_sq3dmix/scene_shuffled_residual_l2": (
                            scene_residual - real_residual
                        ).flatten(1).norm(dim=1).mean(),
                        "gp_sq3dmix/scene_shuffle_fixed_point_count": (
                            permutation
                            == torch.arange(batch_size, device=permutation.device)
                        ).float().sum(),
                    }
                )
                self._last_gp_query_samples.update(
                    {
                        "scene_shuffled_residual": scene_residual,
                        "scene_shuffled_retention": {
                            key: value
                            for key, value in scene_shuffled_samples[
                                "retention"
                            ].items()
                            if not key.startswith("_")
                        },
                    }
                )
            else:
                extension["metrics"][
                    "gp_sq3dmix/scene_shuffle_skipped_batch_size_one"
                ] = torch.ones((), device=action_queries.device)
        selected = real_queries
        metrics = real_metrics
        if self.gp_intervention == "hard_shuffled":
            selected, metrics = hard_queries, hard_metrics
        elif self.gp_intervention == "spatial_shuffled":
            selected, metrics = spatial_queries, spatial_metrics
        elif self.gp_intervention != "real":
            selected, metrics, _ = self._enhance(
                action_queries, extension, self.gp_intervention
            )
        extension["metrics"].update(metrics)
        return selected, None, metrics

    def evaluate_scene_shuffled_queries(self, action_queries, extension, permutation):
        """Evaluator-only scene-conditioning intervention for gated variants."""

        if self.gp_mode != "gated_residual":
            raise RuntimeError("scene_shuffled is defined only for gated_residual")
        permutation = torch.as_tensor(
            permutation,
            device=extension["scene_summary"].device,
            dtype=torch.long,
        )
        if permutation.shape != (action_queries.shape[0],):
            raise ValueError("scene-shuffle permutation must be [B]")
        if action_queries.shape[0] > 1 and torch.any(
            permutation == torch.arange(action_queries.shape[0], device=permutation.device)
        ):
            raise ValueError("scene-shuffle permutation must have no fixed point")
        return self._enhance(
            action_queries,
            extension,
            "real",
            scene_override=extension["scene_summary"].index_select(0, permutation),
        )

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
        enhanced, diagnostics, samples = self._enhance(
            action_queries, extension, self.gp_intervention
        )
        self._last_gp_inference_samples = {
            "residual_action_ratio_per_horizon": samples[
                "residual_action_ratio_per_horizon"
            ],
            "retention": samples["retention"],
            "alpha": self.centered_geometry_reader.alpha.detach(),
        }
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
        hard = extension["hard_shuffled_action_queries"].repeat(repeats, 1, 1)
        spatial = extension["spatial_shuffled_action_queries"].repeat(
            repeats, 1, 1
        )
        scene_shuffled = (
            extension["scene_shuffled_action_queries"].repeat(repeats, 1, 1)
            if "scene_shuffled_action_queries" in extension
            else None
        )
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
        # all four conditions share a dropout mask without shifting the next
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
        hard_loss = replay_matched_loss(hard)
        spatial_loss = replay_matched_loss(spatial)
        scene_shuffled_loss = (
            replay_matched_loss(scene_shuffled)
            if scene_shuffled is not None
            else None
        )
        detached_base = base_loss.detach().clamp_min(1e-4)
        hard_margin = self.rank_margin_ratio * detached_base
        spatial_margin = self.spatial_margin_ratio * detached_base
        rank_hard_loss = F.relu(hard_margin + real_loss - hard_loss).mean()
        rank_spatial_loss = F.relu(
            spatial_margin + real_loss - spatial_loss
        ).mean()
        action_loss = real_loss.mean()
        extension["losses"]["geometry_rank_hard"] = rank_hard_loss
        extension["losses"]["geometry_rank_spatial"] = rank_spatial_loss
        if self.training_stage == "stage_a":
            fidelity_loss = F.relu(
                real_loss - (1.0 + self.fidelity_tolerance) * base_loss.detach()
            ).mean()
            extension["losses"]["baseline_fidelity"] = fidelity_loss
        extension["metrics"].update(
            {
                # Runtime smoke records these sentinels in addition to the
                # monkeypatched regression test that verifies object identity
                # and identical RNG draws for every condition.
                "gp_sq3dmix/shared_flow_state_condition_count": torch.tensor(
                    4.0, device=real_loss.device
                ),
                "gp_sq3dmix/shared_dropout_stream_condition_count": torch.tensor(
                    4.0, device=real_loss.device
                ),
                "gp_sq3dmix/base_flow_loss": base_loss.detach().mean(),
                "gp_sq3dmix/real_flow_loss": real_loss.detach().mean(),
                "gp_sq3dmix/hard_flow_loss": hard_loss.detach().mean(),
                "gp_sq3dmix/spatial_flow_loss": spatial_loss.detach().mean(),
                "gp_sq3dmix/relative_hard_real_gap": (
                    (hard_loss.detach() - real_loss.detach()).mean()
                    / real_loss.detach().mean().clamp_min(1e-8)
                ),
                "gp_sq3dmix/relative_spatial_real_gap": (
                    (spatial_loss.detach() - real_loss.detach()).mean()
                    / real_loss.detach().mean().clamp_min(1e-8)
                ),
            }
        )
        self._last_gp_gate_samples = {
            "base_loss": base_loss.detach(),
            "real_loss": real_loss.detach(),
            "hard_loss": hard_loss.detach(),
            "spatial_loss": spatial_loss.detach(),
            "flow_state_id": id(flow_state),
        }
        if scene_shuffled_loss is not None:
            extension["metrics"][
                "gp_sq3dmix/scene_shuffled_flow_loss"
            ] = scene_shuffled_loss.detach().mean()
            self._last_gp_gate_samples[
                "scene_shuffled_loss"
            ] = scene_shuffled_loss.detach()
        return action_loss

    def get_planning_usage_metrics(self) -> dict[str, Tensor]:
        parameters = {
            "gp_sq3dmix/adapter_grad_norm": self.geometry_memory_adapter.feature_projection.weight,
            "gp_sq3dmix/gate_grad_norm": self.scene_conditioned_geometry_gate.gate_projection.weight,
            "gp_sq3dmix/reader_grad_norm": self.centered_geometry_reader.up_projection.weight,
        }
        metrics = dict(getattr(self, "_gp_gradient_norms", {}))
        metrics.update(
            {
                name: parameter.grad.detach().float().norm()
                for name, parameter in parameters.items()
                if parameter.grad is not None
            }
        )
        return metrics

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
