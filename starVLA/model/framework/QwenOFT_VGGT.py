"""Independent VGGT V2/V3 planner with offline teacher supervision.

The model does not load a baseline planner, draft trajectory, or VGGT during
training/inference. V2 uses layer-11 global targets; V3 uses a reconstructable
native-VGGT codec target. At runtime Qwen builds one shared 195-slot memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.vggt_query.alignment import VGGTQueryAligner
from starVLA.model.modules.vggt_query.geometry_memory import (
    SharedGeometryAdapter,
    extract_qwen_spatial_memory,
)
from starVLA.model.modules.vggt_query.planning_heads import (
    AuxiliaryTrajectoryHead,
    PhysicalGeometryHead,
    V3ResidualGeometryFusion,
    WaypointGeometryReader,
)
from starVLA.model.modules.vggt_query.types import (
    VGGTQueryLayout,
    build_vggt_global_query_tokens,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@FRAMEWORK_REGISTRY.register("QwenOFT_VGGT")
class Qwenvl_OFT_VGGT(Qwenvl_OFT):
    """End-to-end planner whose shared 195-slot memory learns VGGT knowledge."""

    def __init__(
        self,
        config: Optional[dict] = None,
        accelerator=None,
        infer_not_load_wan=0,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            accelerator=accelerator,
            infer_not_load_wan=infer_not_load_wan,
            **kwargs,
        )
        self.vggt_enabled = bool(
            OmegaConf.select(config, "framework.vggt.enabled", default=False)
        )
        self._last_planning_context_grad_norm = None
        self._vggt_intervention_mode = "real"
        if not self.vggt_enabled:
            return
        self.vggt_version = int(
            OmegaConf.select(config, "framework.vggt.version", default=2)
        )
        if self.vggt_version not in (2, 3):
            raise ValueError(f"Unsupported VGGT planner version: {self.vggt_version}")
        if (
            bool(config.datasets.video_data.load_2d_data)
            or bool(config.datasets.gs_data.load_3d_data)
            or bool(config.datasets.reward_data.load_reward_data)
            or bool(OmegaConf.select(config, "w_depth", default=False))
        ):
            raise ValueError(
                "QwenOFT_VGGT supports action-only V2/V3 training; disable video, "
                "GS/depth and reward losses."
            )

        layout_cfg = OmegaConf.select(config, "framework.vggt.layout", default={})
        self.vggt_layout = VGGTQueryLayout(
            view_count=int(OmegaConf.select(layout_cfg, "view_count", default=3)),
            special_per_view=int(
                OmegaConf.select(layout_cfg, "special_per_view", default=5)
            ),
            spatial_rows=int(OmegaConf.select(layout_cfg, "spatial_rows", default=6)),
            spatial_cols=int(OmegaConf.select(layout_cfg, "spatial_cols", default=10)),
            teacher_dim=int(OmegaConf.select(layout_cfg, "teacher_dim", default=1024)),
        )
        expected_global = int(
            OmegaConf.select(
                config, "framework.vggt.expected_global_query_count", default=15
            )
        )
        expected_memory = int(
            OmegaConf.select(
                config, "framework.vggt.expected_memory_query_count", default=195
            )
        )
        if self.vggt_layout.special_query_count != expected_global:
            raise ValueError("VGGT global-token count does not match layout")
        if self.vggt_layout.query_count != expected_memory:
            raise ValueError("VGGT memory count does not match layout")

        self.vggt_query_tokens = list(
            build_vggt_global_query_tokens(self.vggt_layout)
        )
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        vocabulary = tokenizer.get_vocab()
        missing_tokens = [token for token in self.vggt_query_tokens if token not in vocabulary]
        if missing_tokens:
            raise RuntimeError(
                "The selected VLM is missing the 15 VGGT global tokens. Run "
                "7-add_vggt_tokens.sh; spatial memory does not use text tokens. "
                f"First missing token: {missing_tokens[0]}"
            )
        vggt_token_ids = tuple(tokenizer.convert_tokens_to_ids(self.vggt_query_tokens))
        if len(set(vggt_token_ids)) != expected_global:
            raise RuntimeError("VGGT global query tokens do not map to unique IDs")
        self._special_token_ids["vggt"] = vggt_token_ids

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        configured_hidden = int(
            OmegaConf.select(config, "framework.qwenvl.vl_hidden_dim", default=hidden_dim)
        )
        if configured_hidden != hidden_dim:
            raise ValueError(
                f"Qwen hidden contract mismatch: config={configured_hidden}, model={hidden_dim}"
            )
        visual = self.qwen_vl_interface.model.model.visual
        self._vggt_spatial_merge_size = int(visual.spatial_merge_size)
        memory_dim = self.vggt_layout.teacher_dim
        adapter_cfg = OmegaConf.select(config, "framework.vggt.geometry_adapter", default={})
        self.vggt_geometry_adapter = SharedGeometryAdapter(
            input_dim=hidden_dim,
            memory_dim=memory_dim,
            expansion=int(OmegaConf.select(adapter_cfg, "expansion", default=2)),
        )
        align_cfg = OmegaConf.select(config, "framework.vggt.alignment", default={})
        alignment_mode = str(OmegaConf.select(align_cfg, "mode", default="raw"))
        self.vggt_scene_residual_enabled = bool(
            OmegaConf.select(align_cfg, "scene_residual_enabled", default=False)
        )
        if alignment_mode != "raw" or self.vggt_scene_residual_enabled:
            raise ValueError(
                "This run intentionally uses raw alignment; scene-residual remains "
                "a disabled configuration/diagnostic interface."
            )
        self.vggt_aligner = VGGTQueryAligner(
            student_dim=memory_dim,
            teacher_dim=memory_dim,
            special_query_count=self.vggt_layout.special_query_count,
            cosine_weight=float(OmegaConf.select(align_cfg, "cosine_weight", default=1.0)),
            smooth_l1_weight=float(
                OmegaConf.select(align_cfg, "smooth_l1_weight", default=0.1)
            ),
            relational_weight=float(
                OmegaConf.select(align_cfg, "relational_weight", default=0.0)
            ),
            scene_relation_weight=1.0,
        )
        reader_cfg = OmegaConf.select(config, "framework.vggt.planner_reader", default={})
        output_query_count = int(
            OmegaConf.select(reader_cfg, "output_query_count", default=8)
        )
        if output_query_count != len(self.act_query_tokens):
            raise ValueError("VGGT reader must emit exactly one readout per action query")
        self.vggt_supervision_enabled = bool(
            OmegaConf.select(config, "framework.vggt.supervision_enabled", default=True)
        )
        self.vggt_access_enabled = bool(
            OmegaConf.select(config, "framework.vggt.access_enabled", default=True)
        )
        if not self.vggt_supervision_enabled and not self.vggt_access_enabled:
            raise ValueError("Enabled VGGT framework needs supervision or planner access")
        self._load_slot_statistics(config)
        if self.vggt_version == 3:
            if self.vggt_slot_mean is None:
                raise RuntimeError(
                    "VGGT V3 needs cache slot statistics for centered residual fusion"
                )
            fusion_cfg = OmegaConf.select(
                config, "framework.vggt.residual_fusion", default={}
            )
            self.vggt_residual_fusion = V3ResidualGeometryFusion(
                action_dim=hidden_dim,
                memory_dim=memory_dim,
                num_heads=int(OmegaConf.select(reader_cfg, "num_heads", default=16)),
                layout=self.vggt_layout,
                minimum_scale=float(
                    OmegaConf.select(fusion_cfg, "minimum_scale", default=0.05)
                ),
                maximum_scale=float(
                    OmegaConf.select(fusion_cfg, "maximum_scale", default=0.50)
                ),
                initial_scale=float(
                    OmegaConf.select(fusion_cfg, "initial_scale", default=0.10)
                ),
                reference_memory=self.vggt_slot_mean,
            )
        else:
            self.vggt_waypoint_reader = WaypointGeometryReader(
                action_dim=hidden_dim,
                memory_dim=memory_dim,
                num_heads=int(OmegaConf.select(reader_cfg, "num_heads", default=16)),
                layout=self.vggt_layout,
            )
            probe_cfg = OmegaConf.select(
                config, "framework.vggt.geometry_probe", default={}
            )
            self.vggt_geometry_probe = PhysicalGeometryHead(
                memory_dim=memory_dim,
                hidden_dim=int(
                    OmegaConf.select(probe_cfg, "hidden_dim", default=512)
                ),
            )
            aux_cfg = OmegaConf.select(
                config, "framework.vggt.aux_plan_head", default={}
            )
            self.vggt_aux_plan_head = AuxiliaryTrajectoryHead(
                input_dim=hidden_dim,
                hidden_dim=int(OmegaConf.select(aux_cfg, "hidden_dim", default=512)),
                action_dim=4,
            )
        self._use_named_loss_contract = True

    def _load_slot_statistics(self, config) -> None:
        log_residual = bool(
            OmegaConf.select(
                config,
                "framework.vggt.alignment.log_scene_residual_metrics",
                default=True,
            )
        )
        if not self.vggt_supervision_enabled or not log_residual:
            self.register_buffer("vggt_slot_mean", None, persistent=False)
            self.register_buffer("vggt_slot_scale", None, persistent=False)
            return
        cache_root = str(
            OmegaConf.select(config, "framework.vggt.cache.root", default="")
        ).strip()
        if not cache_root:
            raise ValueError("VGGT residual diagnostics require framework.vggt.cache.root")
        component = Path(cache_root).expanduser() / "vggt_query"
        manifest_path = component / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing VGGT cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if getattr(self, "vggt_version", 2) == 3:
            expected_representation = "layer11_global_codec_tail_reconstructable"
            if manifest.get("teacher_representation") != expected_representation:
                raise RuntimeError(
                    "VGGT V3 requires the reconstructable native-codec cache; "
                    f"found {manifest.get('teacher_representation')!r}"
                )
            if manifest.get("teacher_layer_index") != 11 or manifest.get(
                "teacher_attention_branch"
            ) != "global":
                raise RuntimeError("VGGT V3 teacher source must be layer-11 global")
            if manifest.get("codec_source_feature") != "layer11_global":
                raise RuntimeError("VGGT V3 codec source contract changed")
            if not manifest.get("codec_gates", {}).get(
                "teacher_codec_downstream", False
            ):
                raise RuntimeError(
                    "VGGT V3 cache was not produced by a codec that passed the "
                    "frozen native downstream gate"
                )
        statistics_name = manifest.get("slot_statistics_file")
        if not statistics_name:
            raise RuntimeError("VGGT cache manifest has no slot_statistics_file")
        statistics_path = component / str(statistics_name)
        if not statistics_path.is_file():
            raise FileNotFoundError(f"Missing VGGT slot statistics: {statistics_path}")
        expected_hash = manifest.get("slot_statistics_sha256")
        if not expected_hash or _sha256_file(statistics_path) != expected_hash:
            raise RuntimeError("VGGT slot statistics checksum mismatch")
        statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
        slot_mean = statistics["slot_mean"].float()
        slot_scale = statistics["slot_scale"].float()
        assert slot_mean.shape == (
            self.vggt_layout.query_count,
            self.vggt_layout.teacher_dim,
        )
        assert slot_scale.shape == (self.vggt_layout.query_count,)
        self.register_buffer("vggt_slot_mean", slot_mean, persistent=False)
        self.register_buffer("vggt_slot_scale", slot_scale, persistent=False)

    def _build_action_prompt_suffix(self) -> str:
        if not getattr(self, "vggt_enabled", False):
            return super()._build_action_prompt_suffix()
        # Action queries precede VGGT global tokens so causal Qwen attention
        # cannot create a hidden global-query bypass into the ActionHead. The
        # explicit waypoint reader/fusion is the only VGGT-memory access path.
        return (
            f" {self.robot_history_token}"
            f"{''.join(self.act_query_tokens)}"
            f"{''.join(self.vggt_query_tokens)}"
        )

    def _build_qwen_batch(self, examples, instructions):
        if getattr(self, "vggt_enabled", False):
            for example in examples:
                payload = example.get("qwen_feature_cache")
                if payload is not None and "vggt_positions" not in payload:
                    raise RuntimeError(
                        "The Qwen feature cache predates VGGT global tokens. Disable "
                        "the Qwen cache or regenerate it with the VGGT prompt."
                    )
        return super()._build_qwen_batch(examples, instructions)

    @staticmethod
    def _gather_queries(last_hidden, positions):
        assert last_hidden.ndim == 3, "last_hidden must be [B,L,H]"
        assert positions.ndim == 2, "query positions must be [B,Q]"
        index = positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
        return last_hidden.gather(dim=1, index=index)

    def _build_student_memory(
        self,
        last_hidden,
        *,
        input_ids,
        image_grid_thw,
        global_positions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input_ids is not None and image_grid_thw is not None
        global_queries = self._gather_queries(last_hidden, global_positions)
        assert global_queries.shape[1] == self.vggt_layout.special_query_count
        spatial_queries, spatial_valid = extract_qwen_spatial_memory(
            last_hidden,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
            spatial_merge_size=self._vggt_spatial_merge_size,
            view_count=self.vggt_layout.view_count,
            output_size=(self.vggt_layout.spatial_rows, self.vggt_layout.spatial_cols),
        )
        memory = self.vggt_geometry_adapter(global_queries, spatial_queries)
        global_valid = torch.ones(
            memory.shape[0],
            self.vggt_layout.special_query_count,
            device=memory.device,
            dtype=torch.bool,
        )
        valid = torch.cat((global_valid, spatial_valid), dim=1)
        assert memory.shape == (
            last_hidden.shape[0],
            self.vggt_layout.query_count,
            self.vggt_layout.teacher_dim,
        )
        return memory, valid

    def _record_planning_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        self._last_planning_context_grad_norm = (
            gradient.detach().float().norm(dim=-1).mean()
        )
        return gradient

    def _compute_query_extension(
        self,
        last_hidden,
        token_positions,
        examples,
        *,
        input_ids=None,
        image_grid_thw=None,
    ):
        if not self.vggt_enabled:
            return super()._compute_query_extension(
                last_hidden,
                token_positions,
                examples,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
            )
        self._last_planning_context_grad_norm = None
        raw_memory, student_valid = self._build_student_memory(
            last_hidden,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            global_positions=token_positions["vggt"],
        )
        losses = {}
        metrics = {
            "student_memory_norm": raw_memory.float().norm(dim=-1).mean().detach(),
        }
        planning_memory = self.vggt_aligner.project_student(raw_memory)
        planner_mask = student_valid

        if self.vggt_supervision_enabled:
            payloads = [example.get("vggt_query_feature_cache") for example in examples]
            if not all(payload is not None for payload in payloads):
                missing = [index for index, payload in enumerate(payloads) if payload is None]
                raise RuntimeError(
                    "VGGT supervision requires a strict offline cache for every sample; "
                    f"missing batch indices: {missing}"
                )
            teacher_features, teacher_masks = [], []
            geometry_targets, geometry_confidences, geometry_masks = [], [], []
            for payload in payloads:
                features = payload["features"]
                mask = payload["valid_mask"].bool()
                assert features.shape == (
                    self.vggt_layout.query_count,
                    self.vggt_layout.teacher_dim,
                )
                assert mask.shape == (self.vggt_layout.query_count,)
                active_mask = payload.get("active_slot_mask")
                if active_mask is not None:
                    assert active_mask.shape == mask.shape
                    mask = mask & active_mask.bool()
                teacher_features.append(features)
                teacher_masks.append(mask)
                if self.vggt_version == 2:
                    geometry_target = payload["geometry_target"]
                    geometry_confidence = payload["geometry_confidence"]
                    geometry_mask = payload["geometry_valid_mask"].bool()
                    assert geometry_target.shape == (
                        self.vggt_layout.spatial_query_count,
                        3,
                    )
                    assert geometry_confidence.shape == geometry_mask.shape == (
                        self.vggt_layout.spatial_query_count,
                    )
                    geometry_targets.append(geometry_target)
                    geometry_confidences.append(geometry_confidence)
                    geometry_masks.append(geometry_mask)
            teacher = torch.stack(teacher_features).to(
                device=raw_memory.device, dtype=torch.float32, non_blocking=True
            )
            alignment_mask = torch.stack(teacher_masks).to(
                device=raw_memory.device, dtype=torch.bool, non_blocking=True
            )
            alignment_mask = alignment_mask & student_valid
            aligned = self.vggt_aligner(
                raw_memory,
                teacher,
                alignment_mask,
                slot_mean=self.vggt_slot_mean,
                slot_scale=self.vggt_slot_scale,
            )
            # This exact representation is shared by alignment, geometry probe,
            # and planning. There is no raw-Qwen bypass.
            planning_memory = aligned.projected_queries
            planner_mask = alignment_mask
            losses.update(
                {
                    "vggt_global_alignment": aligned.losses["global"],
                    "vggt_spatial_alignment": aligned.losses["spatial"],
                    "vggt_scene_relation": aligned.losses["scene_relation"],
                }
            )
            metrics.update(
                {f"alignment_{key}": value for key, value in aligned.metrics.items()}
            )
            if self.vggt_version == 2:
                geometry_target = torch.stack(geometry_targets).to(
                    device=raw_memory.device, dtype=torch.float32, non_blocking=True
                )
                geometry_confidence = torch.stack(geometry_confidences).to(
                    device=raw_memory.device, dtype=torch.float32, non_blocking=True
                )
                geometry_mask = torch.stack(geometry_masks).to(
                    device=raw_memory.device, dtype=torch.bool, non_blocking=True
                )
                geometry_mask = geometry_mask & alignment_mask[
                    :, self.vggt_layout.special_query_count :
                ]
                geometry_output = self.vggt_geometry_probe(
                    planning_memory[:, self.vggt_layout.special_query_count :],
                    geometry_target,
                    geometry_confidence,
                    geometry_mask,
                )
                losses["vggt_geometry"] = geometry_output.loss
                metrics.update(geometry_output.metrics)

        planning_context = planning_memory if self.vggt_access_enabled else None
        if planning_context is not None and planning_context.requires_grad:
            planning_context.register_hook(self._record_planning_gradient)
        return {
            "action_context": planning_context,
            "context_mask": planner_mask,
            "action_targets": [example["action"] for example in examples],
            "losses": losses,
            "metrics": metrics,
        }

    def _condition_action_queries(self, action_queries, extension):
        context = extension.get("action_context")
        if not self.vggt_enabled or not self.vggt_access_enabled or context is None:
            return action_queries, None, {}
        intervention_context = self._apply_vggt_intervention(context)
        intervention_mask = extension["context_mask"]
        if self._vggt_intervention_mode == "shuffled" and context.shape[0] > 1:
            intervention_mask = intervention_mask.roll(shifts=1, dims=0)
        if self.vggt_version == 3:
            conditioned, diagnostics = self.vggt_residual_fusion(
                action_queries, intervention_context, intervention_mask
            )
            extension["metrics"]["planner_context_norm"] = (
                context.float().norm(dim=-1).sum()
                / extension["context_mask"].sum().clamp_min(1)
            ).detach()
            return conditioned, None, diagnostics
        readout, diagnostics = self.vggt_waypoint_reader(
            action_queries, intervention_context, intervention_mask
        )
        target_action = torch.as_tensor(
            np.asarray(extension["action_targets"]),
            device=readout.device,
            dtype=torch.float32,
        )
        assert target_action.shape == (*readout.shape[:2], 4)
        auxiliary = self.vggt_aux_plan_head(readout, target_action)
        extension["losses"]["vggt_aux_plan"] = auxiliary.loss
        extension["metrics"].update(auxiliary.metrics)
        extension["metrics"]["planner_context_norm"] = (
            context.float().norm(dim=-1).sum()
            / extension["context_mask"].sum().clamp_min(1)
        ).detach()
        return action_queries, readout, diagnostics

    def set_vggt_intervention(self, mode: str) -> None:
        """Select a diagnostic-only planning-memory intervention."""

        allowed = {"real", "zero", "shuffled", "slot_mean"}
        if mode not in allowed:
            raise ValueError(f"Unknown VGGT intervention {mode!r}; expected {sorted(allowed)}")
        self._vggt_intervention_mode = mode

    def _apply_vggt_intervention(self, context: torch.Tensor) -> torch.Tensor:
        mode = self._vggt_intervention_mode
        if mode == "real":
            return context
        if mode == "zero":
            return torch.zeros_like(context)
        if mode == "shuffled":
            if context.shape[0] < 2:
                return context
            return context.roll(shifts=1, dims=0)
        if self.vggt_slot_mean is None:
            raise RuntimeError("slot_mean intervention requires cache slot statistics")
        return self.vggt_slot_mean.to(
            device=context.device, dtype=context.dtype
        ).unsqueeze(0).expand(context.shape[0], -1, -1)

    def _condition_inference_action_queries(
        self,
        last_hidden,
        input_ids,
        action_queries,
        *,
        image_grid_thw=None,
        examples=None,
    ):
        del examples
        if not self.vggt_enabled or not self.vggt_access_enabled:
            return action_queries, None, {}
        token_ids = self._special_token_ids["vggt"]
        ids = torch.as_tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device)
        present = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1)).any(dim=1)
        if not present.all():
            raise RuntimeError("Inference prompt is missing one or more VGGT global tokens")
        positions = self._find_token_positions(input_ids, token_ids)
        raw_memory, valid_mask = self._build_student_memory(
            last_hidden,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            global_positions=positions,
        )
        planning_memory = self.vggt_aligner.project_student(raw_memory)
        planning_memory = self._apply_vggt_intervention(planning_memory)
        if self.vggt_version == 3:
            conditioned, diagnostics = self.vggt_residual_fusion(
                action_queries, planning_memory, valid_mask
            )
            return conditioned, None, diagnostics
        readout, diagnostics = self.vggt_waypoint_reader(
            action_queries, planning_memory, valid_mask
        )
        return action_queries, readout, diagnostics

    def get_planning_usage_metrics(self):
        """Return planning-path gradients captured after backward."""

        metrics = {}
        if self._last_planning_context_grad_norm is not None:
            metrics["vggt/planning_context_grad_norm"] = (
                self._last_planning_context_grad_norm
            )
        modules = {
            "geometry_adapter": self.vggt_geometry_adapter.adapter[3].weight,
            "alignment_projection": self.vggt_aligner.student_projection.weight,
        }
        if self.vggt_version == 3:
            modules["residual_fusion"] = (
                self.vggt_residual_fusion.reader.cross_attention.out_proj.weight
            )
            modules["residual_scale"] = self.vggt_residual_fusion.residual_scale_logit
        else:
            modules.update(
                {
                    "waypoint_reader": self.vggt_waypoint_reader.cross_attention.out_proj.weight,
                    "geometry_probe": self.vggt_geometry_probe.head[-1].weight,
                    "aux_plan_head": self.vggt_aux_plan_head.head[-1].weight,
                }
            )
        for name, parameter in modules.items():
            if parameter.grad is not None:
                metrics[f"vggt/{name}_grad_norm"] = parameter.grad.detach().float().norm()
        return metrics
