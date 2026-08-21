"""Joint Qwen + Flow-DiT + DrivoR + DriveSuprim training framework."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.scene_encoder import GlobalSceneQFormer, SceneContext
from starVLA.model.modules.trajectory_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
    DrivoRDynamicScorer,
    HierarchicalDrivoRSuprimScorer,
    StaticVocabScoreStore,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.hierarchical_schedule import HierarchicalTrainingSchedule
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _tensor_batch(values: Sequence[Any], reference: Tensor) -> Tensor:
    return torch.as_tensor(
        np.asarray(values), device=reference.device, dtype=reference.dtype
    )


@FRAMEWORK_REGISTRY.register("QwenPI-DrivoRSuprim")
class QwenPIDrivoRSuprim(baseframework):
    """A single jointly optimized hierarchical planner with frozen Qwen.

    Qwen is the only pretrained component.  The scene encoder, Flow action
    head, DrivoR pre-scorer, and DriveSuprim scorers are constructed from
    scratch and share one optimizer/backward/checkpoint in the trainer.
    """

    def __init__(
        self,
        config,
        *,
        qwen_vl_interface: Optional[nn.Module] = None,
        action_model: Optional[LayerwiseFlowmatchingActionHead] = None,
        static_vocab: Optional[Tensor] = None,
        static_score_store: Optional[StaticVocabScoreStore] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = qwen_vl_interface or get_vlm_model(config=config)
        self._freeze_qwen()
        if not bool(config.framework.qwenvl.get("freeze", True)):
            raise ValueError("QwenPI-DrivoRSuprim requires qwenvl.freeze=true")

        qwen_dim = int(config.framework.qwenvl.get("vl_hidden_dim", 2048))
        action_cfg = config.framework.action_model
        dit_layers = int(action_cfg.diffusion_model_cfg.num_layers)
        action_cfg.DiTConfig = {
            "num_layers": dit_layers,
            "input_embedding_dim": int(action_cfg.hidden_size),
            "attention_head_dim": 64,
            "num_attention_heads": int(action_cfg.hidden_size) // 64,
        }
        self.action_model = action_model or get_action_model(config=config)
        if self.action_model.action_dim != TrajectoryCodec.action_dim:
            raise ValueError(
                "QwenPI-DrivoRSuprim requires action_dim=4 with "
                "[x_norm,y_norm,sin(heading),cos(heading)]"
            )

        scene_cfg = config.framework.scene_encoder
        scene_dim = int(scene_cfg.get("output_dim", scene_cfg.get("hidden_dim", 2048)))
        if int(action_cfg.get("scene_dim", scene_dim)) != scene_dim:
            raise ValueError("action_model.scene_dim must match scene_encoder output_dim")
        self.scene_encoder = GlobalSceneQFormer(
            input_dim=int(scene_cfg.get("input_dim", qwen_dim)),
            hidden_dim=int(scene_cfg.get("hidden_dim", 2048)),
            output_dim=scene_dim,
            num_queries=int(scene_cfg.get("num_queries", 16)),
            num_layers=int(scene_cfg.get("num_layers", 4)),
            num_heads=int(scene_cfg.get("num_heads", 32)),
            ffn_dim=int(scene_cfg.get("ffn_dim", 8192)),
            dropout=float(scene_cfg.get("dropout", 0.0)),
            query_init_std=float(scene_cfg.get("query_init_std", 0.02)),
            use_gradient_checkpointing=bool(
                scene_cfg.get("use_gradient_checkpointing", True)
            ),
            debug_validate_finite=bool(scene_cfg.get("debug_validate_finite", False)),
        )

        scorer_cfg = config.framework.hierarchical_scorer
        dynamic_cfg = scorer_cfg.dynamic
        joint_cfg = scorer_cfg.joint
        refinement_cfg = scorer_cfg.refinement
        if int(refinement_cfg.get("num_stages", 1)) != 1:
            raise ValueError("DriveSuprim joint training supports exactly one refinement stage")
        if int(dynamic_cfg.get("num_poses", 8)) != 8:
            raise ValueError("DrivoR dynamic proposals require num_poses=8")
        if int(joint_cfg.get("vocab_num_poses", 40)) != 40:
            raise ValueError("DriveSuprim static vocabulary requires 40 poses")
        for forbidden_weight in (
            "diversity_loss_weight",
            "rl_loss_weight",
            "scorer_guided_generator_weight",
        ):
            if float(scorer_cfg.get(forbidden_weight, 0.0)) != 0.0:
                raise ValueError(f"{forbidden_weight} must remain 0.0")
        model_dim = int(dynamic_cfg.get("model_dim", 256))
        joint_model_dim = int(joint_cfg.get("model_dim", model_dim))
        if joint_model_dim != model_dim:
            raise ValueError("DrivoR and DriveSuprim planning model_dim must match")
        ego_state_dim = int(action_cfg.state_dim)
        dynamic_prescorer = DrivoRDynamicScorer(
            scene_dim=scene_dim,
            ego_state_dim=ego_state_dim,
            model_dim=model_dim,
            ffn_dim=int(dynamic_cfg.get("ffn_dim", 1024)),
            num_layers=int(dynamic_cfg.get("num_layers", 4)),
            num_heads=int(dynamic_cfg.get("num_heads", 8)),
            dropout=float(dynamic_cfg.get("dropout", 0.0)),
            noc=float(dynamic_cfg.get("noc", 1.0)),
            dac=float(dynamic_cfg.get("dac", 1.0)),
            ddc=float(dynamic_cfg.get("ddc", 0.0)),
            ttc=float(dynamic_cfg.get("ttc", 5.0)),
            ep=float(dynamic_cfg.get("ep", 5.0)),
            comfort=float(dynamic_cfg.get("comfort", 2.0)),
            debug_validate_finite=bool(scorer_cfg.get("debug_validate_finite", False)),
        )
        coarse = DriveSuprimCoarseScorer(
            vocab_path=joint_cfg.get("vocab_path"),
            static_vocab=static_vocab,
            vocab_size=int(joint_cfg.get("vocab_size", 8192)),
            num_poses=int(joint_cfg.get("vocab_num_poses", 40)),
            scene_dim=scene_dim,
            ego_state_dim=ego_state_dim,
            model_dim=joint_model_dim,
            ffn_dim=int(joint_cfg.get("ffn_dim", 1024)),
            num_heads=int(joint_cfg.get("num_heads", 8)),
            num_layers=int(joint_cfg.get("coarse_layers", 3)),
            coarse_topk=int(joint_cfg.get("coarse_topk", 256)),
            dropout=float(joint_cfg.get("dropout", 0.0)),
            normalize_vocab_pos=bool(joint_cfg.get("normalize_vocab_pos", False)),
            debug_validate_finite=bool(scorer_cfg.get("debug_validate_finite", False)),
        )
        fine = DriveSuprimFineRefiner(
            scene_dim=scene_dim,
            model_dim=joint_model_dim,
            ffn_dim=int(joint_cfg.get("ffn_dim", 1024)),
            num_heads=int(joint_cfg.get("num_heads", 8)),
            num_layers=int(refinement_cfg.get("num_layers", 3)),
            dropout=float(refinement_cfg.get("dropout", 0.0)),
            use_mid_output=bool(refinement_cfg.get("use_mid_output", True)),
            use_imitation=bool(refinement_cfg.get("use_imitation", True)),
            debug_validate_finite=bool(scorer_cfg.get("debug_validate_finite", False)),
        )
        self.hierarchical_scorer = HierarchicalDrivoRSuprimScorer(
            dynamic_prescorer,
            coarse,
            fine,
            detach_scene_for_scorer=bool(
                scorer_cfg.get("detach_scene_for_scorer", False)
            ),
            sigma=float(joint_cfg.get("sigma", 0.5)),
            use_refinement_imitation=bool(
                refinement_cfg.get("use_imitation", True)
            ),
            fine_memory_source=str(
                refinement_cfg.get("memory_source", "dense_qwen_memory")
            ),
        )
        self.trajectory_codec = TrajectoryCodec()
        # Static metric labels are a training-only dependency.  Keep their
        # store lazy so learned-only predict_action neither opens nor requires
        # a PDM/metric cache.
        self.static_score_store = static_score_store
        self._static_score_store_config = config.framework.static_score_store
        self._static_vocab_size = int(joint_cfg.get("vocab_size", 8192))
        self.num_dynamic_candidates = int(dynamic_cfg.get("num_candidates", 64))
        self.final_dynamic_topm = int(dynamic_cfg.get("final_topm", 32))
        self.candidate_chunk_size = int(dynamic_cfg.get("candidate_chunk_size", 8))
        if not (0 < self.final_dynamic_topm <= self.num_dynamic_candidates):
            raise ValueError("dynamic final_topm must not exceed num_candidates")
        if int(joint_cfg.get("coarse_topk", 256)) > int(
            joint_cfg.get("vocab_size", 8192)
        ) + self.final_dynamic_topm:
            raise ValueError("coarse_topk exceeds the final joint candidate count")
        self.repeated_diffusion_steps = int(
            config.trainer.get("repeated_diffusion_steps", 1)
        )
        self.future_action_window_size = int(action_cfg.future_action_window_size)
        if self.future_action_window_size != 8:
            raise ValueError("hierarchical NAVSIM training requires an 8-pose horizon")
        if bool(scorer_cfg.get("scorer_guides_dit", False)):
            raise ValueError("scorer_guides_dit must remain false")

    def _freeze_qwen(self) -> None:
        for parameter in self.qwen_vl_interface.parameters():
            parameter.requires_grad_(False)
        self.qwen_vl_interface.eval()

    def train(self, mode: bool = True) -> "QwenPIDrivoRSuprim":
        super().train(mode)
        self.qwen_vl_interface.eval()
        return self

    def _extract_examples(
        self, examples: Sequence[dict]
    ) -> tuple[List[Any], List[str], List[str], Sequence[Any], Sequence[Any]]:
        if not examples:
            raise ValueError("framework requires a non-empty example batch")
        required = {"image", "lang", "state", "token"}
        for index, example in enumerate(examples):
            missing = required.difference(example)
            if missing:
                raise KeyError(f"example {index} is missing {sorted(missing)}")
        return (
            [example["image"] for example in examples],
            [str(example["lang"]) for example in examples],
            [str(example["token"]) for example in examples],
            [example["state"] for example in examples],
            [example.get("action") for example in examples],
        )

    def _qwen_features(
        self, images: Sequence[Any], instructions: Sequence[str]
    ) -> tuple[List[Tensor], Tensor, Tensor]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is None:
            raise KeyError("Qwen inputs must contain attention_mask")
        with torch.no_grad():
            output = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        expected = len(self.action_model.model.transformer_blocks)
        if hidden_states is None or len(hidden_states) < expected:
            raise RuntimeError(
                f"Qwen returned insufficient hidden layers for {expected} DiT blocks"
            )
        # Preserve QwenPI's actual contract: each DiT block consumes one full
        # sequence from the final N hidden layers (there is no action-token gather).
        layerwise = [value.detach() for value in hidden_states[-expected:]]
        last_hidden = hidden_states[-1].detach()
        if tuple(attention_mask.shape) != tuple(last_hidden.shape[:2]):
            raise ValueError(
                "Qwen full hidden sequence and attention_mask lengths differ: "
                f"{tuple(last_hidden.shape)} versus {tuple(attention_mask.shape)}"
            )
        return layerwise, last_hidden, attention_mask

    def _sample_dynamic(
        self,
        vl_embs_list: List[Tensor],
        state: Tensor,
        global_tokens: Tensor,
        num_candidates: int,
    ) -> Tensor:
        was_training = self.action_model.training
        self.action_model.eval()
        try:
            normalized = self.action_model.predict_multi_action(
                vl_embs_list=[value.detach() for value in vl_embs_list],
                state=state.detach(),
                global_scene_tokens=global_tokens.detach(),
                num_candidates=num_candidates,
                candidate_chunk_size=self.candidate_chunk_size,
            )
        finally:
            self.action_model.train(was_training)
        return self.trajectory_codec.flow_to_navsim(normalized)

    def _flow_loss(
        self,
        vl_embs_list: List[Tensor],
        actions: Tensor,
        state: Tensor,
        global_tokens: Tensor,
    ) -> Tensor:
        repeats = self.repeated_diffusion_steps
        if repeats <= 0:
            raise ValueError("repeated_diffusion_steps must be positive")
        return self.action_model(
            [value.repeat(repeats, 1, 1) for value in vl_embs_list],
            actions.repeat(repeats, 1, 1),
            state.repeat(repeats, 1, 1),
            global_scene_tokens=global_tokens.repeat(repeats, 1, 1),
        )

    def _get_static_score_store(self) -> StaticVocabScoreStore:
        """Construct the training label store on first training forward only."""

        if self.static_score_store is None:
            store_cfg = self._static_score_store_config
            cache_root = store_cfg.get("cache_root")
            if cache_root is None or str(cache_root).strip().lower() in {"", "null", "none"}:
                raise FileNotFoundError(
                    "joint training requires framework.static_score_store.cache_root; "
                    "learned-only predict_action does not"
                )
            self.static_score_store = StaticVocabScoreStore(
                str(cache_root),
                split=str(store_cfg.get("split", "train")),
                vocab_size=self._static_vocab_size,
                cache_size=int(store_cfg.get("cache_size", 64)),
                mmap=bool(store_cfg.get("mmap", True)),
            )
        return self.static_score_store

    def forward(
        self,
        examples: Sequence[dict],
        training_schedule: Optional[HierarchicalTrainingSchedule] = None,
        metric_supervisor=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute four raw losses for one later-combined backward pass."""

        images, instructions, tokens, state_values, action_values = self._extract_examples(
            examples
        )
        if any(value is None for value in action_values):
            raise KeyError("joint training examples require normalized Flow actions")
        layerwise, last_hidden, attention_mask = self._qwen_features(
            images, instructions
        )
        scene: SceneContext = self.scene_encoder(last_hidden, attention_mask)
        state = _tensor_batch(state_values, last_hidden)
        actions = _tensor_batch(action_values, last_hidden)[
            :, -self.future_action_window_size :
        ]
        if tuple(actions.shape[-2:]) != (8, 4):
            raise ValueError(
                f"joint training action target must end with [8,4], got {tuple(actions.shape)}"
            )
        gt_trajectory_8 = self.trajectory_codec.flow_to_navsim(actions.detach())
        static_targets = self._get_static_score_store().get(
            tokens, device=last_hidden.device, dtype=last_hidden.dtype
        )

        if training_schedule is None:
            training_schedule = HierarchicalTrainingSchedule(
                progress=1.0,
                dynamic_enabled=True,
                num_dynamic_candidates=self.num_dynamic_candidates,
                dynamic_topm=self.final_dynamic_topm,
                lambda_flow=1.0,
                lambda_drivor=1.0,
                lambda_suprim_coarse=1.0,
                lambda_suprim_fine=1.0,
            )
        dynamic_proposals = dynamic_targets = None
        if training_schedule.dynamic_enabled:
            if metric_supervisor is None:
                raise RuntimeError(
                    "dynamic curriculum requires a per-rank DynamicMetricSupervisor"
                )
            dynamic_proposals = self._sample_dynamic(
                layerwise,
                state,
                scene.global_tokens,
                training_schedule.num_dynamic_candidates,
            )
            # Online CPU scoring happens before the trainable Flow graph exists.
            dynamic_targets = metric_supervisor.score(tokens, dynamic_proposals)

        flow_loss = self._flow_loss(
            layerwise, actions, state, scene.global_tokens
        )
        if training_schedule.dynamic_enabled:
            scorer_output = self.hierarchical_scorer.forward_full(
                dynamic_proposals_8=dynamic_proposals,
                dynamic_targets=dynamic_targets,
                global_scene_tokens=scene.global_tokens,
                dense_scene_memory=scene.dense_memory,
                memory_key_padding_mask=scene.memory_key_padding_mask,
                ego_state=state,
                gt_trajectory_8=gt_trajectory_8,
                static_targets=static_targets,
                dynamic_topm=training_schedule.dynamic_topm,
            )
        else:
            scorer_output = self.hierarchical_scorer.forward_static_only(
                global_scene_tokens=scene.global_tokens,
                dense_scene_memory=scene.dense_memory,
                memory_key_padding_mask=scene.memory_key_padding_mask,
                ego_state=state,
                gt_trajectory_8=gt_trajectory_8,
                static_targets=static_targets,
            )
        losses = {"flow": flow_loss, **scorer_output["losses"]}
        predictions = {
            name: value.detach() if torch.is_tensor(value) else value
            for name, value in scorer_output["outputs"].items()
        }
        return {
            "losses": losses,
            "metrics": scorer_output["metrics"],
            "predictions": predictions,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Optional[Sequence[Any]] = None,
        instructions: Optional[Sequence[str]] = None,
        state: Optional[Sequence[Any]] = None,
        examples: Optional[Sequence[dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run learned-only hierarchy and retain legacy normalized action output."""

        if examples is None and batch_images and isinstance(batch_images[0], dict):
            examples = batch_images
            batch_images = None
        if examples is not None:
            if batch_images is not None or instructions is not None or state is not None:
                raise ValueError("pass examples or explicit inputs, not both")
            images, instructions, _, state, _ = self._extract_examples(examples)
            batch_images = images
        if batch_images is None or instructions is None or state is None:
            raise ValueError("inference requires images, instructions, and ego state")
        image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if image_size:
            batch_images = resize_images(batch_images, target_size=image_size)
        layerwise, last_hidden, attention_mask = self._qwen_features(
            batch_images, instructions
        )
        scene = self.scene_encoder(last_hidden, attention_mask)
        state_tensor = _tensor_batch(state, last_hidden)
        dynamic = self._sample_dynamic(
            layerwise,
            state_tensor,
            scene.global_tokens,
            self.num_dynamic_candidates,
        )
        selected = self.hierarchical_scorer.predict(
            dynamic_proposals_8=dynamic,
            global_scene_tokens=scene.global_tokens,
            dense_scene_memory=scene.dense_memory,
            memory_key_padding_mask=scene.memory_key_padding_mask,
            ego_state=state_tensor,
            dynamic_topm=self.final_dynamic_topm,
        )
        normalized = self.trajectory_codec.navsim_to_flow(
            selected["selected_trajectory_8"]
        )
        return {
            # NumPy has no bfloat16 dtype; keep the legacy evaluator payload
            # portable by materializing normalized actions as CPU float32.
            "normalized_actions": normalized.detach().float().cpu().numpy(),
            "trajectory_navsim_8": selected["selected_trajectory_8"].detach().cpu(),
            "trajectory_navsim_40": selected["selected_trajectory_40"].detach().cpu(),
            "selected_source": selected["selected_source"].detach().cpu(),
            "selected_absolute_index": selected[
                "selected_absolute_index"
            ].detach().cpu(),
            "dynamic_topm_indices": selected["dynamic_topm_indices"].detach().cpu(),
            "coarse_topk_indices": selected["coarse_topk_indices"].detach().cpu(),
        }

    def assert_qwen_frozen(self) -> None:
        trainable = [
            name
            for name, parameter in self.qwen_vl_interface.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"frozen Qwen has trainable parameters: {trainable[:8]}")
        if self.qwen_vl_interface.training:
            raise RuntimeError("frozen Qwen must remain in eval mode")
