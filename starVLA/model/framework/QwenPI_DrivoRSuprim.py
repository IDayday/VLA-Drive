"""Baseline-matched Qwen + Flow-DiT + DrivoR + DriveSuprim planner."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.baseline_qwen import (
    apply_baseline_qwen_trainability,
    baseline_qwen_language_forward,
    build_baseline_qwen_batch,
    extract_baseline_action_conditions,
    get_frozen_parameter_names,
    get_trainable_parameter_names,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    MLP,
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
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images


logger = initialize_overwatch(__name__)

QwenFeatureExtractor = Callable[
    ["QwenPIDrivoRSuprim", Sequence[dict]], tuple[Tensor, Tensor, Tensor]
]


def _tensor_batch(values: Sequence[Any], reference: Tensor) -> Tensor:
    return torch.as_tensor(
        np.asarray(values), device=reference.device, dtype=reference.dtype
    )


def _qwen_hidden_size(qwen_vl_interface: nn.Module, fallback: int) -> int:
    model = getattr(qwen_vl_interface, "model", None)
    model_config = getattr(model, "config", None)
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        text_config = getattr(model_config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
    return int(hidden_size if hidden_size is not None else fallback)


def _bf16_context(reference: Tensor):
    if reference.device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optional_cpu(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


@FRAMEWORK_REGISTRY.register("QwenPI-DrivoRSuprim")
class QwenPIDrivoRSuprim(baseframework):
    """One jointly optimized baseline-matched hierarchical planner.

    The generator is the local ``QwenOFT`` action-token path and its original
    Flow-matching head. The only generator extension is optional concatenation
    of one shared 16x256 Q-Former scene memory. Dynamic proposal coordinates
    remain detached from all scorer losses.
    """

    def __init__(
        self,
        config,
        *,
        qwen_vl_interface: Optional[nn.Module] = None,
        action_model: Optional[FlowmatchingActionHead] = None,
        static_vocab: Optional[Tensor] = None,
        static_score_store: Optional[StaticVocabScoreStore] = None,
        qwen_feature_extractor: Optional[QwenFeatureExtractor] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = qwen_vl_interface or get_vlm_model(config=config)
        self.baseline_qwen_trainability = apply_baseline_qwen_trainability(
            self.qwen_vl_interface, config
        )
        self._qwen_feature_extractor = qwen_feature_extractor

        action_cfg = config.framework.action_model
        configured_qwen_dim = int(config.framework.qwenvl.get("vl_hidden_dim", 2048))
        self.qwen_dim = _qwen_hidden_size(
            self.qwen_vl_interface, configured_qwen_dim
        )
        if self.qwen_dim != configured_qwen_dim:
            raise ValueError(
                "configured Qwen hidden size does not match the loaded model: "
                f"{configured_qwen_dim} != {self.qwen_dim}"
            )

        # Match QwenOFT exactly: ego state is injected at the history token and
        # the eight action-token hidden states condition Flow DiT.
        self.act_tok = int(OmegaConf.select(config, "act_tok", default=8))
        self.action_input_model = MLP(
            input_dim=int(action_cfg.action_dim),
            hidden_dim=self.qwen_dim,
            output_dim=self.qwen_dim,
        )
        self._special_token_ids: Dict[str, tuple[int, ...]] = {}
        if self._qwen_feature_extractor is None:
            from starVLA.cache.navsim_feature_cache import (
                GS_QUERY_TOKENS,
                REWARD_QUERY_TOKENS,
                RGB_QUERY_TOKENS,
                ROBOT_HISTORY_TOKEN,
                action_query_tokens,
            )

            tokenizer = self.qwen_vl_interface.processor.tokenizer
            self._special_token_ids = {
                "history": (tokenizer.convert_tokens_to_ids(ROBOT_HISTORY_TOKEN),),
                "rgb": tuple(
                    tokenizer.convert_tokens_to_ids(list(RGB_QUERY_TOKENS))
                ),
                "gs": tuple(
                    tokenizer.convert_tokens_to_ids(list(GS_QUERY_TOKENS))
                ),
                "action": tuple(
                    tokenizer.convert_tokens_to_ids(
                        list(action_query_tokens(self.act_tok))
                    )
                ),
                "reward": tuple(
                    tokenizer.convert_tokens_to_ids(list(REWARD_QUERY_TOKENS))
                ),
            }

        hidden_size = int(action_cfg.hidden_size)
        if hidden_size % 64:
            raise ValueError("baseline Flow DiT hidden_size must be divisible by 64")
        dit_layers = int(action_cfg.diffusion_model_cfg.num_layers)
        action_cfg.DiTConfig = {
            "num_layers": dit_layers,
            "input_embedding_dim": hidden_size,
            "attention_head_dim": 64,
            "num_attention_heads": hidden_size // 64,
        }
        action_model_was_injected = action_model is not None
        self.action_model = action_model or get_action_model(config=config)
        if self.action_model.action_dim != TrajectoryCodec.action_dim:
            raise ValueError("baseline-matched planner requires action_dim=4")
        if int(self.action_model.action_horizon) != 8:
            raise ValueError("baseline-matched planner requires action_horizon=8")
        # Production construction must never silently fall back to the compact
        # 1024x16 experiment. Injected tiny test doubles are exempt.
        if not action_model_was_injected:
            if int(self.action_model.hidden_size) != 1536:
                raise AssertionError("main Flow DiT hidden_size must be 1536")
            if len(self.action_model.model.transformer_blocks) != 24:
                raise AssertionError("main Flow DiT must contain 24 blocks")

        self.action_horizon = int(action_cfg.get("action_horizon", 8))
        self.action_dim = int(action_cfg.get("action_dim", 4))
        self.flow_train_repeats = int(
            action_cfg.get(
                "flow_train_repeats",
                action_cfg.get("repeated_diffusion_steps", 8),
            )
        )
        if self.flow_train_repeats <= 0:
            raise ValueError("flow_train_repeats must be positive")
        if self.action_horizon != 8 or self.action_dim != 4:
            raise ValueError("main action contract must be [B,8,4]")

        scene_cfg = config.framework.scene_encoder
        self.scene_enabled = bool(scene_cfg.get("enabled", True))
        self.inject_scene_into_dit = bool(
            action_cfg.get("use_global_scene_tokens", False)
        )
        if self.inject_scene_into_dit and not self.scene_enabled:
            raise ValueError("scene injection requires scene_encoder.enabled=true")
        self.scene_encoder: Optional[GlobalSceneQFormer]
        self.scene_dim = int(scene_cfg.get("output_dim", 256))
        if self.inject_scene_into_dit and int(
            action_cfg.get("scene_dim", self.scene_dim)
        ) != self.scene_dim:
            raise ValueError(
                "action_model.scene_dim must match scene_encoder.output_dim"
            )
        if self.scene_enabled:
            configured_input_dim = int(scene_cfg.get("input_dim", self.qwen_dim))
            if configured_input_dim != self.qwen_dim:
                raise ValueError(
                    "scene_encoder.input_dim must equal loaded Qwen hidden size"
                )
            self.scene_encoder = GlobalSceneQFormer(
                input_dim=self.qwen_dim,
                hidden_dim=int(scene_cfg.get("hidden_dim", 256)),
                output_dim=self.scene_dim,
                num_queries=int(scene_cfg.get("num_queries", 16)),
                num_layers=int(scene_cfg.get("num_layers", 4)),
                num_heads=int(scene_cfg.get("num_heads", 8)),
                ffn_dim=int(scene_cfg.get("ffn_dim", 1024)),
                dropout=float(scene_cfg.get("dropout", 0.0)),
                query_init_std=float(scene_cfg.get("query_init_std", 0.02)),
                detach_qwen_input=bool(
                    scene_cfg.get("detach_qwen_input", True)
                ),
                use_gradient_checkpointing=bool(
                    scene_cfg.get("use_gradient_checkpointing", True)
                ),
                debug_validate_finite=bool(
                    scene_cfg.get("debug_validate_finite", False)
                ),
            )
            if self.scene_encoder.parameter_count >= 10_000_000:
                raise AssertionError("production Q-Former must remain below 10M params")
        else:
            self.scene_encoder = None

        scorer_cfg = config.framework.hierarchical_scorer
        dynamic_cfg = scorer_cfg.dynamic
        joint_cfg = scorer_cfg.joint
        refinement_cfg = scorer_cfg.refinement
        self.scorer_enabled = bool(scorer_cfg.get("enabled", True))
        self.dynamic_enabled = self.scorer_enabled and bool(
            dynamic_cfg.get("enabled", True)
        )
        self.joint_enabled = self.scorer_enabled and bool(
            joint_cfg.get("enabled", True)
        )
        if self.scorer_enabled and not self.scene_enabled:
            raise ValueError("hierarchical scorers require scene_encoder.enabled=true")
        if self.joint_enabled and not self.dynamic_enabled:
            raise ValueError("B4 joint scoring requires the DrivoR dynamic scorer")
        if bool(scorer_cfg.get("scorer_guides_dit", False)):
            raise ValueError("scorer_guides_dit must remain false")
        for forbidden in (
            "diversity_loss_weight",
            "rl_loss_weight",
            "scorer_guided_generator_weight",
        ):
            if float(scorer_cfg.get(forbidden, 0.0)) != 0.0:
                raise ValueError(f"{forbidden} must remain 0.0")

        self.num_dynamic_candidates = int(dynamic_cfg.get("num_candidates", 1))
        self.candidate_chunk_size = int(
            dynamic_cfg.get("candidate_chunk_size", 8)
        )
        self.final_dynamic_topm = int(dynamic_cfg.get("dynamic_topm", 32))
        self.final_dynamic_topm = int(
            dynamic_cfg.get("final_topm", self.final_dynamic_topm)
        )
        if self.dynamic_enabled and not (
            0 < self.final_dynamic_topm <= self.num_dynamic_candidates
        ):
            raise ValueError("dynamic_topm must lie inside num_dynamic_candidates")
        if self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive")

        ego_state_dim = int(action_cfg.get("state_dim", 4))
        model_dim = int(dynamic_cfg.get("model_dim", 256))
        if self.scene_enabled and self.scene_dim != model_dim and self.scorer_enabled:
            raise ValueError("Q-Former and scorers must share one 256-D space")
        dynamic_prescorer = None
        if self.dynamic_enabled:
            dynamic_prescorer = DrivoRDynamicScorer(
                scene_dim=self.scene_dim,
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
                debug_validate_finite=bool(
                    scorer_cfg.get("debug_validate_finite", False)
                ),
            )

        coarse = fine = None
        self._static_vocab_size = int(joint_cfg.get("vocab_size", 8192))
        self.joint_coarse_topk = int(joint_cfg.get("coarse_topk", 256))
        self.refinement_layers = int(refinement_cfg.get("num_layers", 3))
        if self.joint_enabled:
            if int(refinement_cfg.get("num_stages", 1)) != 1:
                raise ValueError("DriveSuprim supports exactly one refinement stage")
            if int(joint_cfg.get("vocab_num_poses", 40)) != 40:
                raise ValueError("DriveSuprim static vocabulary requires 40 poses")
            joint_model_dim = int(joint_cfg.get("model_dim", 256))
            if joint_model_dim != model_dim or joint_model_dim != self.scene_dim:
                raise ValueError("DrivoR, DriveSuprim, and scene memory must be 256-D")
            if self.joint_coarse_topk > (
                self._static_vocab_size + self.final_dynamic_topm
            ):
                raise ValueError("coarse_topk exceeds the 8224-candidate pool")
            coarse = DriveSuprimCoarseScorer(
                vocab_path=joint_cfg.get("vocab_path"),
                static_vocab=static_vocab,
                vocab_size=self._static_vocab_size,
                num_poses=int(joint_cfg.get("vocab_num_poses", 40)),
                scene_dim=self.scene_dim,
                ego_state_dim=ego_state_dim,
                model_dim=joint_model_dim,
                ffn_dim=int(joint_cfg.get("ffn_dim", 1024)),
                num_heads=int(joint_cfg.get("num_heads", 8)),
                num_layers=int(joint_cfg.get("coarse_layers", 3)),
                coarse_topk=self.joint_coarse_topk,
                dropout=float(joint_cfg.get("dropout", 0.0)),
                normalize_vocab_pos=bool(
                    joint_cfg.get("normalize_vocab_pos", False)
                ),
                debug_validate_finite=bool(
                    scorer_cfg.get("debug_validate_finite", False)
                ),
            )
            fine = DriveSuprimFineRefiner(
                scene_dim=self.scene_dim,
                model_dim=joint_model_dim,
                ffn_dim=int(joint_cfg.get("ffn_dim", 1024)),
                num_heads=int(joint_cfg.get("num_heads", 8)),
                num_layers=self.refinement_layers,
                dropout=float(refinement_cfg.get("dropout", 0.0)),
                use_mid_output=bool(refinement_cfg.get("use_mid_output", True)),
                use_imitation=bool(refinement_cfg.get("use_imitation", True)),
                debug_validate_finite=bool(
                    scorer_cfg.get("debug_validate_finite", False)
                ),
            )

        self.hierarchical_scorer: Optional[HierarchicalDrivoRSuprimScorer]
        if self.scorer_enabled:
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
        else:
            self.hierarchical_scorer = None

        self.trajectory_codec = TrajectoryCodec()
        self.static_score_store = static_score_store
        self._static_score_store_config = config.framework.get(
            "static_score_store", {}
        )

    @property
    def baseline_qwen_trainable_names(self) -> set[str]:
        return set(self.baseline_qwen_trainability.trainable_names)

    @property
    def baseline_qwen_frozen_names(self) -> set[str]:
        return set(self.baseline_qwen_trainability.frozen_names)

    @property
    def supports_dynamic_training(self) -> bool:
        return self.dynamic_enabled

    @property
    def supports_joint_training(self) -> bool:
        return self.joint_enabled

    def _extract_examples(
        self, examples: Sequence[dict]
    ) -> tuple[list[str], list[str], Sequence[Any], Sequence[Any]]:
        if not examples:
            raise ValueError("framework requires a non-empty example batch")
        for index, example in enumerate(examples):
            missing = {"image", "lang", "state"}.difference(example)
            if missing:
                raise KeyError(f"example {index} is missing {sorted(missing)}")
        return (
            [str(example["lang"]) for example in examples],
            [str(example.get("token", "")) for example in examples],
            [example["state"] for example in examples],
            [example.get("action") for example in examples],
        )

    def _baseline_qwen_features(
        self, examples: Sequence[dict]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the exact local QwenOFT history/action-token extraction path."""

        if self._qwen_feature_extractor is not None:
            action_conditions, last_hidden, attention_mask = (
                self._qwen_feature_extractor(self, examples)
            )
            if action_conditions.shape[1] != self.action_horizon:
                raise ValueError("injected action conditions must retain T=8")
            return action_conditions, last_hidden, attention_mask

        instructions = [
            str(example["lang"])
            for example in examples
        ]
        from starVLA.cache.navsim_feature_cache import append_world_action_tokens

        instructions = [
            append_world_action_tokens(instruction, self.act_tok, False)
            for instruction in instructions
        ]
        (
            input_ids,
            attention_mask,
            position_ids,
            token_positions,
            image_embeds,
            deepstack_embeds,
        ) = build_baseline_qwen_batch(
            self.qwen_vl_interface,
            examples,
            instructions,
            self._special_token_ids,
        )
        with _bf16_context(input_ids):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(
                input_ids
            )

        state_values = [example["state"] for example in examples]
        state_embed = self._encode_ego_state_for_qwen(
            state_values, text_embeds
        )
        batch_indices = torch.arange(
            text_embeds.shape[0], device=text_embeds.device
        )
        text_embeds[
            batch_indices, token_positions["history"][:, 0], :
        ] = state_embed

        with _bf16_context(input_ids):
            last_hidden = baseline_qwen_language_forward(
                self.qwen_vl_interface,
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=deepstack_embeds,
            )
        action_conditions = extract_baseline_action_conditions(
            last_hidden, token_positions["action"]
        )
        if action_conditions.shape[1] != 8:
            raise RuntimeError("baseline Qwen path did not produce eight action tokens")
        return action_conditions, last_hidden, attention_mask

    def _encode_ego_state_for_qwen(
        self, state_values: Sequence[Any], text_embeds: Tensor
    ) -> Tensor:
        """Encode history state in the MLP's active DeepSpeed precision."""

        no_autocast = (
            torch.autocast("cuda", enabled=False)
            if text_embeds.device.type == "cuda"
            else nullcontext()
        )
        with no_autocast:
            # DeepSpeed BF16 converts this small MLP's parameters before the
            # first forward.  Match the actual parameter dtype explicitly;
            # otherwise an FP32 numpy state reaches BF16 Linear weights while
            # autocast is disabled.  In ordinary FP32 construction this stays
            # FP32, so the baseline precision policy is unchanged.
            state_reference = next(self.action_input_model.parameters())
            states = torch.as_tensor(
                np.asarray(state_values),
                device=state_reference.device,
                dtype=state_reference.dtype,
            )[:, 0, :]
            state_embed = self.action_input_model(states)
        return state_embed.to(
            device=text_embeds.device, dtype=text_embeds.dtype
        )

    def _encode_scene(
        self, last_hidden: Tensor, attention_mask: Tensor
    ) -> Optional[SceneContext]:
        if self.scene_encoder is None:
            return None
        return self.scene_encoder(last_hidden, attention_mask)

    def _repeat_flow_training_batch(
        self,
        action_conditions: Tensor,
        actions_target: Tensor,
        global_scene_tokens: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """Expand only Flow's batch; T=8 and K remain absent and unchanged."""

        repeat = self.flow_train_repeats
        repeated_conditions = action_conditions.repeat(repeat, 1, 1)
        repeated_actions = actions_target.repeat(repeat, 1, 1)
        repeated_scene = (
            global_scene_tokens.repeat(repeat, 1, 1)
            if global_scene_tokens is not None
            else None
        )
        return repeated_conditions, repeated_actions, repeated_scene

    def _flow_loss(
        self,
        action_conditions: Tensor,
        actions: Tensor,
        scene: Optional[SceneContext],
    ) -> Tensor:
        scene_tokens = scene.global_tokens if self.inject_scene_into_dit else None
        repeated_conditions, repeated_actions, repeated_scene = (
            self._repeat_flow_training_batch(
                action_conditions, actions, scene_tokens
            )
        )
        # FlowmatchingActionHead samples noise [B*repeat,T,D] and timestep
        # [B*repeat] internally, so all eight repetitions are independent.
        return self.action_model(
            repeated_conditions,
            repeated_actions,
            global_scene_tokens=repeated_scene,
        )

    def _sample_dynamic(
        self,
        action_conditions: Tensor,
        scene: SceneContext,
        state: Tensor,
        num_candidates: int,
    ) -> Tensor:
        """Generate K independent proposals on a separate inference interface."""

        was_training = self.action_model.training
        self.action_model.eval()
        try:
            # no_grad encloses only proposal sampling, never the Flow loss.
            with torch.no_grad():
                normalized = self.action_model.predict_multi_action(
                    action_conditions.detach(),
                    state=state.detach(),
                    global_scene_tokens=(
                        scene.global_tokens.detach()
                        if self.inject_scene_into_dit
                        else None
                    ),
                    num_candidates=num_candidates,
                    candidate_chunk_size=self.candidate_chunk_size,
                )
        finally:
            self.action_model.train(was_training)
        return self.trajectory_codec.flow_to_navsim(normalized)

    def _get_static_score_store(self) -> StaticVocabScoreStore:
        if not self.joint_enabled:
            raise RuntimeError("static score labels are only used by DriveSuprim")
        if self.static_score_store is None:
            store_cfg = self._static_score_store_config
            cache_root = store_cfg.get("cache_root")
            if cache_root is None or str(cache_root).strip().lower() in {
                "",
                "null",
                "none",
            }:
                raise FileNotFoundError(
                    "joint training requires static_score_store.cache_root"
                )
            self.static_score_store = StaticVocabScoreStore(
                str(cache_root),
                split=str(store_cfg.get("split", "train")),
                vocab_size=self._static_vocab_size,
                cache_size=int(store_cfg.get("cache_size", 64)),
                mmap=bool(store_cfg.get("mmap", True)),
            )
        return self.static_score_store

    def _default_schedule(self) -> HierarchicalTrainingSchedule:
        return HierarchicalTrainingSchedule(
            progress=1.0,
            dynamic_enabled=self.dynamic_enabled,
            num_dynamic_candidates=(
                self.num_dynamic_candidates if self.dynamic_enabled else 0
            ),
            dynamic_topm=(self.final_dynamic_topm if self.dynamic_enabled else 0),
            lambda_flow=1.0,
            lambda_drivor=1.0 if self.dynamic_enabled else 0.0,
            lambda_suprim_coarse=1.0 if self.joint_enabled else 0.0,
            lambda_suprim_fine=1.0 if self.joint_enabled else 0.0,
        )

    def forward(
        self,
        examples: Sequence[dict],
        training_schedule: Optional[HierarchicalTrainingSchedule] = None,
        metric_supervisor=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return four raw losses for one weighted total and one backward."""

        _, tokens, state_values, action_values = self._extract_examples(examples)
        if any(value is None for value in action_values):
            raise KeyError("training examples require normalized Flow actions")
        action_conditions, last_hidden, attention_mask = (
            self._baseline_qwen_features(examples)
        )
        scene = self._encode_scene(last_hidden, attention_mask)
        state = _tensor_batch(state_values, last_hidden)
        actions = _tensor_batch(action_values, last_hidden)[:, -self.action_horizon :]
        if tuple(actions.shape[-2:]) != (8, 4):
            raise ValueError(
                f"Flow target must end with [8,4], got {tuple(actions.shape)}"
            )
        flow_loss = self._flow_loss(action_conditions, actions, scene)
        zero = flow_loss.new_zeros(())
        schedule = training_schedule or self._default_schedule()

        if not self.scorer_enabled:
            return {
                "losses": {
                    "flow": flow_loss,
                    "drivor": zero,
                    "suprim_coarse": zero,
                    "suprim_fine": zero,
                },
                "metrics": {},
                "predictions": {},
            }
        if self.hierarchical_scorer is None or scene is None:
            raise RuntimeError("enabled scorer is not initialized")

        dynamic_proposals = dynamic_targets = None
        if schedule.dynamic_enabled:
            if not self.dynamic_enabled:
                raise ValueError("schedule enabled dynamics for a non-dynamic model")
            if metric_supervisor is None:
                raise RuntimeError("dynamic training requires a metric supervisor")
            dynamic_proposals = self._sample_dynamic(
                action_conditions,
                scene,
                state,
                schedule.num_dynamic_candidates,
            )
            dynamic_targets = metric_supervisor.score(tokens, dynamic_proposals)

        if self.dynamic_enabled and not self.joint_enabled:
            if not schedule.dynamic_enabled:
                raise ValueError("B3 DrivoR training requires dynamic sampling")
            scorer_output = self.hierarchical_scorer.forward_dynamic_only(
                dynamic_proposals_8=dynamic_proposals,
                dynamic_targets=dynamic_targets,
                global_scene_tokens=scene.global_tokens,
                ego_state=state,
                dynamic_topm=schedule.dynamic_topm,
            )
        elif self.joint_enabled:
            gt_trajectory_8 = self.trajectory_codec.flow_to_navsim(
                actions.detach()
            )
            static_targets = self._get_static_score_store().get(
                tokens, device=last_hidden.device, dtype=last_hidden.dtype
            )
            if schedule.dynamic_enabled:
                scorer_output = self.hierarchical_scorer.forward_full(
                    dynamic_proposals_8=dynamic_proposals,
                    dynamic_targets=dynamic_targets,
                    global_scene_tokens=scene.global_tokens,
                    dense_scene_memory=scene.dense_memory,
                    memory_key_padding_mask=scene.memory_key_padding_mask,
                    ego_state=state,
                    gt_trajectory_8=gt_trajectory_8,
                    static_targets=static_targets,
                    dynamic_topm=schedule.dynamic_topm,
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
        else:
            raise RuntimeError("scorer is enabled without DrivoR or DriveSuprim")

        return {
            "losses": {"flow": flow_loss, **scorer_output["losses"]},
            "metrics": scorer_output["metrics"],
            "predictions": {
                name: value.detach() if torch.is_tensor(value) else value
                for name, value in scorer_output["outputs"].items()
            },
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
        """Run B1/B2 single, B3 DrivoR, or B4 full learned inference."""

        if examples is None and batch_images and isinstance(batch_images[0], dict):
            examples = batch_images
            batch_images = None
        if examples is None:
            if batch_images is None or instructions is None or state is None:
                raise ValueError("inference requires images, instructions, and state")
            image_size = getattr(self.config.datasets.vla_data, "image_size", None)
            if image_size:
                batch_images = resize_images(batch_images, target_size=image_size)
            examples = [
                {
                    "image": image,
                    "lang": instruction,
                    "state": ego_state,
                    "token": "",
                }
                for image, instruction, ego_state in zip(
                    batch_images, instructions, state
                )
            ]
        action_conditions, last_hidden, attention_mask = (
            self._baseline_qwen_features(examples)
        )
        scene = self._encode_scene(last_hidden, attention_mask)
        state_values = [example["state"] for example in examples]
        state_tensor = _tensor_batch(state_values, last_hidden)
        scene_tokens = (
            scene.global_tokens
            if self.inject_scene_into_dit and scene is not None
            else None
        )

        if not self.dynamic_enabled:
            normalized = self.action_model.predict_action(
                action_conditions,
                global_scene_tokens=scene_tokens,
            )
            selected_8 = self.trajectory_codec.flow_to_navsim(normalized)
            batch_size = normalized.shape[0]
            selected = {
                "selected_trajectory_8": selected_8,
                "selected_trajectory_40": self.trajectory_codec.upsample_8_to_40(
                    selected_8[:, None]
                )[:, 0],
                "selected_source": torch.zeros(
                    batch_size, device=normalized.device, dtype=torch.long
                ),
                "selected_absolute_index": torch.zeros(
                    batch_size, device=normalized.device, dtype=torch.long
                ),
                "dynamic_topm_indices": None,
                "coarse_topk_indices": None,
            }
        else:
            if scene is None or self.hierarchical_scorer is None:
                raise RuntimeError("dynamic inference requires scene and scorer")
            dynamic = self._sample_dynamic(
                action_conditions,
                scene,
                state_tensor,
                self.num_dynamic_candidates,
            )
            if self.joint_enabled:
                selected = self.hierarchical_scorer.predict(
                    dynamic_proposals_8=dynamic,
                    global_scene_tokens=scene.global_tokens,
                    dense_scene_memory=scene.dense_memory,
                    memory_key_padding_mask=scene.memory_key_padding_mask,
                    ego_state=state_tensor,
                    dynamic_topm=self.final_dynamic_topm,
                )
            else:
                selected = self.hierarchical_scorer.predict_dynamic_only(
                    dynamic_proposals_8=dynamic,
                    global_scene_tokens=scene.global_tokens,
                    ego_state=state_tensor,
                    dynamic_topm=self.final_dynamic_topm,
                )
            normalized = self.trajectory_codec.navsim_to_flow(
                selected["selected_trajectory_8"]
            )

        return {
            "normalized_actions": normalized.detach().float().cpu().numpy(),
            "trajectory_navsim_8": _optional_cpu(
                selected["selected_trajectory_8"]
            ),
            "trajectory_navsim_40": _optional_cpu(
                selected["selected_trajectory_40"]
            ),
            "selected_source": _optional_cpu(selected["selected_source"]),
            "selected_absolute_index": _optional_cpu(
                selected["selected_absolute_index"]
            ),
            "dynamic_topm_indices": _optional_cpu(
                selected["dynamic_topm_indices"]
            ),
            "coarse_topk_indices": _optional_cpu(
                selected["coarse_topk_indices"]
            ),
        }

    def assert_qwen_trainability(self) -> None:
        """Compare exact current names with the shared baseline manifests."""

        trainable = get_trainable_parameter_names(self.qwen_vl_interface)
        frozen = get_frozen_parameter_names(self.qwen_vl_interface)
        if trainable != self.baseline_qwen_trainable_names:
            missing = sorted(self.baseline_qwen_trainable_names - trainable)
            extra = sorted(trainable - self.baseline_qwen_trainable_names)
            raise RuntimeError(
                "Qwen trainable manifest diverged from baseline: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        if frozen != self.baseline_qwen_frozen_names:
            missing = sorted(self.baseline_qwen_frozen_names - frozen)
            extra = sorted(frozen - self.baseline_qwen_frozen_names)
            raise RuntimeError(
                "Qwen frozen manifest diverged from baseline: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )

    def log_architecture_summary(self, output_logger=logger) -> None:
        """Print all easily-confused production quantities once at startup."""

        named_qwen = list(self.qwen_vl_interface.named_parameters())
        qwen_total = sum(parameter.numel() for _, parameter in named_qwen)
        qwen_trainable = sum(
            parameter.numel()
            for _, parameter in named_qwen
            if parameter.requires_grad
        )
        trainable_groups = sorted(
            {
                ".".join(name.split(".")[:3])
                for name, parameter in named_qwen
                if parameter.requires_grad
            }
        )
        output_logger.info(
            "Qwen: total_params=%d trainable_params=%d frozen_params=%d "
            "trainable_parameter_groups=%s",
            qwen_total,
            qwen_trainable,
            qwen_total - qwen_trainable,
            trainable_groups,
        )
        output_logger.info(
            "DiT: hidden_size=%d num_layers=%d attention_heads=%d",
            int(self.action_model.hidden_size),
            len(self.action_model.model.transformer_blocks),
            int(self.action_model.hidden_size) // 64,
        )
        output_logger.info(
            "Flow: action_horizon=%d flow_train_repeats=%d inference_steps=%d",
            self.action_horizon,
            self.flow_train_repeats,
            int(self.action_model.num_inference_timesteps),
        )
        if self.scene_encoder is not None:
            output_logger.info(
                "Q-Former: query_count=%d hidden_dim=%d layer_count=%d "
                "scene_encoder_parameter_count=%d detach_qwen_input=%s",
                self.scene_encoder.num_queries,
                self.scene_encoder.hidden_dim,
                len(self.scene_encoder.blocks),
                self.scene_encoder.parameter_count,
                self.scene_encoder.detach_qwen_input,
            )
        output_logger.info(
            "Dynamic: num_dynamic_candidates=%d candidate_chunk_size=%d "
            "drivor_topm=%d enabled=%s",
            self.num_dynamic_candidates,
            self.candidate_chunk_size,
            self.final_dynamic_topm,
            self.dynamic_enabled,
        )
        if self.joint_enabled:
            output_logger.info(
                "DriveSuprim: static_vocab_size=%d joint_candidate_count=%d "
                "coarse_topk=%d refinement_layers=%d",
                self._static_vocab_size,
                self._static_vocab_size + self.final_dynamic_topm,
                self.joint_coarse_topk,
                self.refinement_layers,
            )
