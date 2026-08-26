"""Qwen + global Q-Former + deterministic Register1/Register64 generator."""

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
)
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.register_planner import (
    RegisterTrajectoryGenerator,
    RegisterTrajectoryLoss,
)
from starVLA.model.modules.scene_encoder import GlobalSceneQFormer, SceneContext
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.visual_training import (
    configure_qwen_visual_backbone,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


QwenHiddenExtractor = Callable[
    ["QwenRegisterGenerator", Sequence[dict]], tuple[Tensor, Tensor]
]


def _qwen_hidden_size(qwen_vl_interface: nn.Module, fallback: int) -> int:
    model = getattr(qwen_vl_interface, "model", None)
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    return int(hidden_size if hidden_size is not None else fallback)


def _autocast_context(reference: Tensor):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if reference.device.type == "cuda"
        else nullcontext()
    )


def _state_batch(values: Sequence[Any], reference: Tensor) -> Tensor:
    state = torch.as_tensor(
        np.asarray(values), device=reference.device, dtype=reference.dtype
    )
    if state.ndim == 3 and state.shape[1] == 1:
        state = state[:, 0]
    if state.ndim != 2 or state.shape[-1] != 4:
        raise ValueError("Register planner ego state must have shape [B,4] or [B,1,4]")
    return state


@FRAMEWORK_REGISTRY.register("QwenRegisterGenerator")
class QwenRegisterGenerator(baseframework):
    """Stage-G framework; no Flow, scorer, metric evaluator, or static store."""

    def __init__(
        self,
        config,
        *,
        qwen_vl_interface: Optional[nn.Module] = None,
        action_input_model: Optional[nn.Module] = None,
        scene_encoder: Optional[GlobalSceneQFormer] = None,
        register_generator: Optional[RegisterTrajectoryGenerator] = None,
        generator_loss: Optional[RegisterTrajectoryLoss] = None,
        qwen_hidden_extractor: Optional[QwenHiddenExtractor] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = qwen_vl_interface or get_vlm_model(config=config)
        self.baseline_qwen_trainability = apply_baseline_qwen_trainability(
            self.qwen_vl_interface, config
        )
        visual = self.qwen_visual
        self.qwen_visual_frozen = not any(
            parameter.requires_grad for parameter in visual.parameters()
        )
        qwenvl_config = config.framework.qwenvl
        has_visual_policy = (
            "freeze_visual" in qwenvl_config
            or "visual_gradient_checkpointing" in qwenvl_config
        )
        if has_visual_policy:
            configured_frozen = bool(qwenvl_config.get("freeze_visual", True))
            if configured_frozen != self.qwen_visual_frozen:
                raise ValueError(
                    "framework.qwenvl.freeze_visual disagrees with "
                    "trainer.freeze_modules"
                )
            configure_qwen_visual_backbone(
                visual,
                freeze_visual=configured_frozen,
                gradient_checkpointing=bool(
                    qwenvl_config.get("visual_gradient_checkpointing", False)
                ),
            )
        self._qwen_hidden_extractor = qwen_hidden_extractor

        configured_qwen_dim = int(config.framework.qwenvl.get("vl_hidden_dim", 2048))
        self.qwen_dim = _qwen_hidden_size(
            self.qwen_vl_interface, configured_qwen_dim
        )
        if self.qwen_dim != configured_qwen_dim:
            raise ValueError(
                "configured Qwen hidden size does not match loaded model: "
                f"{configured_qwen_dim} != {self.qwen_dim}"
            )
        self.act_tok = int(OmegaConf.select(config, "act_tok", default=8))
        self.action_input_model = action_input_model or nn.Sequential(
            nn.Linear(4, self.qwen_dim),
            nn.ReLU(),
            nn.Linear(self.qwen_dim, self.qwen_dim),
        )

        scene_cfg = config.framework.scene_encoder
        if bool(scene_cfg.get("detach_qwen_input", False)):
            raise ValueError("Stage G requires scene_encoder.detach_qwen_input=false")
        configured_scene_input = int(scene_cfg.get("input_dim", self.qwen_dim))
        if configured_scene_input != self.qwen_dim:
            raise ValueError("scene_encoder.input_dim must equal Qwen hidden size")
        self.scene_dim = int(scene_cfg.get("output_dim", 256))
        self.scene_encoder = scene_encoder or GlobalSceneQFormer(
            input_dim=self.qwen_dim,
            hidden_dim=int(scene_cfg.get("hidden_dim", 256)),
            output_dim=self.scene_dim,
            num_queries=int(scene_cfg.get("num_queries", 16)),
            num_layers=int(scene_cfg.get("num_layers", 4)),
            num_heads=int(scene_cfg.get("num_heads", 8)),
            ffn_dim=int(scene_cfg.get("ffn_dim", 1024)),
            dropout=float(scene_cfg.get("dropout", 0.0)),
            query_init_std=float(scene_cfg.get("query_init_std", 0.02)),
            detach_qwen_input=False,
            use_gradient_checkpointing=bool(
                scene_cfg.get("use_gradient_checkpointing", False)
            ),
            debug_validate_finite=bool(scene_cfg.get("debug_validate_finite", False)),
        )
        if self.scene_encoder.detach_qwen_input:
            raise AssertionError("Stage-G Q-Former must preserve Qwen gradients")

        generator_cfg = config.framework.register_generator
        loss_cfg = config.framework.get("generator_loss", {})
        stage_loss_mode = str(loss_cfg.get("stage_loss_mode", "final_only"))
        if int(generator_cfg.get("model_dim", 256)) != self.scene_dim:
            raise ValueError("Register generator and scene encoder dimensions differ")
        self.register_generator = register_generator or RegisterTrajectoryGenerator(
            proposal_num=int(generator_cfg.get("proposal_num", 64)),
            num_poses=int(generator_cfg.get("num_poses", 8)),
            state_dim=int(generator_cfg.get("state_dim", 3)),
            model_dim=int(generator_cfg.get("model_dim", 256)),
            ffn_dim=int(generator_cfg.get("ffn_dim", 1024)),
            num_layers=int(generator_cfg.get("num_layers", 4)),
            num_heads=int(generator_cfg.get("num_heads", 1)),
            one_token_per_trajectory=bool(
                generator_cfg.get("one_token_per_trajectory", True)
            ),
            proj_drop=float(generator_cfg.get("proj_drop", 0.1)),
            drop_path=float(generator_cfg.get("drop_path", 0.2)),
            layer_scale_init=float(generator_cfg.get("layer_scale_init", 0.0)),
            ego_state_dim=int(generator_cfg.get("ego_state_dim", 4)),
            stage_loss_mode=stage_loss_mode,
        )
        if self.register_generator.stage_loss_mode != stage_loss_mode:
            raise ValueError(
                "Register generator topology and generator loss mode differ"
            )
        self.generator_loss = generator_loss or RegisterTrajectoryLoss(
            stage_loss_mode=str(loss_cfg.get("stage_loss_mode", "final_only")),
            stage_loss_weights=loss_cfg.get(
                "stage_loss_weights", [0.1, 0.2, 0.3, 0.5, 1.0]
            ),
            diversity_weight=float(loss_cfg.get("diversity_weight", 0.0)),
        )
        if self.generator_loss.stage_loss_mode != stage_loss_mode:
            raise ValueError(
                "Register generator topology and generator loss instance differ"
            )
        self.trajectory_codec = TrajectoryCodec()
        self._special_token_ids: Dict[str, tuple[int, ...]] = {}
        if self._qwen_hidden_extractor is None:
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
                "rgb": tuple(tokenizer.convert_tokens_to_ids(list(RGB_QUERY_TOKENS))),
                "gs": tuple(tokenizer.convert_tokens_to_ids(list(GS_QUERY_TOKENS))),
                "action": tuple(
                    tokenizer.convert_tokens_to_ids(list(action_query_tokens(self.act_tok)))
                ),
                "reward": tuple(
                    tokenizer.convert_tokens_to_ids(list(REWARD_QUERY_TOKENS))
                ),
            }

    @property
    def qwen_visual(self) -> nn.Module:
        visual = getattr(
            getattr(self.qwen_vl_interface, "model", None), "visual", None
        )
        if not isinstance(visual, nn.Module):
            raise TypeError("Qwen interface does not expose model.visual")
        return visual

    @property
    def baseline_qwen_trainable_names(self) -> set[str]:
        return set(self.baseline_qwen_trainability.trainable_names)

    @property
    def baseline_qwen_frozen_names(self) -> set[str]:
        return set(self.baseline_qwen_trainability.frozen_names)

    def assert_qwen_trainability(self) -> None:
        current_trainable = {
            name
            for name, parameter in self.qwen_vl_interface.named_parameters()
            if parameter.requires_grad
        }
        if current_trainable != self.baseline_qwen_trainable_names:
            raise AssertionError("Qwen trainability changed after framework construction")

    def _extract_examples(
        self, examples: Sequence[dict], *, require_actions: bool
    ) -> tuple[Sequence[Any], Sequence[Any]]:
        if not examples:
            raise ValueError("Register framework requires a non-empty batch")
        states, actions = [], []
        for index, example in enumerate(examples):
            required = {"lang", "state"}
            if self._qwen_hidden_extractor is None:
                required.add("image")
            missing = required.difference(example)
            if missing:
                raise KeyError(f"example {index} is missing {sorted(missing)}")
            if require_actions and example.get("action") is None:
                raise KeyError(f"example {index} has no generator target action")
            states.append(example["state"])
            actions.append(example.get("action"))
        return states, actions

    def _encode_history_state(
        self, state_values: Sequence[Any], text_embeds: Tensor
    ) -> Tensor:
        parameter = next(self.action_input_model.parameters())
        states = _state_batch(state_values, parameter)
        context = (
            torch.autocast("cuda", enabled=False)
            if parameter.device.type == "cuda"
            else nullcontext()
        )
        with context:
            encoded = self.action_input_model(states)
        return encoded.to(device=text_embeds.device, dtype=text_embeds.dtype)

    def _qwen_hidden(self, examples: Sequence[dict]) -> tuple[Tensor, Tensor]:
        if self._qwen_hidden_extractor is not None:
            last_hidden, attention_mask = self._qwen_hidden_extractor(self, examples)
            if last_hidden.ndim != 3 or attention_mask.ndim != 2:
                raise ValueError("injected Qwen features must be [B,L,H] and [B,L]")
            return last_hidden, attention_mask

        from starVLA.cache.navsim_feature_cache import append_world_action_tokens

        instructions = [
            append_world_action_tokens(str(example["lang"]), self.act_tok, False)
            for example in examples
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
        with _autocast_context(input_ids):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)
        state_embed = self._encode_history_state(
            [example["state"] for example in examples], text_embeds
        )
        rows = torch.arange(text_embeds.shape[0], device=text_embeds.device)
        text_embeds[rows, token_positions["history"][:, 0], :] = state_embed
        with _autocast_context(input_ids):
            last_hidden = baseline_qwen_language_forward(
                self.qwen_vl_interface,
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=deepstack_embeds,
            )
        return last_hidden, attention_mask

    def encode_scene_and_generate(
        self, examples: Sequence[dict]
    ) -> tuple[SceneContext, Any, Tensor]:
        state_values, _ = self._extract_examples(examples, require_actions=False)
        last_hidden, attention_mask = self._qwen_hidden(examples)
        with _autocast_context(last_hidden):
            scene = self.scene_encoder(last_hidden, attention_mask)
            ego_state = _state_batch(state_values, scene.global_tokens)
            generator_output = self.register_generator(scene.global_tokens, ego_state)
        return scene, generator_output, ego_state

    def forward(
        self,
        examples: Sequence[dict],
        *,
        generate_only: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if generate_only:
            scene, generator_output, ego_state = self.encode_scene_and_generate(
                examples
            )
            return {
                "scene_context": scene,
                "generator_output": generator_output,
                "ego_state": ego_state,
            }
        state_values, action_values = self._extract_examples(
            examples, require_actions=True
        )
        last_hidden, attention_mask = self._qwen_hidden(examples)
        with _autocast_context(last_hidden):
            scene = self.scene_encoder(last_hidden, attention_mask)
            ego_state = _state_batch(state_values, scene.global_tokens)
            generator_output = self.register_generator(scene.global_tokens, ego_state)
        actions = torch.as_tensor(
            np.asarray(action_values),
            device=scene.global_tokens.device,
            dtype=torch.float32,
        )
        if actions.ndim != 3 or tuple(actions.shape[-2:]) != (8, 4):
            raise ValueError("generator target actions must have shape [B,8,4]")
        gt_trajectory_8 = self.trajectory_codec.flow_to_navsim(actions)
        loss_output = self.generator_loss(
            generator_output.proposal_list, gt_trajectory_8
        )
        return {
            "loss": loss_output.loss,
            "losses": {"trajectory": loss_output.loss},
            "metrics": loss_output.metrics,
            "predictions": {
                "proposals": generator_output.proposals.detach(),
                "winner_index": loss_output.winner_index.detach(),
            },
        }

    def log_architecture_summary(self, logger) -> None:
        trainable_qwen = sum(
            parameter.numel()
            for parameter in self.qwen_vl_interface.parameters()
            if parameter.requires_grad
        )
        trainable_visual = sum(
            parameter.numel()
            for parameter in self.qwen_visual.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "Register generator: proposals=%d decoder=%dx%d heads=%d "
            "proposal_head_mode=%s proposal_heads=%d "
            "qwen_trainable=%d visual_trainable=%d visual_checkpointing=%s "
            "scene_params=%d generator_params=%d",
            self.register_generator.proposal_num,
            self.register_generator.num_layers,
            self.register_generator.model_dim,
            self.register_generator.num_heads,
            self.register_generator.stage_loss_mode,
            self.register_generator.proposal_head_count,
            trainable_qwen,
            trainable_visual,
            bool(getattr(self.qwen_visual, "gradient_checkpointing", False)),
            sum(parameter.numel() for parameter in self.scene_encoder.parameters()),
            sum(parameter.numel() for parameter in self.register_generator.parameters()),
        )
