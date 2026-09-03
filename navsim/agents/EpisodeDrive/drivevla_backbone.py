from dataclasses import asdict, dataclass
import json
import os
from typing import Dict, List, Mapping, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from navsim.planning.training.formal_timing import PhaseTimer

from .utils.conversation import get_conv_template
from .utils.internvl_tokenize import build_internvl_model_inputs
from .layers.planning_registers import (
    InternVLPlanningRegisters,
    inject_internvit_qv_lora,
)

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


_PLANREG_LEGACY_ALLOWED_MISSING_PARTS = (
    "backbone.planning_register_adapter.planning_registers",
    "backbone.planning_register_adapter.register_norm.",
    "backbone.planning_register_adapter.register_projection.",
    "backbone.planning_register_adapter.tile_position_mlp.",
    "backbone.planning_register_adapter.tile_gate",
    "planning_register_adapter.planning_registers",
    "planning_register_adapter.register_norm.",
    "planning_register_adapter.register_projection.",
    "planning_register_adapter.tile_position_mlp.",
    "planning_register_adapter.tile_gate",
    ".q_lora_a.",
    ".q_lora_b.",
    ".v_lora_a.",
    ".v_lora_b.",
    "action_head.semantic_gate",
    "action_head.scene_norm.",
    "action_head._optimizer_step",
    "action_head._total_optimizer_steps",
    "future_register_predictor.",
    "_ema_optimizer_step",
    "_world_model_optimizer_step",
    "_world_model_total_optimizer_steps",
)


@dataclass
class LegacyCheckpointAudit:
    source_key_count: int
    target_key_count: int
    loaded_key_count: int
    direct_key_count: int
    merged_lora_module_count: int
    allowed_missing_keys: List[str]
    invalid_missing_keys: List[str]
    unexpected_source_keys: List[str]
    shape_errors: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalized_checkpoint_key(key: str) -> str:
    return key[len("agent."):] if key.startswith("agent.") else key


def load_exact_student_checkpoint_with_audit(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Restore an exported PlanReg student without legacy key allowances.

    Formal deployment checkpoints have the same student topology as the
    inference model. Unlike historical M0 migration, every tensor must match
    exactly and no PEFT delta may be folded or rescaled.
    """
    normalized_source: Dict[str, torch.Tensor] = {}
    for original_key, value in source_state.items():
        normalized = _normalized_checkpoint_key(str(original_key))
        if normalized in normalized_source:
            raise RuntimeError(
                "Student checkpoint key collision after removing agent prefix: "
                f"{normalized}"
            )
        normalized_source[normalized] = value

    target_state = module.state_dict()
    missing = sorted(set(target_state) - set(normalized_source))
    unexpected = sorted(set(normalized_source) - set(target_state))
    shape_errors = sorted(
        f"{key}: checkpoint={tuple(normalized_source[key].shape)}, "
        f"model={tuple(target_state[key].shape)}"
        for key in set(target_state).intersection(normalized_source)
        if normalized_source[key].shape != target_state[key].shape
    )
    audit: Dict[str, object] = {
        "source_key_count": len(normalized_source),
        "target_key_count": len(target_state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_errors": shape_errors,
        "legacy_lora_merge_applied": False,
        "legacy_lora_scale_applied": False,
    }
    print("EXACT_STUDENT_CHECKPOINT_AUDIT " + json.dumps(audit, sort_keys=True))
    if missing or unexpected or shape_errors:
        raise RuntimeError(
            "Exact formal student checkpoint audit failed: "
            f"missing={missing[:16]}, unexpected={unexpected[:16]}, "
            f"shape_errors={shape_errors[:16]}"
        )
    module.load_state_dict(normalized_source, strict=True)
    return audit


def _target_key_candidates(key: str) -> List[str]:
    candidates = [key]
    if key.startswith("backbone.base_model.model."):
        candidates.append("backbone." + key[len("backbone.base_model.model."):])
    if key.startswith("base_model.model."):
        candidates.append(key[len("base_model.model."):])

    expanded: List[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        if ".base_layer." in candidate:
            expanded.append(candidate.replace(".base_layer.", "."))
        if candidate.endswith(".weight"):
            expanded.append(candidate[:-len(".weight")] + ".base_layer.weight")
        elif candidate.endswith(".bias"):
            expanded.append(candidate[:-len(".bias")] + ".base_layer.bias")
    return list(dict.fromkeys(expanded))


def _find_target_key(
    source_key: str,
    source_value: torch.Tensor,
    target_state: Mapping[str, torch.Tensor],
) -> Tuple[Optional[str], Optional[str]]:
    shape_error = None
    for candidate in _target_key_candidates(source_key):
        if candidate not in target_state:
            continue
        if source_value.shape == target_state[candidate].shape:
            return candidate, None
        shape_error = (
            f"{source_key} -> {candidate}: checkpoint={tuple(source_value.shape)}, "
            f"model={tuple(target_state[candidate].shape)}"
        )
    return None, shape_error


def load_legacy_checkpoint_with_planreg_audit(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    legacy_lora_scale: float = 2.0,
) -> LegacyCheckpointAudit:
    """Strictly restore a legacy agent, merging PEFT only when topology changed.

    The original PEFT topology is loaded directly when it still exists. In the
    PlanReg topology, frozen legacy PEFT deltas are folded into their base
    weights so the new trainable adaptation consists only of InternViT Q/V
    LoRA. Only the explicitly new planning/QV tensors may remain missing.
    """
    normalized_source: Dict[str, torch.Tensor] = {}
    for original_key, value in source_state.items():
        normalized = _normalized_checkpoint_key(original_key)
        if normalized in normalized_source:
            raise RuntimeError(
                f"Checkpoint key collision after removing agent prefix: {normalized}"
            )
        normalized_source[normalized] = value

    target_state = module.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    consumed = set()
    source_to_target: Dict[str, str] = {}
    shape_errors: List[str] = []

    # Direct loading also preserves the exact legacy PEFT path when the target
    # still has that topology.
    for source_key, source_value in normalized_source.items():
        target_key, shape_error = _find_target_key(
            source_key, source_value, target_state
        )
        if target_key is not None:
            if target_key in loaded:
                raise RuntimeError(
                    f"Multiple checkpoint tensors map to target key {target_key}"
                )
            loaded[target_key] = source_value
            consumed.add(source_key)
            source_to_target[source_key] = target_key
        elif shape_error is not None:
            shape_errors.append(shape_error)

    direct_key_count = len(loaded)
    merged_lora_modules = 0
    lora_a_suffix = ".lora_A.default.weight"
    lora_b_suffix = ".lora_B.default.weight"
    for a_key, a_weight in normalized_source.items():
        if a_key in consumed or not a_key.endswith(lora_a_suffix):
            continue
        root = a_key[:-len(lora_a_suffix)]
        b_key = root + lora_b_suffix
        base_key = root + ".base_layer.weight"
        b_weight = normalized_source.get(b_key)
        if b_weight is None:
            shape_errors.append(f"{a_key} has no matching {b_key}")
            continue
        target_key = source_to_target.get(base_key)
        if target_key is None:
            base_weight = normalized_source.get(base_key)
            if base_weight is not None:
                target_key, shape_error = _find_target_key(
                    base_key, base_weight, target_state
                )
                if shape_error is not None:
                    shape_errors.append(shape_error)
            if target_key is None:
                shape_errors.append(
                    f"Cannot map legacy LoRA base weight {base_key} into target model"
                )
                continue
        base_value = loaded.get(target_key)
        if base_value is None:
            shape_errors.append(
                f"Legacy LoRA target {target_key} was not populated by {base_key}"
            )
            continue
        if a_weight.ndim != 2 or b_weight.ndim != 2:
            shape_errors.append(f"Legacy LoRA tensors must be matrices: {a_key}, {b_key}")
            continue
        delta = torch.matmul(b_weight.float(), a_weight.float())
        delta.mul_(float(legacy_lora_scale))
        if delta.shape != base_value.shape:
            shape_errors.append(
                f"Legacy LoRA delta {root} has shape {tuple(delta.shape)}, "
                f"base is {tuple(base_value.shape)}"
            )
            continue
        loaded[target_key] = base_value + delta.to(base_value.dtype)
        consumed.update((a_key, b_key))
        merged_lora_modules += 1

    missing = sorted(set(target_state) - set(loaded))
    allowed_missing = sorted(
        key
        for key in missing
        if any(part in key for part in _PLANREG_LEGACY_ALLOWED_MISSING_PARTS)
    )
    invalid_missing = sorted(set(missing) - set(allowed_missing))
    unexpected = sorted(set(normalized_source) - consumed)
    audit = LegacyCheckpointAudit(
        source_key_count=len(normalized_source),
        target_key_count=len(target_state),
        loaded_key_count=len(loaded),
        direct_key_count=direct_key_count,
        merged_lora_module_count=merged_lora_modules,
        allowed_missing_keys=allowed_missing,
        invalid_missing_keys=invalid_missing,
        unexpected_source_keys=unexpected,
        shape_errors=sorted(set(shape_errors)),
    )
    print("LEGACY_CHECKPOINT_AUDIT " + json.dumps(audit.to_dict(), sort_keys=True))
    if invalid_missing or unexpected or shape_errors:
        raise RuntimeError(
            "Legacy checkpoint audit failed: "
            f"invalid_missing={invalid_missing}, "
            f"unexpected={unexpected}, shape_errors={sorted(set(shape_errors))}"
        )

    incompatible = module.load_state_dict(loaded, strict=False)
    if sorted(incompatible.missing_keys) != allowed_missing:
        raise RuntimeError(
            "Legacy checkpoint load returned an un-audited missing-key set: "
            f"expected={allowed_missing}, actual={sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Legacy checkpoint load produced unexpected keys: {incompatible.unexpected_keys}"
        )
    return audit

class DriveVLABackbone(nn.Module):
    """
    The DriveVLA-M0 vision-language backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda",
                 extra_token_count: int = 0,
                 target_vocab_size: Optional[int] = None,
                 use_flash_attn: bool = True,
                 initialize_from_config: bool = False,
                 skip_lm_head: bool = False,
                 gradient_checkpointing: bool = False,
                 planning_registers_enabled: bool = False,
                 num_planning_registers: int = 16,
                 planning_register_dim: int = 256,
                 tile_register_aggregation: str = "mean",
                 planning_register_attention_mode: str = "bidirectional",
                 planning_register_attention_backend: str = "eager",
                 vision_qv_lora_enabled: bool = False,
                 vision_qv_lora_rank: int = 32,
                 vision_qv_lora_dropout: float = 0.0,
                 strict_vocab_alignment: bool = False,
                 semantic_frozen_llm_no_grad: bool = False,
                 semantic_backprop_to_vision: bool = True):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device
        self.skip_lm_head = skip_lm_head
        self.gradient_checkpointing_enabled = bool(gradient_checkpointing)
        self._formal_phase_timer = PhaseTimer(
            os.getenv("PLANREG_FORMAL_TIMING", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self.strict_vocab_alignment = bool(strict_vocab_alignment)
        self.semantic_frozen_llm_no_grad = bool(semantic_frozen_llm_no_grad)
        self.semantic_backprop_to_vision = bool(semantic_backprop_to_vision)
        if self.semantic_frozen_llm_no_grad and self.semantic_backprop_to_vision:
            raise ValueError(
                "frozen_llm_no_grad=true is incompatible with "
                "semantic_path.backprop_to_vision=true"
            )
        self.planning_registers_enabled = bool(planning_registers_enabled)
        self.planning_register_attention_mode = str(
            planning_register_attention_mode
        )
        self.vision_qv_lora_enabled = bool(vision_qv_lora_enabled)
        self.planning_register_adapter = None
        self.injected_vision_qv_lora_layers: Tuple[str, ...] = ()

        if (
            self.planning_registers_enabled or self.vision_qv_lora_enabled
        ) and self.model_type != "internvl":
            raise NotImplementedError(
                "PlanReg-WM-V1 supports only the confirmed InternVL/InternViT "
                "runtime. A Qwen3-VL planning-register adapter is not implemented."
            )

        if (
            self.planning_registers_enabled
            and self.planning_register_attention_mode == "read_only"
            and use_flash_attn
        ):
            raise RuntimeError(
                "planning_registers.attention_mode=read_only requires "
                "vlm_config.use_flash_attn=false"
            )

        print(f"Initializing DriveVLA-M0 backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            if initialize_from_config:
                model_config = AutoConfig.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                )
                if hasattr(model_config, "vision_config"):
                    model_config.vision_config.use_flash_attn = bool(use_flash_attn)
                self.model = AutoModel.from_config(
                    model_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    use_flash_attn=use_flash_attn,
                ).to(self.device).eval()
                print("Initialized InternVL architecture from config; awaiting checkpoint weights.")
            else:
                runtime_config = None
                if self.planning_register_attention_mode == "read_only":
                    runtime_config = AutoConfig.from_pretrained(
                        checkpoint_path,
                        trust_remote_code=True,
                    )
                    runtime_config.vision_config.use_flash_attn = False
                self.model = AutoModel.from_pretrained(
                    checkpoint_path,
                    config=runtime_config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_flash_attn=use_flash_attn,
                    device_map=self.device
                ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            if self.strict_vocab_alignment and extra_token_count:
                raise ValueError(
                    "strict_vocab_alignment=true prohibits adding synthetic tokens "
                    "at runtime; prepare an auditable standalone VLM checkpoint"
                )
            if extra_token_count:
                extra_tokens = [
                    f"<DRIVEVLA_EXTRA_{index}>"
                    for index in range(int(extra_token_count))
                ]
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": extra_tokens}
                )

            tokenizer_size = len(self.tokenizer)
            if target_vocab_size is not None and tokenizer_size != int(target_vocab_size):
                raise ValueError(
                    f"Expanded tokenizer has {tokenizer_size} entries, "
                    f"expected {target_vocab_size}"
                )
            embedding_size = self.model.language_model.get_input_embeddings().num_embeddings
            if embedding_size != tokenizer_size:
                if self.strict_vocab_alignment:
                    raise RuntimeError(
                        "Formal VLM initialization rejected a tokenizer/embedding "
                        f"mismatch ({tokenizer_size} tokenizer IDs versus "
                        f"{embedding_size} embedding rows). Silent resize would change "
                        "the scientific initialization; prepare and audit an aligned "
                        "standalone VLM checkpoint first."
                    )
                self.model.language_model.resize_token_embeddings(tokenizer_size)
                print(
                    f"Expanded InternVL language embeddings from "
                    f"{embedding_size} to {tokenizer_size}."
                )
            # Load model-specific configuration
            self._configure_internvl()
            if self.planning_registers_enabled:
                InternVLPlanningRegisters.validate_runtime_structure(self.model)
                vision_model = self.model.vision_model
                hidden_dim = int(vision_model.config.hidden_size)
                reference_parameter = next(vision_model.parameters())
                self.planning_register_adapter = InternVLPlanningRegisters(
                    vision_hidden_dim=hidden_dim,
                    num_registers=int(num_planning_registers),
                    register_dim=int(planning_register_dim),
                    tile_aggregation=tile_register_aggregation,
                    attention_mode=planning_register_attention_mode,
                    read_only_attention_backend=planning_register_attention_backend,
                    use_flash_attn=bool(use_flash_attn),
                    device=reference_parameter.device,
                    dtype=reference_parameter.dtype,
                )
                self.planning_register_adapter.configure_vision_attention(
                    vision_model
                )
            if self.vision_qv_lora_enabled:
                self.injected_vision_qv_lora_layers = inject_internvit_qv_lora(
                    self.model.vision_model,
                    rank=int(vision_qv_lora_rank),
                    dropout=float(vision_qv_lora_dropout),
                )
            if gradient_checkpointing:
                self._enable_internvl_gradient_checkpointing()
            self.num_image_token = 256

        elif self.model_type == 'qwen':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")

    def _enable_internvl_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing for full VLM fine-tuning."""
        vision_model = self.model.vision_model
        if hasattr(vision_model, "gradient_checkpointing"):
            vision_model.gradient_checkpointing = True
        if hasattr(vision_model, "encoder") and hasattr(
            vision_model.encoder, "gradient_checkpointing"
        ):
            vision_model.encoder.gradient_checkpointing = True

        language_model = self.model.language_model
        language_model.config.use_cache = False
        if not self.semantic_frozen_llm_no_grad:
            if hasattr(language_model, "gradient_checkpointing_enable"):
                language_model.gradient_checkpointing_enable()
            elif hasattr(language_model, "_set_gradient_checkpointing"):
                language_model._set_gradient_checkpointing()
            print("Enabled InternVL vision and language gradient checkpointing.")
        else:
            print(
                "Enabled InternVL vision gradient checkpointing; frozen semantic "
                "LLM runs under no_grad without language checkpoint wrappers."
            )

    def activate_gradient_checkpointing_train_mode(self) -> None:
        """Activate checkpoint wrappers without enabling frozen-model dropout.

        PlanReg keeps the frozen VLM in eval mode, but both the local InternViT
        encoder and Transformers checkpointing layers additionally key off
        their own ``training`` flag. Set only those wrapper flags after
        ``eval()``; child attention/drop-path/dropout modules remain in eval
        mode, so activation checkpointing does not change the model function.
        """
        if not self.gradient_checkpointing_enabled:
            return
        if self.model_type != "internvl":
            raise RuntimeError(
                "Selective frozen-model checkpointing is implemented only for InternVL"
            )
        vision_encoder = getattr(self.model.vision_model, "encoder", None)
        if vision_encoder is None:
            raise RuntimeError("InternVL gradient checkpointing requires vision encoder")
        vision_encoder.training = True

        if bool(getattr(self, "semantic_frozen_llm_no_grad", False)):
            return
        decoder = getattr(self.model.language_model, "model", None)
        decoder_layers = getattr(decoder, "layers", None)
        if decoder_layers is None or len(decoder_layers) == 0:
            raise RuntimeError(
                "InternVL gradient checkpointing requires language_model.model.layers"
            )
        for layer in decoder_layers:
            if not bool(getattr(layer, "gradient_checkpointing", False)):
                raise RuntimeError(
                    "Language decoder layer was not configured for gradient checkpointing"
                )
            layer.training = True
        if not vision_encoder.training or not all(
            layer.training for layer in decoder_layers
        ):
            raise RuntimeError(
                "Failed to activate frozen InternVL checkpoint wrapper flags"
            )

    def _forward_internvl_without_lm_head(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_flags: torch.Tensor,
        vit_embeds: Optional[torch.Tensor] = None,
    ):
        """Return decoder hidden states without materializing vocabulary logits.

        DriveVLA consumes only the final hidden state. Calling the causal-LM
        wrapper would still execute its 151k-way ``lm_head`` even though the
        logits are discarded, wasting both compute and activation memory.
        """
        internvl_model = self.model
        image_flags = image_flags.squeeze(-1)
        input_embeds = internvl_model.language_model.get_input_embeddings()(
            input_ids
        ).clone()

        if vit_embeds is None:
            vit_embeds = internvl_model.extract_feature(pixel_values)
        image_flags = image_flags.to(vit_embeds.device)
        vit_embeds = vit_embeds[image_flags == 1]
        batch_size, sequence_length, hidden_size = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(-1, hidden_size)
        flat_input_ids = input_ids.reshape(-1)
        selected = flat_input_ids == internvl_model.img_context_token_id
        flat_vit_embeds = vit_embeds.reshape(-1, hidden_size).to(
            flat_input_embeds.device
        )
        if int(selected.sum()) != flat_vit_embeds.shape[0]:
            raise RuntimeError(
                "InternVL image-token mismatch while bypassing lm_head: "
                f"prompt has {int(selected.sum())} image tokens but vision "
                f"encoder produced {flat_vit_embeds.shape[0]} tokens"
            )
        flat_input_embeds[selected] = (
            flat_input_embeds[selected] * 0.0 + flat_vit_embeds
        )
        input_embeds = flat_input_embeds.reshape(
            batch_size, sequence_length, hidden_size
        )

        decoder = getattr(internvl_model.language_model, "model", None)
        if decoder is None:
            raise RuntimeError(
                "skip_lm_head requires a causal language model exposing its "
                "decoder as `.model`"
            )
        return decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def _forward_internvl_from_patch_features(
        self,
        patch_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_flags: torch.Tensor,
    ):
        """Run the frozen LLM from patch-only InternVL features."""
        if self.skip_lm_head:
            return self._forward_internvl_without_lm_head(
                pixel_values=patch_features.new_empty(0),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                vit_embeds=patch_features,
            )

        internvl_model = self.model
        image_flags = image_flags.squeeze(-1).to(patch_features.device)
        input_embeds = internvl_model.language_model.get_input_embeddings()(
            input_ids
        ).clone()
        patch_features = patch_features[image_flags == 1]
        batch_size, sequence_length, hidden_size = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(-1, hidden_size)
        flat_input_ids = input_ids.reshape(-1)
        selected = flat_input_ids == internvl_model.img_context_token_id
        flat_patch_features = patch_features.reshape(-1, hidden_size).to(
            flat_input_embeds.device
        )
        if int(selected.sum()) != flat_patch_features.shape[0]:
            raise RuntimeError(
                "InternVL image-token mismatch with planning registers: "
                f"prompt has {int(selected.sum())} image tokens but patch-only "
                f"vision features contain {flat_patch_features.shape[0]} tokens"
            )
        flat_input_embeds[selected] = (
            flat_input_embeds[selected] * 0.0 + flat_patch_features
        )
        input_embeds = flat_input_embeds.reshape(
            batch_size, sequence_length, hidden_size
        )
        return internvl_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def _forward_semantic_language_path(
        self,
        patch_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_flags: torch.Tensor,
    ):
        """Apply the explicit formal semantic-path gradient boundary."""
        patch_features_for_llm = (
            patch_features
            if bool(getattr(self, "semantic_backprop_to_vision", True))
            else patch_features.detach()
        )
        if bool(getattr(self, "semantic_frozen_llm_no_grad", False)):
            with torch.no_grad():
                return self._forward_internvl_from_patch_features(
                    patch_features_for_llm,
                    input_ids,
                    attention_mask,
                    position_ids,
                    image_flags,
                )
        return self._forward_internvl_from_patch_features(
            patch_features_for_llm,
            input_ids,
            attention_mask,
            position_ids,
            image_flags,
        )

    def encode_internvl_planning_vision(
        self,
        pixel_values: torch.Tensor,
        num_patches_list: List[int],
        tile_metadata: Optional[torch.Tensor] = None,
    ):
        """Encode InternViT patches/registers without invoking the LLM."""
        if not self.planning_registers_enabled or self.planning_register_adapter is None:
            raise RuntimeError("InternVL planning registers are not enabled")
        return self.planning_register_adapter(
            self.model,
            pixel_values,
            num_patches_list,
            tile_metadata,
        )

    def forward_internvl_with_planning_registers(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_flags: torch.Tensor,
        num_patches_list: List[int],
        tile_metadata: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return LLM hidden states plus InternViT-internal scene registers."""
        phase_timer = getattr(self, "_formal_phase_timer", None)
        vision_timer = (
            phase_timer.start("student_vision_time")
            if phase_timer is not None
            else None
        )
        planning_output = self.encode_internvl_planning_vision(
            pixel_values,
            num_patches_list,
            tile_metadata,
        )
        if phase_timer is not None:
            phase_timer.stop(vision_timer)
        language_timer = (
            phase_timer.start("frozen_llm_time")
            if phase_timer is not None
            else None
        )
        language_output = self._forward_semantic_language_path(
            planning_output.patch_features,
            input_ids,
            attention_mask,
            position_ids,
            image_flags,
        )
        if phase_timer is not None:
            phase_timer.stop(language_timer)
        hidden_states = getattr(language_output, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "InternVL language model did not return hidden_states for the "
                "planning-register path"
            )
        return {
            "last_hidden_state": hidden_states[-1],
            "planning_registers": planning_output.scene_registers,
            "per_tile_registers": planning_output.per_tile_registers,
        }

    def consume_formal_timings(self) -> Dict[str, float]:
        phase_timer = getattr(self, "_formal_phase_timer", None)
        return {} if phase_timer is None else phase_timer.consume()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: List[int],
        return_vision=False,
        model_inputs=None,
        tile_metadata: Optional[torch.Tensor] = None,
    ):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
            
        if model_inputs is None:
            model_inputs = build_internvl_model_inputs(
                self.tokenizer,
                questions,
                num_patches_list,
                system_message,
                self.num_image_token,
            )
        device = pixel_values.device
        input_ids = model_inputs['input_ids'].to(device, non_blocking=True)
        attention_mask = model_inputs['attention_mask'].to(device, non_blocking=True)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

        if self.planning_registers_enabled:
            if return_vision:
                return self.encode_internvl_planning_vision(
                    pixel_values.bfloat16(),
                    num_patches_list,
                    tile_metadata,
                )
            return self.forward_internvl_with_planning_registers(
                pixel_values=pixel_values.bfloat16(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                num_patches_list=num_patches_list,
                tile_metadata=tile_metadata,
            )
        if return_vision:
            return self.model.vision_model(
                pixel_values=pixel_values.bfloat16(),
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            if self.skip_lm_head:
                return self._forward_internvl_without_lm_head(
                    pixel_values=pixel_values.bfloat16(),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_flags=image_flags,
                )
            return self.model(
                    pixel_values=pixel_values.bfloat16(),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_flags=image_flags.squeeze(-1),
                    output_hidden_states=True,
                    return_dict=True,
            )
