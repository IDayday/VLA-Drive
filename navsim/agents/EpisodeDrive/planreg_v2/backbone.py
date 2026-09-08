import torch
from types import MethodType
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput
from .language import SoftTaskQueries, inject_language_lora
from .memory import pad_tile_registers
from ..drivevla_backbone import DriveVLABackbone
from .timing import StepTiming

V2_SYSTEM_PROMPT = ('You are an autonomous driving assistant. The visual input is one current front-camera image, '
                    'represented by image tiles. Numeric ego history and navigation are provided as text. '
                    'There are no historical, future, side or rear camera images.')


def non_reentrant_vision_encoder(self,inputs_embeds,output_hidden_states=None,return_dict=None):
    """Audited InternVisionEncoder operation sequence, with non-reentrant checkpoint only.

    Needed for read-only same-batch autograd.grad diagnostics. No module/global torch patch.
    """
    output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    return_dict = self.config.use_return_dict if return_dict is None else return_dict
    states = () if output_hidden_states else None
    hidden = inputs_embeds
    for layer in self.layers:
        if output_hidden_states: states += (hidden,)
        hidden = checkpoint(layer,hidden,use_reentrant=False) if self.gradient_checkpointing and self.training else layer(hidden)
    if output_hidden_states: states += (hidden,)
    if not return_dict: return tuple(v for v in (hidden,states) if v is not None)
    return BaseModelOutput(last_hidden_state=hidden,hidden_states=states)


class V2Backbone(DriveVLABackbone):
    def __init__(self, vlm_path, device='cpu', gradient_checkpointing=True, register_init_std=.02):
        from pathlib import Path
        from ..formal_initialization import discover_weight_files,_state_keys_from_weights,scan_forbidden_state_keys
        weight_files = discover_weight_files(Path(vlm_path))
        if not weight_files or scan_forbidden_state_keys(_state_keys_from_weights(weight_files)):
            raise ValueError('V2 initialization requires standalone VLM-only tensors, never agent/planner state')
        super().__init__(model_type='internvl', checkpoint_path=vlm_path, device=device,
            initialize_from_config=False, extra_token_count=0, strict_vocab_alignment=True,
            use_flash_attn=False, skip_lm_head=True, gradient_checkpointing=gradient_checkpointing,
            planning_registers_enabled=True, planning_register_attention_mode='read_only',
            planning_register_attention_backend='eager', tile_register_aggregation='mean',
            vision_qv_lora_enabled=True, vision_qv_lora_rank=32,
            semantic_frozen_llm_no_grad=False, semantic_backprop_to_vision=True,
            planning_register_init_std=register_init_std)
        self.requires_grad_(False)
        self.planning_register_adapter.requires_grad_(True)
        self.planning_register_adapter.float()
        for name, parameter in self.model.vision_model.named_parameters():
            if any(field in name for field in ('q_lora_a.', 'q_lora_b.', 'v_lora_a.', 'v_lora_b.')):
                parameter.requires_grad_(True)
                parameter.data = parameter.data.float()
        self.language_lora_modules = inject_language_lora(self.model.language_model)
        hidden = self.model.language_model.get_input_embeddings().embedding_dim
        self.task_queries = SoftTaskQueries(hidden).to(device=device)
        self.model.system_message = V2_SYSTEM_PROMPT
        self.step_timing=StepTiming()
        vision = self.model.vision_model
        if len(vision.encoder.layers) != 24 or int(vision.config.hidden_size) != 1024:
            raise ValueError('This V2 protocol requires the audited 24-layer InternVL3-2B vision tower')
        if gradient_checkpointing:
            encoder = self.model.vision_model.encoder
            if type(encoder).__name__ != 'InternVisionEncoder':
                raise TypeError('Non-reentrant V2 adapter requires audited InternVisionEncoder')
            encoder.forward = MethodType(non_reentrant_vision_encoder,encoder)
            self.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def train(self, mode=True):
        super().train(mode)
        # Frozen dropout stays deterministic, checkpoint wrappers selectively train.
        self.model.eval()
        if mode:
            self.activate_gradient_checkpointing_train_mode()
        return self

    def forward(self, pixels, counts, metadata, model_inputs):
        with self.step_timing.stage('student_vision_seconds',pixels.is_cuda):
            vision = self.encode_internvl_planning_vision(pixels.to(self.compute_dtype), counts, metadata)
        ids = model_inputs['input_ids'].to(pixels.device)
        mask = model_inputs['attention_mask'].to(pixels.device).bool()
        prefix = self.model.language_model.get_input_embeddings()(ids).clone()
        selected = (ids == self.model.img_context_token_id) & mask
        patches = vision.patch_features.reshape(-1, prefix.shape[-1])
        if int(selected.sum()) != len(patches):
            raise ValueError('Prompt image token count does not match patch-only vision output')
        prefix[selected] = patches.to(prefix.dtype)  # CopySlices preserves the visual gradient.
        with self.step_timing.stage('language_seconds',pixels.is_cuda):
            semantic = self.task_queries(self.model.language_model, prefix, mask)
        content, geometry, valid = pad_tile_registers(vision.per_tile_registers, counts, metadata)
        return dict(visual_content=content, tile_geometry=geometry, scene_valid_mask=valid,
                    semantic_queries=semantic, per_tile_registers=vision.per_tile_registers)
