import torch
from .language import SoftTaskQueries, inject_language_lora
from .memory import pad_tile_registers
from ..drivevla_backbone import DriveVLABackbone

V2_SYSTEM_PROMPT = ('You are an autonomous driving assistant. The visual input is one current front-camera image, '
                    'represented by image tiles. Numeric ego history and navigation are provided as text. '
                    'There are no historical, future, side or rear camera images.')


class V2Backbone(DriveVLABackbone):
    def __init__(self, vlm_path, device='cpu', gradient_checkpointing=True):
        super().__init__(model_type='internvl', checkpoint_path=vlm_path, device=device,
            initialize_from_config=False, extra_token_count=0, strict_vocab_alignment=True,
            use_flash_attn=False, skip_lm_head=True, gradient_checkpointing=gradient_checkpointing,
            planning_registers_enabled=True, planning_register_attention_mode='read_only',
            planning_register_attention_backend='eager', tile_register_aggregation='mean',
            vision_qv_lora_enabled=True, vision_qv_lora_rank=32,
            semantic_frozen_llm_no_grad=False, semantic_backprop_to_vision=True)
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

    def train(self, mode=True):
        super().train(mode)
        # Frozen dropout stays deterministic, checkpoint wrappers selectively train.
        self.model.eval()
        if mode:
            self.activate_gradient_checkpointing_train_mode()
        return self

    def forward(self, pixels, counts, metadata, model_inputs):
        vision = self.encode_internvl_planning_vision(pixels.to(self.compute_dtype), counts, metadata)
        ids = model_inputs['input_ids'].to(pixels.device)
        mask = model_inputs['attention_mask'].to(pixels.device).bool()
        prefix = self.model.language_model.get_input_embeddings()(ids).clone()
        selected = (ids == self.model.img_context_token_id) & mask
        patches = vision.patch_features.reshape(-1, prefix.shape[-1])
        if int(selected.sum()) != len(patches):
            raise ValueError('Prompt image token count does not match patch-only vision output')
        prefix[selected] = patches.to(prefix.dtype)  # CopySlices preserves the visual gradient.
        semantic = self.task_queries(self.model.language_model, prefix, mask)
        content, geometry, valid = pad_tile_registers(vision.per_tile_registers, counts, metadata)
        return dict(visual_content=content, tile_geometry=geometry, scene_valid_mask=valid,
                    semantic_queries=semantic, per_tile_registers=vision.per_tile_registers)
