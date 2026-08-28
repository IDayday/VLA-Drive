from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils.conversation import get_conv_template
from .utils.internvl_tokenize import build_internvl_model_inputs

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
                 gradient_checkpointing: bool = False):
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

        print(f"Initializing DriveVLA-M0 backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            if initialize_from_config:
                model_config = AutoConfig.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                )
                self.model = AutoModel.from_config(
                    model_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    use_flash_attn=use_flash_attn,
                ).to(self.device).eval()
                print("Initialized InternVL architecture from config; awaiting checkpoint weights.")
            else:
                self.model = AutoModel.from_pretrained(
                    checkpoint_path,
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
                self.model.language_model.resize_token_embeddings(tokenizer_size)
                print(
                    f"Expanded InternVL language embeddings from "
                    f"{embedding_size} to {tokenizer_size}."
                )
            # Load model-specific configuration
            self._configure_internvl()
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
        if hasattr(language_model, "gradient_checkpointing_enable"):
            language_model.gradient_checkpointing_enable()
        elif hasattr(language_model, "_set_gradient_checkpointing"):
            language_model._set_gradient_checkpointing()
        language_model.config.use_cache = False
        print("Enabled InternVL vision and language gradient checkpointing.")

    def _forward_internvl_without_lm_head(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_flags: torch.Tensor,
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

        vit_embeds = internvl_model.extract_feature(pixel_values)
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
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: List[int],
        return_vision=False,
        model_inputs=None,
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
