import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.video_model.videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                            ASPECT_RATIO_RANDOM_CROP_512,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            RandomSampler, get_closest_ratio)
from starVLA.model.modules.video_model.videox_fun.data.dataset_image_video import (ImageVideoDataset,
                                                 ImageVideoSampler,
                                                 get_random_mask)
from starVLA.model.modules.video_model.videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel)
from starVLA.model.modules.video_model.videox_fun.pipeline import WanFunInpaintPipeline, WanFunPipeline
from starVLA.model.modules.video_model.videox_fun.utils.discrete_sampler import DiscreteSampling
from starVLA.model.modules.video_model.videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)

import torch
import os
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import numpy as np
from einops import rearrange
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import math
import pickle
import time


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))

class WanWorldHead(nn.Module):
    def __init__(
        self,
        full_config,
        accelerator = None
    ):
        super().__init__()

        self.full_config = full_config
        config = full_config.framework.video_model
        self.config = config
        wan_config = OmegaConf.load(config.config_path)
        self.wan_config = wan_config
        
        self.transformer3d = WanTransformer3DModel.from_pretrained(
            # os.path.join(config.model_name, wan_config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            config.model_name,
            transformer_additional_kwargs=OmegaConf.to_container(wan_config['transformer_additional_kwargs']),
        )

        if full_config.datasets.video_data.text_input:

            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(config.model_name, wan_config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )

            self.text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join(config.model_name, wan_config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(wan_config['text_encoder_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16
            ).eval()

        # Get Vae
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(config.model_name, wan_config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(wan_config['vae_kwargs']),
        ).eval()

        # Get Clip Image Encoder
        self.clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(config.model_name, wan_config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
        ).eval()

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(wan_config['scheduler_kwargs']))
        )

        self.accelerator = accelerator
        if accelerator == None:
            process_index = 0
            device = 'cpu'
        else:
            process_index = accelerator.process_index
            device = accelerator.device

        self.rng = np.random.default_rng(np.random.PCG64(full_config.seed + process_index))
        self.torch_rng = torch.Generator(device).manual_seed(full_config.seed + process_index)

        self.idx_sampling = DiscreteSampling(config.train_sampling_steps, uniform_sampling=config.uniform_sampling)

        self.qwen_proj_video = MLP(
            input_dim = full_config.framework.qwenvl.vl_hidden_dim,
            hidden_dim = 4096,
            output_dim = 4096,
        )

        # for qwen_embedding
        self.rng_2 = np.random.default_rng(np.random.PCG64(full_config.seed * 137 + process_index))


    def forward(self, rgb_data, prompt_embeds):
        weight_dtype = prompt_embeds.dtype
        cached = [item.get("feature_cache") for item in rgb_data]
        if any(value is not None for value in cached) and not all(value is not None for value in cached):
            raise RuntimeError("A batch cannot mix cached and uncached Wan samples")

        batch = {}
        if self.full_config.datasets.video_data.text_input:
            if all(value is not None for value in cached):
                raise RuntimeError("Wan text-input caching is not supported")
            batch['text'] = [item['text'] for item in rgb_data]

        with torch.no_grad():
            if all(value is not None for value in cached):
                latent_parameters = torch.stack(
                    [value["latent_parameters"] for value in cached]
                ).to(prompt_embeds.device, dtype=weight_dtype, non_blocking=True)
                mask_latent_parameters = torch.stack(
                    [value["mask_latent_parameters"] for value in cached]
                ).to(prompt_embeds.device, dtype=weight_dtype, non_blocking=True)
                latents = DiagonalGaussianDistribution(latent_parameters).sample()
                mask_latents = DiagonalGaussianDistribution(mask_latent_parameters).sample()
                latent_mask = torch.stack(
                    [value["latent_mask"] for value in cached]
                ).to(prompt_embeds.device, dtype=weight_dtype, non_blocking=True)
                clip_context = torch.stack(
                    [value["clip_context"] for value in cached]
                ).to(prompt_embeds.device, dtype=weight_dtype, non_blocking=True)
                t2v_source = [bool(value["mask_is_all_ones"]) for value in cached]
            else:
                batch["pixel_values"] = torch.stack([item['pixel_values'] for item in rgb_data])
                batch["clip_pixel_values"] = torch.stack([item['clip_pixel_values'] for item in rgb_data])
                batch["mask_pixel_values"] = torch.stack([item['mask_pixel_values'] for item in rgb_data])
                batch["mask"] = torch.stack([item['mask'] for item in rgb_data])

                pixel_values = batch["pixel_values"].to(weight_dtype)
                clip_pixel_values = batch["clip_pixel_values"].to(weight_dtype)
                mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                mask = batch["mask"].to(weight_dtype)

                def _batch_encode_vae(values):
                    values = rearrange(values, "b f c h w -> b c f h w")
                    posterior = self.vae.encode(values)[0]
                    return posterior.sample()

                latents = _batch_encode_vae(pixel_values)
                mask_latents = _batch_encode_vae(mask_pixel_values)
                t2v_source = mask.eq(1).flatten(1).all(1).detach().cpu().tolist()
                mask = rearrange(mask, "b f c h w -> b c f h w")
                mask = torch.concat(
                    [
                        torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2),
                        mask[:, :, 1:],
                    ],
                    dim=2,
                )
                mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                mask = mask.transpose(1, 2)
                latent_mask = resize_mask(1 - mask, latents)

                clip_images = []
                for clip_pixel_value in clip_pixel_values:
                    clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                    clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5)
                    clip_images.append(
                        clip_image[:, None, :, :].to(
                            self.clip_image_encoder.device,
                            weight_dtype,
                        )
                    )
                clip_context = self.clip_image_encoder(clip_images)

            t2v_values = []
            for is_all_ones in t2v_source:
                if is_all_ones and np.random.rand() < 0.90:
                    t2v_values.append(0)
                else:
                    t2v_values.append(1)
            t2v_flag = torch.as_tensor(
                t2v_values,
                device=prompt_embeds.device,
                dtype=weight_dtype,
            )

            inpaint_latents = torch.concat([latent_mask, mask_latents], dim=1)
            inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

            qwen_query = self.qwen_proj_video(prompt_embeds)
            clip_keep = []
            qwen_keep = []
            for _ in range(len(rgb_data)):
                if self.rng is None:
                    zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
                else:
                    zero_init_clip_in = self.rng.choice([True, False], p=[0.1, 0.9])
                clip_keep.append(not zero_init_clip_in)

                if self.rng_2 is None:
                    zero_init_clip_in = np.random.choice([True, False], p=[0.0, 1.0])
                else:
                    zero_init_clip_in = self.rng.choice([True, False], p=[0.0, 1.0])
                qwen_keep.append(not zero_init_clip_in)

            clip_keep = torch.as_tensor(clip_keep, device=clip_context.device)
            qwen_keep = torch.as_tensor(qwen_keep, device=qwen_query.device)
            clip_context = torch.where(
                clip_keep.view(-1, *([1] * (clip_context.ndim - 1))),
                clip_context,
                torch.zeros_like(clip_context),
            )
            qwen_query = torch.where(
                qwen_keep.view(-1, 1, 1),
                qwen_query,
                torch.zeros_like(qwen_query),
            )

        if self.full_config.datasets.video_data.text_input:
            with torch.no_grad():
                prompt_ids = self.tokenizer(
                    batch['text'], 
                    padding="max_length", 
                    max_length=512, 
                    truncation=True, 
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
                text_input_ids = prompt_ids.input_ids
                prompt_attention_mask = prompt_ids.attention_mask

                seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                text_embeds = self.text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                text_embeds = torch.stack([u[:v] for u, v in zip(text_embeds, seq_lens)])
        else:
            text_embeds = None


        bsz, channel, num_frames, height, width = latents.size()
        noise = torch.randn(latents.size(), device=latents.device, generator=self.torch_rng, dtype=weight_dtype)

        indices = self.idx_sampling(bsz, generator=self.torch_rng, device=latents.device).long()
        schedule_timesteps = self.noise_scheduler.timesteps.to(latents.device)
        schedule_sigmas = self.noise_scheduler.sigmas.to(latents.device, dtype=latents.dtype)
        timesteps = schedule_timesteps.index_select(0, indices)

        def get_sigmas(step_indices, n_dim=4):
            sigma = schedule_sigmas.index_select(0, step_indices).flatten()
            return sigma.view(sigma.shape[0], *([1] * (n_dim - 1)))

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(indices, n_dim=latents.ndim)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        # Add noise
        target = noise - latents

        target_shape = (self.vae.latent_channels, num_frames, width, height)
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.transformer3d.config.patch_size[1] * self.transformer3d.config.patch_size[2]) *
            target_shape[1]
        )

        if text_embeds is not None:
            qwen_query = torch.cat((qwen_query, text_embeds), 1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            noise_pred = self.transformer3d(
                x=noisy_latents,
                context=qwen_query,
                t=timesteps,
                seq_len=seq_len,
                y=inpaint_latents,
                clip_fea=clip_context,
            )

        # print("dit forward: ", time.time()-a)

        def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
            noise_pred = noise_pred.float()
            target = target.float()
            diff = noise_pred - target
            mse_loss = F.mse_loss(noise_pred, target, reduction='none')
            mask = (diff.abs() <= threshold).float()
            masked_loss = mse_loss * mask
            if weighting is not None:
                masked_loss = masked_loss * weighting
            final_loss = masked_loss.mean()
            return final_loss
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.config.weighting_scheme, sigmas=sigmas)
        loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
        loss = loss.mean()

        sigma = sigmas.to(dtype=noise_pred.dtype, device=noise_pred.device)

        x0_pred = noisy_latents - sigma * noise_pred

        return loss, x0_pred.detach()

    @torch.no_grad()
    def predict_rgb(self, rgb_data, prompt_embeds):

        weight_dtype = prompt_embeds.dtype

        clip_pixel_values = torch.stack([b['clip_pixel_values'] for b in rgb_data]).to(weight_dtype)
        pixel_values = torch.stack([b['pixel_values'] for b in rgb_data]).to(weight_dtype)

        if self.full_config.datasets.video_data.text_input:
            text = [b['text'] for b in rgb_data]


        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(self.wan_config['scheduler_kwargs']))
        )

        pipeline = WanFunInpaintPipeline(
            vae=self.vae, 
            text_encoder=None,
            tokenizer=None,
            transformer=self.transformer3d,
            scheduler=scheduler,
            clip_image_encoder=self.clip_image_encoder,
        )

        pipeline = pipeline.to(self.accelerator.device)

        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.full_config.seed)


        if self.full_config.datasets.video_data.text_input:
            with torch.no_grad():
                prompt_ids = self.tokenizer(
                    text,
                    padding="max_length", 
                    max_length=512, 
                    truncation=True, 
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
                text_input_ids = prompt_ids.input_ids
                prompt_attention_mask = prompt_ids.attention_mask

                seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                text_embeds = self.text_encoder(text_input_ids.to(prompt_embeds.device), attention_mask=prompt_attention_mask.to(prompt_embeds.device))[0]
                text_embeds = torch.stack([u[:v] for u, v in zip(text_embeds, seq_lens)])
        else:
            text_embeds = None

        qwen_query = self.qwen_proj_video(prompt_embeds)
        if text_embeds is not None:
            qwen_query = torch.cat((qwen_query, text_embeds), 1)

        samples = []
        for i in range(len(rgb_data)):
            with torch.autocast("cuda", dtype=weight_dtype):
                
                # video_length = int((9 - 1) // self.vae.config.temporal_compression_ratio * self.vae.config.temporal_compression_ratio) + 1
                # input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[self.config.video_sample_size, self.config.video_sample_size])
                
                # data["pixel_values"]: (F, C, H, W) 或 (1, C, H, W) 首帧
                first_frame = pixel_values[i][0]   # 假设是 (C,H,W)
                C, H, W = first_frame.shape

                # --- 构造 video: 用首帧填充所有时间步 ---
                num_frames = 9   # 或你想要的长度
                video_length = num_frames
                video = first_frame.unsqueeze(1).repeat(1, num_frames, 1, 1)   # (C, F, H, W)
                input_video = video.unsqueeze(0)  # (1, C, F, H, W)

                # --- 构造 mask_video: 0 表示不改，1/255 表示要生成 ---
                # 例子：第 0 帧不改，后面全生成
                mask_video = torch.zeros(1, 1, num_frames, H, W, device=prompt_embeds.device)
                mask_video[:, :, 1:, :, :] = 1.0  # 或 255.0，mask_processor 有 do_binarize

                # 转成 0~255 区间，和官方习惯一致
                input_video_mask = (mask_video * 255).to(prompt_embeds.dtype)

                # sample = pipeline(
                #     # prompt = None,
                #     # negative_prompt = "bad detailed",
                #     height      = self.config.video_sample_size,
                #     width       = self.config.video_sample_size,
                #     num_frames = video_length,
                #     guidance_scale = 6.0,
                #     generator   = generator,
                #     prompt_embeds = self.qwen_proj_video(prompt_embeds),
                #     negative_prompt_embeds = neg_prompt_embeds,

                #     video        = input_video,
                #     mask_video   = input_video_mask,

                #     clip_image = clip_image[i]
                # ).videos



                # no cfg
                sample = pipeline(
                    # prompt = None,
                    # negative_prompt = "bad detailed",
                    height      = H,
                    width       = W,
                    num_frames = video_length,
                    guidance_scale = 1.0,
                    generator   = generator,
                    prompt_embeds = qwen_query[i:i+1],
                    # negative_prompt_embeds = neg_prompt_embeds,

                    video        = input_video,
                    mask_video   = input_video_mask,

                    clip_image = Image.fromarray(np.uint8(clip_pixel_values[i].float().cpu().numpy()))
                ).videos[0]
                samples.append((sample.cpu().numpy()*255.0).astype(np.uint8))

        return samples   # list of video tensor
