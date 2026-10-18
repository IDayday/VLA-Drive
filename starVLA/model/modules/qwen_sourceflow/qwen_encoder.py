"""Official Qwen3-VL forward and processor, exposing all current image tokens."""
import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

CAMERAS = ('cam_l0', 'cam_f0', 'cam_r0')
IMAGE_HEIGHT, IMAGE_WIDTH = 384, 672


def pack_indices(mask):
    lengths = mask.sum(1)
    if (lengths == 0).any():
        raise ValueError('Every scene requires visual and text evidence')
    width = int(lengths.max())
    indices = torch.zeros((len(mask), width), device=mask.device, dtype=torch.long)
    valid = torch.arange(width, device=mask.device)[None] < lengths[:, None]
    for i, row in enumerate(mask):
        indices[i, :lengths[i]] = row.nonzero().flatten()
    return indices, valid


class QwenObservationProcessor:
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        # One resize, by the official processor. The target area is fixed;
        # smart_resize preserves aspect ratio up to the native 32-pixel rounding.
        self.processor.image_processor.size = {'shortest_edge': IMAGE_HEIGHT * IMAGE_WIDTH,
                                              'longest_edge': IMAGE_HEIGHT * IMAGE_WIDTH}
        self.processor.tokenizer.padding_side = 'right'

    def __call__(self, observations):
        messages, images, views = [], [], []
        for obs in observations:
            state = obs['proprio']
            if len(state) != 8:
                raise ValueError('Expected current vx,vy,ax,ay and four-way command (including unknown)')
            content = [dict(type='text', text=(
                'Plan the ego vehicle trajectory for the next 4 seconds. '
                f'Navigation: {obs["navigation"]}. '
                f'Current velocity (m/s): x={float(state[0]):.4f}, y={float(state[1]):.4f}. '
                f'Current acceleration (m/s^2): x={float(state[2]):.4f}, y={float(state[3]):.4f}. '
                'Current synchronized camera views follow in left/front/right order.'))]
            available = []
            for camera in CAMERAS:
                im = obs['images'].get(camera)
                content.append(dict(type='text', text=f'Camera {camera}: ' + ('missing.' if im is None else '')))
                if im is not None:
                    content.append(dict(type='image', image=im))
                    images.append(im); available.append(CAMERAS.index(camera))
            if not available:
                raise ValueError('All views missing; infrastructure/data error')
            messages.append([dict(role='user', content=content)])
            views.append(available)
        text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        batch = self.processor(text=text, images=images, padding=True, return_tensors='pt')
        batch['camera_indices'] = views
        return batch


class QwenEncoder(nn.Module):
    def __init__(self, model_path, *, dtype=torch.float32, attention='sdpa'):
        super().__init__()
        full, info = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
            dtype=dtype, attn_implementation=attention, local_files_only=True,
            output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
            raise ValueError('Generic Qwen weights did not load exactly: ' + repr(info))
        self.model = full.model  # no vocabulary output head / no language CE
        cfg = self.model.config
        if cfg.model_type != 'qwen3_vl' or cfg.text_config.hidden_size != 2048 or cfg.text_config.num_hidden_layers != 28:
            raise ValueError('The registered generic backbone is Qwen3-VL-2B-Instruct')
        self.hidden_dim, self.num_layers = cfg.text_config.hidden_size, cfg.text_config.num_hidden_layers
        self.merge_size = cfg.vision_config.spatial_merge_size
        self.image_token_id = cfg.image_token_id
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        device = next(self.parameters()).device
        kwargs = {k: v.to(device) for k, v in batch.items() if k in
                  ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
        out = self.model(**kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states
        if hidden is None or len(hidden) != self.num_layers + 1:
            raise ValueError('Official hidden-state indexing changed: embedding + 28 block outputs required')
        valid = kwargs['attention_mask'].bool()
        visual = (kwargs['input_ids'] == self.image_token_id) & valid
        vi, vm = pack_indices(visual)
        ti, tm = pack_indices(valid & ~visual)
        positions = torch.zeros((*vm.shape, 3), device=device, dtype=torch.long)
        camera_ids = torch.full(vm.shape, -1, device=device, dtype=torch.long)
        grids = kwargs['image_grid_thw'].tolist()
        cursor = 0
        for b, cameras in enumerate(batch['camera_indices']):
            offset = 0
            # Reserve the same tile width for a missing view; no black-image token.
            scene_grids = grids[cursor:cursor+len(cameras)]
            tile_width = max(g[2] // self.merge_size for g in scene_grids)
            for camera, (t, h, w) in zip(cameras, scene_grids):
                if t != 1 or h % self.merge_size or w % self.merge_size:
                    raise ValueError('Unexpected current-image merge grid')
                mh, mw = h // self.merge_size, w // self.merge_size
                n = mh * mw
                # Official merger order: row-major blocks after spatial merging.
                positions[b, offset:offset+n, 1] = torch.arange(mh, device=device).repeat_interleave(mw)
                positions[b, offset:offset+n, 2] = torch.arange(mw, device=device).repeat(mh) + camera * tile_width
                camera_ids[b, offset:offset+n] = camera
                offset += n
            if offset != int(vm[b].sum()):
                raise ValueError('Image token count differs from image_grid_thw / merge^2')
            cursor += len(cameras)
        if cursor != len(grids):
            raise ValueError('Image grids cross scene boundary')
        return dict(hidden_states=hidden, final_hidden=out.last_hidden_state,
                    visual_indices=vi, visual_mask=vm, visual_positions=positions,
                    camera_ids=camera_ids, text_indices=ti, text_mask=tm)
