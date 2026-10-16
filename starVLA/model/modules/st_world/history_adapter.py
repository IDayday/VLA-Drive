"""Causal history adapters around the unchanged S0 Qwen/Flow interfaces.

Budget mode inserts 96 ordinary memory positions before ego/W/action. Their
embeddings come from a two-layer latent reader of the SAME frozen Qwen visual.
They are never image placeholders and never receive an invented image grid or
DeepStack stream. Native mode sends each existing real image through the actual
processor and is a layout reference, not a numerically equivalent architecture.

Design basis: OpenFlamingo 655f693fbfa04cd6e9a987d960654624d48917cf,
helpers.py PerceiverAttention/Resampler (media+latent K/V, residual attention
and FFN). This is an independent PyTorch adaptation with explicit physical
time, camera, xy, ego-motion, cross-group fusion and missing-data semantics.
It is not a claim that OpenFlamingo implements these driving adaptations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from typing import Optional

import numpy as np
import torch
from torch import nn

from starVLA.dataloader.st_world_schema import ModelInputs
from starVLA.dataloader.st_world_history import collate_history_inputs, VIEWS, HISTORY_TIMES
from starVLA.model.modules.foresight.tokens import replace_query_embeddings
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed as seeded_initialization


@dataclass(frozen=True)
class HistoryVisualInput:
    tokens: torch.Tensor             # B,2,3,N,D; frozen final-merged Qwen output
    patch_valid: torch.Tensor        # B,2,3,N
    frame_valid: torch.Tensor        # B,2,3
    xy: torch.Tensor                 # B,2,3,N,2; real merged-grid cell centers [0,1]
    actual_dt_s: torch.Tensor        # B,2; strictly past, no target-derived mask
    ego_relative: torch.Tensor       # B,2,4,4; current<-past, metres/radians
    identities: tuple

    def validate(self):
        if self.tokens.ndim != 5 or self.tokens.shape[1:3] != (2, 3) or min(self.tokens.shape) < 1:
            raise ValueError('History final-merged tokens must be nonempty B,2,3,N,D')
        b = len(self.tokens)
        if (self.patch_valid.dtype != torch.bool or self.patch_valid.shape != self.tokens.shape[:-1] or
                self.frame_valid.dtype != torch.bool or self.frame_valid.shape != (b, 2, 3) or
                not torch.equal(self.patch_valid.any(-1), self.frame_valid)):
            raise ValueError('History patch/frame validity mismatch')
        if (self.tokens.requires_grad or not self.tokens.is_floating_point() or
                not torch.isfinite(self.tokens[self.patch_valid]).all()):
            raise ValueError('History visual tokens must be finite, frozen observations')
        if (self.xy.shape != (*self.tokens.shape[:-1], 2) or not self.xy.is_floating_point() or
                not torch.isfinite(self.xy[self.patch_valid]).all() or
                ((self.xy[self.patch_valid] < 0) | (self.xy[self.patch_valid] > 1)).any()):
            raise ValueError('Actual merged-grid xy required')
        if self.actual_dt_s.shape != (b, 2) or not torch.isfinite(self.actual_dt_s).all():
            raise ValueError('History requires actual physical past times')
        expected = self.actual_dt_s.new_tensor(HISTORY_TIMES)
        if (self.actual_dt_s >= 0).any() or ((self.actual_dt_s - expected).abs() > .050001).any():
            raise ValueError('Only causal -1/-0.5s frames within registered tolerance are allowed')
        if (self.ego_relative.shape != (b, 2, 4, 4) or not torch.isfinite(self.ego_relative).all() or
                not torch.allclose(self.ego_relative[..., 3, :],
                                   self.ego_relative.new_tensor([0., 0., 0., 1.]).expand(b, 2, 4))):
            raise ValueError('Finite homogeneous observed ego transforms required')
        if len(self.identities) != b:
            raise ValueError('History observation identity count changed')
        if any(x.device != self.tokens.device for x in (self.patch_valid, self.frame_valid, self.xy,
                                                       self.actual_dt_s, self.ego_relative)):
            raise ValueError('History inputs must share one device')
        return self


@dataclass(frozen=True)
class HistoryMemory:
    tokens: torch.Tensor  # B,2,3,16,D_llm
    valid: torch.Tensor   # B,2,3,16

    def validate(self):
        if (self.tokens.ndim != 5 or self.tokens.shape[1:4] != (2, 3, 16) or
                self.valid.shape != self.tokens.shape[:-1] or self.valid.dtype != torch.bool or
                self.tokens.device != self.valid.device or not torch.isfinite(self.tokens).all()):
            raise ValueError('Expected finite registered 96-slot history memory and Boolean mask')
        if (self.tokens[~self.valid] != 0).any():
            raise ValueError('Missing history cannot become nonzero memory')
        return self


class _LatentReadoutBlock(nn.Module):
    """Media+latent K/V with an explicit mask and a safe all-missing sentinel."""
    def __init__(self, dim, heads):
        super().__init__()
        self.media_norm, self.query_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, queries, media, query_valid, media_valid):
        q = self.query_norm(queries)
        kv = torch.cat((self.media_norm(media), q, torch.zeros_like(q[:, :1])), dim=1)
        has_any = media_valid.any(1) | query_valid.any(1)
        valid = torch.cat((media_valid, query_valid, (~has_any)[:, None]), dim=1)
        update = self.attention(q, kv, kv, key_padding_mask=~valid, need_weights=False)[0]
        result = queries + update
        result = result + self.ff(self.ff_norm(result))
        return torch.where(query_valid[..., None], result, 0.)


class HistoryResampler(nn.Module):
    """Two layers, 16 identifiable queries per past time/view; all-past fusion."""
    def __init__(self, visual_dim, llm_dim, *, dim, heads):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (visual_dim, llm_dim, dim, heads)) or dim % heads:
            raise ValueError('Explicit real visual/LLM dimensions and compatible readout heads required')
        self.visual_dim, self.llm_dim, self.dim, self.heads = visual_dim, llm_dim, dim, heads
        self.project = nn.Linear(visual_dim, dim)
        self.xy = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ego = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.camera = nn.Embedding(3, dim)
        self.queries = nn.Parameter(torch.randn(16, dim) * .02)
        self.blocks = nn.ModuleList([_LatentReadoutBlock(dim, heads) for _ in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, llm_dim))

    def forward(self, visual: HistoryVisualInput):
        visual.validate()
        if visual.tokens.shape[-1] != self.visual_dim:
            raise ValueError('History visual channel contract changed')
        # FP32 metadata arithmetic, then inherit the runtime's AMP policy.
        safe = torch.where(visual.patch_valid[..., None], visual.tokens, 0.)
        media = self.project(safe)
        seconds = visual.actual_dt_s.float()
        temporal = self.time(torch.stack((seconds, torch.sin(torch.pi * seconds),
                                         torch.cos(torch.pi * seconds)), -1).to(media.dtype))
        pose = visual.ego_relative.float()
        yaw = torch.atan2(pose[..., 1, 0], pose[..., 0, 0])
        ego = self.ego(torch.stack((pose[..., 0, 3] / 10., pose[..., 1, 3] / 10.,
                                   torch.sin(yaw), torch.cos(yaw)), -1).to(media.dtype))
        camera = self.camera(torch.arange(3, device=media.device))
        group = temporal[:, :, None] + ego[:, :, None] + camera[None, None]
        coordinates = torch.where(visual.patch_valid[..., None], visual.xy, 0.)
        media = media + self.xy(coordinates.to(media.dtype)) + group[..., None, :]
        media = torch.where(visual.patch_valid[..., None], media, 0.).flatten(1, 3)
        queries = self.queries[None, None, None].to(media.dtype) + group[..., None, :]
        valid = visual.frame_valid[..., None].expand(-1, -1, -1, 16)
        queries = torch.where(valid[..., None], queries, 0.).flatten(1, 3)
        for block in self.blocks:
            queries = block(queries, media, valid.flatten(1), visual.patch_valid.flatten(1))
        result = self.output(queries).reshape(len(queries), 2, 3, 16, self.llm_dim)
        return HistoryMemory(torch.where(valid[..., None], result, 0.), valid).validate()


@dataclass(frozen=True)
class QwenHistoryBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    slots: dict
    image_embeds: torch.Tensor
    deepstack: tuple
    image_grid_thw: torch.Tensor
    memory_positions: Optional[torch.Tensor] = None
    memory: Optional[HistoryMemory] = None


def insert_history_slots(input_ids, attention_mask, ego_positions, valid, *, memory_token_id):
    """Insert ordinary positions without changing ANY existing token IDs/order."""
    if (input_ids.ndim != 2 or input_ids.dtype != torch.long or attention_mask.shape != input_ids.shape or
            ego_positions.shape != (len(input_ids), 1) or ego_positions.dtype != torch.long or
            valid.shape != (len(input_ids), 96) or valid.dtype != torch.bool):
        raise ValueError('Expected left-padded token batch, one ego slot and 96 history masks')
    if type(memory_token_id) is not int or memory_token_id < 0:
        raise ValueError('Explicit ordinary memory placeholder ID required')
    if any(x.device != input_ids.device for x in (attention_mask, ego_positions, valid)):
        raise ValueError('Token splice metadata device mismatch')
    if ((ego_positions < 0) | (ego_positions >= input_ids.shape[1])).any():
        raise ValueError('Ego position outside original sequence')
    ids, masks, positions = [], [], []
    for row, at in enumerate(ego_positions[:, 0].tolist()):
        if not attention_mask[row, at]:
            raise ValueError('Cannot insert history relative to a padded ego slot')
        placeholder = input_ids.new_full((96,), memory_token_id)
        ids.append(torch.cat((input_ids[row, :at], placeholder, input_ids[row, at:])))
        masks.append(torch.cat((attention_mask[row, :at], valid[row].to(attention_mask.dtype),
                                attention_mask[row, at:])))
        positions.append(torch.arange(at, at + 96, device=input_ids.device))
    return torch.stack(ids), torch.stack(masks), torch.stack(positions)


def _source_hash(method):
    return hashlib.sha256(inspect.getsource(method).encode()).hexdigest()


def _prompt(parent, instruction):
    dataset = parent.config.datasets.vla_data
    return dataset.get('CoT_prompt', '{instruction}').replace('{instruction}', instruction)


def _instruction(parent, observation):
    return (observation['lang'] + ' ' + parent.robot_history_token + ''.join(parent.foresight_tokens) +
            ''.join(parent.act_query_tokens))


def _history_batch(inputs):
    b = len(inputs)
    images = [[[None for _ in VIEWS] for _ in HISTORY_TIMES] for _ in range(b)]
    valid = torch.zeros(b, 2, 3, dtype=torch.bool)
    times = torch.tensor(HISTORY_TIMES, dtype=torch.float32).expand(b, -1).clone()
    ego = torch.eye(4).expand(b, 2, 4, 4).clone()
    identities = [None] * b
    present = [i for i, item in enumerate(inputs) if item.history is not None]
    if present:
        records = [inputs[i].history for i in present]
        if any(tuple(r['offsets_s']) != HISTORY_TIMES for r in records):
            raise ValueError('This adapter fixes real history at -1/-0.5 seconds')
        batch = collate_history_inputs(records, verify_images=True)
        for j, i in enumerate(present):
            images[i] = batch['history_images'][j]
            valid[i], times[i], ego[i] = batch['history_valid'][j], batch['history_times_s'][j], batch['history_ego_relative'][j]
            identities[i] = records[j]['identity']
    return images, valid, times, ego, tuple(identities)


class HistoryEncoder(nn.Module):
    def __init__(self, *, visual_dim, llm_dim, readout_dim, heads, memory_token_id,
                 initialization_seed, runtime_identity, mode='budget'):
        super().__init__()
        if mode not in ('budget', 'native_reference'):
            raise ValueError('Unknown registered history implementation')
        self.mode, self.memory_token_id = mode, memory_token_id
        self.visual_dim, self.llm_dim = visual_dim, llm_dim
        if mode == 'budget':
            self.resampler = HistoryResampler(visual_dim, llm_dim, dim=readout_dim, heads=heads)
        self.migration = dict(schema='st_history_adapter_v1', mode=mode,
            visual_dim=visual_dim, llm_dim=llm_dim, readout_dim=readout_dim, heads=heads,
            layers=2, queries_per_time_view=16, times_s=list(HISTORY_TIMES), view_names=list(VIEWS),
            memory_token_id=memory_token_id, initialization_seed=initialization_seed,
            insertion='ordinary slots immediately before original ego, W, action',
            latent_kv='all valid past visual groups and all valid past latent queries',
            ego_translation_scale_m=10., xy='actual final-merged cell centers in [0,1]',
            missing='masked zero; no substituted images; safe all-missing sentinel',
            history_deepstack='not injected into compressed memory; final merged only',
            current_deepstack='original complete streams on original image tokens only',
            numeric_policy='inherit parent visual dtype and caller autocast; resampler uses parent.amp',
            runtime_identity=runtime_identity)
        self._identity = hashlib.sha256(json.dumps(self.migration, sort_keys=True).encode()).hexdigest()
        # ZeRO FP32 reconstruction preserves named buffers but drops arbitrary
        # Module.extra_state. Byte values are exact even when reconstructed as
        # FP32 (or temporarily represented by BF16), unlike a large hash integer.
        self.register_buffer('_execution_contract', torch.tensor(list(bytes.fromhex(self._identity)), dtype=torch.uint8))
        self.last_diagnostics = {}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value = state_dict.get(prefix + '_execution_contract')
        if value is not None and (value.shape != self._execution_contract.shape or
                not torch.equal(value.detach().cpu().double(), self._execution_contract.cpu().double())):
            raise ValueError('History adapter execution/initialization contract changed on strict resume')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _validate_parent(self, parent):
        core = parent.qwen_vl_interface.model.model
        if any(p.requires_grad for p in core.visual.parameters()):
            raise ValueError('History caching/encoding requires the existing frozen visual, including mergers')
        core.visual.eval()
        if (core.visual.config.out_hidden_size != self.visual_dim or
                parent.qwen_vl_interface.model.config.hidden_size != self.llm_dim):
            raise ValueError('Bound visual/LLM dimensions changed')
        if parent.qwen_vl_interface.processor.tokenizer.padding_side != 'left':
            raise ValueError('Preserve original left-padding contract')
        return core

    @staticmethod
    def _visual(core, qwen_inputs):
        grid = qwen_inputs['image_grid_thw']
        merge = core.visual.spatial_merge_size
        if grid.ndim != 2 or grid.shape[1] != 3 or (grid[:, 0] != 1).any() or (grid <= 0).any():
            raise ValueError('Only actual still-image grids are accepted')
        if (grid[:, 1:] % merge != 0).any():
            raise ValueError('Actual image grid is not merge divisible')
        # Do not impose a new precision policy. This is the exact legacy
        # no_grad boundary and calls the same get_image_features implementation.
        with torch.no_grad():
            parts, deepstack = core.get_image_features(qwen_inputs['pixel_values'], grid)
        if len(parts) != len(grid):
            raise ValueError('Visual image/feature population mismatch')
        expected = (grid.prod(-1) // merge**2).tolist()
        if any(part.ndim != 2 or len(part) != count or not torch.isfinite(part).all()
               for part, count in zip(parts, expected)):
            raise ValueError('Final merged feature/grid mismatch or nonfinite visual output')
        if any(value.shape != (sum(expected), parts[0].shape[-1]) or not torch.isfinite(value).all()
               for value in deepstack):
            raise ValueError('DeepStack shape/finite contract changed')
        if len(deepstack) != len(core.visual.config.deepstack_visual_indexes):
            raise ValueError('A registered DeepStack stream was lost')
        return parts, tuple(deepstack)

    def _history_visual(self, parent, images, valid, times, ego, identities):
        core = parent.qwen_vl_interface.model.model
        device = parent.qwen_vl_interface.model.device
        addresses = valid.nonzero().tolist()
        parts, grids = [], []
        if addresses:
            selected = [images[b][t][v] for b, t, v in addresses]
            pixels = parent.qwen_vl_interface.processor.image_processor(images=selected, return_tensors='pt').to(device)
            parts, _ = self._visual(core, pixels)
            grids = pixels['image_grid_thw'].tolist()
        count = max((len(part) for part in parts), default=1)
        dtype = parts[0].dtype if parts else next(core.visual.parameters()).dtype
        tokens = torch.zeros(len(images), 2, 3, count, self.visual_dim, device=device, dtype=dtype)
        mask = torch.zeros(tokens.shape[:-1], device=device, dtype=torch.bool)
        xy = torch.zeros(*tokens.shape[:-1], 2, device=device, dtype=torch.float32)
        merge = core.visual.spatial_merge_size
        for (b, t, v), part, grid in zip(addresses, parts, grids):
            _, height, width = grid
            height, width = height // merge, width // merge
            yy, xx = torch.meshgrid((torch.arange(height, device=device) + .5) / height,
                                   (torch.arange(width, device=device) + .5) / width, indexing='ij')
            tokens[b, t, v, :len(part)] = part
            mask[b, t, v, :len(part)] = True
            xy[b, t, v, :len(part)] = torch.stack((xx, yy), -1).reshape(-1, 2)
        return HistoryVisualInput(tokens, mask, valid.to(device), xy, times.to(device), ego.to(device), identities).validate(), grids

    def prepare(self, parent, inputs):
        """Prepare only legal current/past observations, not labels or targets."""
        if not inputs or not all(isinstance(item, ModelInputs) for item in inputs):
            raise ValueError('History encoder accepts only validated ModelInputs records')
        core = self._validate_parent(parent)
        current = [item.legacy_current() for item in inputs]
        images, valid, times, ego, identities = _history_batch(inputs)
        interface = parent.qwen_vl_interface
        instructions = [_instruction(parent, item) for item in current]
        memory, memory_positions, history_grids = None, None, []
        if self.mode == 'budget':
            # The current image/prompt processor path is byte-for-byte the
            # existing public interface, before controlled ordinary-slot splice.
            packed = interface.build_qwenvl_inputs(images=[item['image'] for item in current], instructions=instructions)
            if len(packed['image_grid_thw']) != 3 * len(inputs):
                raise ValueError('Budget path must retain exactly three real current images per scene')
            parts, deepstack = self._visual(core, packed)
            history, history_grids = self._history_visual(parent, images, valid, times, ego, identities)
            with parent.amp():
                memory = self.resampler(history)
            ids, attention = packed['input_ids'], packed['attention_mask']
            ego_slots = parent._find_token_positions(ids, parent._special_token_ids['history'])
            ids, attention, memory_positions = insert_history_slots(ids, attention, ego_slots, memory.valid.flatten(1),
                                                                   memory_token_id=self.memory_token_id)
        else:
            messages = []
            for b, observation in enumerate(current):
                content = []
                for t in range(2):
                    if valid[b, t].any():
                        pose = ego[b, t]
                        yaw = torch.atan2(pose[1, 0], pose[0, 0]).item()
                        content.append({'type': 'text', 'text':
                            f'Observed ego at {times[b,t].item():.6f} seconds in current ego frame: '
                            f'x={pose[0,3].item():.6f} m, y={pose[1,3].item():.6f} m, yaw={yaw:.6f} rad.'})
                    for v, name in enumerate(VIEWS):
                        content.append({'type': 'text', 'text': f'Past camera {name}, physical time {times[b,t].item():.6f} seconds: ' +
                                        ('image.' if valid[b,t,v] else 'missing.')})
                        if valid[b,t,v]:
                            content.append({'type': 'image', 'image': images[b][t][v]})
                for name, image in zip(VIEWS, observation['image']):
                    content.extend([{'type': 'text', 'text': f'Current camera {name}, physical time 0 seconds.'},
                                    {'type': 'image', 'image': image}])
                content.append({'type': 'text', 'text': _prompt(parent, instructions[b])})
                messages.append([{'role': 'user', 'content': content}])
            packed = interface.processor.apply_chat_template(messages, tokenize=True, padding=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to(interface.model.device)
            if len(packed['image_grid_thw']) != int(valid.sum()) + 3 * len(inputs):
                raise ValueError('Native history image population does not match actual valid files')
            parts, deepstack = self._visual(core, packed)
            ids, attention = packed['input_ids'], packed['attention_mask']
        if 'labels' in packed or packed.get('pixel_values_videos') is not None:
            raise ValueError('History input may not contain labels or a different video layout')
        grid = packed['image_grid_thw']
        slots = {name: parent._find_token_positions(ids, token_ids) for name, token_ids in parent._special_token_ids.items()}
        image_mask = ids.eq(interface.model.config.image_token_id)
        total = int((grid.prod(-1) // core.visual.spatial_merge_size**2).sum())
        if int(image_mask.sum()) != total:
            raise ValueError('True image placeholder/grid count mismatch')
        if memory_positions is not None and image_mask.gather(1, memory_positions).any():
            raise ValueError('Compressed history was incorrectly marked as real image patches')
        with torch.no_grad():
            positions, _ = core.get_rope_index(ids, image_grid_thw=grid, attention_mask=attention)
        self.last_diagnostics = dict(mode=self.mode, scene_ids=[item.current['token'] for item in inputs],
            history_identities=identities, current_images=3 * len(inputs), history_images=int(valid.sum()),
            visual_calls=(1 + int(valid.any())) if self.mode == 'budget' else 1,
            language_calls=0, image_grid_thw=grid.detach().cpu().tolist(), history_grid_thw=history_grids,
            real_image_tokens=total, deepstack_streams=len(deepstack), history_slots=96 if memory is not None else 0,
            active_history_slots=int(memory.valid.sum()) if memory is not None else None,
            visual_output_dtype=str(parts[0].dtype), visual_parameter_dtype=str(next(core.visual.parameters()).dtype),
            sequence_lengths=attention.sum(-1).detach().cpu().tolist())
        return QwenHistoryBatch(ids, attention, positions, slots, torch.cat(parts, 0), deepstack, grid,
                                memory_positions, memory)

    def encode(self, parent, inputs):
        batch = self.prepare(parent, inputs)
        ids, slots = batch.input_ids, batch.slots
        parent._diagnostic_world_positions = slots.get('foresight')
        parent.last_sequence_lengths = batch.attention_mask.sum(-1).detach()
        rows = torch.arange(len(inputs), device=ids.device)
        with parent.amp():
            embeddings = parent.qwen_vl_interface.model.get_input_embeddings()(ids)
            states = torch.as_tensor(np.asarray([item.current['state'] for item in inputs]), device=ids.device, dtype=torch.float32)
            if states.shape != (len(inputs), 1, 4) or not torch.isfinite(states).all():
                raise ValueError('Original current ego representation changed')
            embeddings = replace_query_embeddings(embeddings, slots['history'], parent.action_input_model(states))
            if parent.foresight_tokens:
                if not (slots['history'][:, -1] < slots['foresight'][:, 0]).all() or not (slots['foresight'][:, -1] < slots['action'][:, 0]).all():
                    raise ValueError('Original ego/W/action causal order changed')
                embeddings = replace_query_embeddings(embeddings, slots['foresight'], parent.query_embeddings())
            if batch.memory is not None:
                if not (batch.memory_positions[:, -1] < slots['history'][:, 0]).all():
                    raise ValueError('History memory must precede current ego/W/action')
                embeddings = replace_query_embeddings(embeddings, batch.memory_positions, batch.memory.tokens.flatten(1, 3))
            hidden = parent._qwen_language_forward(ids, embeddings, batch.attention_mask, batch.position_ids,
                                                    batch.image_embeds, batch.deepstack)
        self.last_diagnostics['language_calls'] = 1
        return {'W': hidden[rows[:, None], slots['foresight']] if parent.foresight_tokens else hidden[:, :0],
                'action_queries': hidden[rows[:, None], slots['action']]}


def build_history_encoder(parent, *, initialization_seed, mode='budget'):
    """Fresh history parameters only. Parent/tokenizer/embedding rows are unchanged."""
    import transformers
    if not transformers.__version__.startswith('4.57.'):
        raise ValueError('This history adapter requires the reviewed Qwen 4.57 processor family')
    interface = parent.qwen_vl_interface
    core, tokenizer = interface.model.model, interface.processor.tokenizer
    ids = tokenizer.encode('history', add_special_tokens=False)
    forbidden = set(tokenizer.all_special_ids) | {interface.model.config.image_token_id, interface.model.config.video_token_id}
    forbidden.update(token for tokens in parent._special_token_ids.values() for token in tokens)
    if len(ids) != 1 or ids[0] in forbidden:
        raise ValueError('A verified ordinary single-token history placeholder is required')
    visual_dim, llm_dim = core.visual.config.out_hidden_size, interface.model.config.hidden_size
    dim = int(parent.foresight_config.readout_dim)
    # Read the actual existing head instead of silently guessing a head count.
    head = getattr(parent, 'dino_head', None)
    if head is None or not hasattr(head, 'blocks') or not len(head.blocks):
        raise ValueError('Build history encoder before removing the verified original DINO readout')
    heads = head.blocks[0].attn.num_heads
    runtime = dict(transformers=transformers.__version__, qwen_model_type=type(core).__module__,
        get_rope_index_sha256=_source_hash(core.get_rope_index),
        get_image_features_sha256=_source_hash(core.get_image_features),
        processor_type=type(interface.processor).__module__,
        image_processor_type=type(interface.processor.image_processor).__module__)
    with seeded_initialization(initialization_seed):
        result = HistoryEncoder(visual_dim=visual_dim, llm_dim=llm_dim, readout_dim=dim, heads=heads,
            memory_token_id=ids[0], initialization_seed=initialization_seed, runtime_identity=runtime, mode=mode)
        result.to(device=interface.model.device, dtype=next(parent.action_input_model.parameters()).dtype)
    result._validate_parent(parent)
    return result
