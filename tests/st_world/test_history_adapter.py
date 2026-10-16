"""Behavior tests use an explicitly tiny fixture backbone, never real-S0 evidence."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import json
import os
import re

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from transformers import BatchFeature
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as VendoredQwen3VLModel

from starVLA.dataloader.st_world_schema import ModelInputs, identity_hash
from starVLA.dataloader.st_world_history import VIEWS, image_reference
from starVLA.model.modules.foresight.dino_feature_head import DINOFeatureHead
from starVLA.model.modules.foresight.tokens import replace_query_embeddings
from starVLA.model.modules.st_world.history_adapter import (
    HistoryVisualInput, HistoryResampler, build_history_encoder, insert_history_slots)

ROBOT_HISTORY_TOKEN = '<robot_history_action_0>'  # Tiny fixture's original ego placeholder.


class FixtureTokenizer:
    padding_side, pad_token_id = 'left', 0
    all_special_ids = [0, 101, 102, 103, 104, 12, 13, 14, 21, 22]
    names = {ROBOT_HISTORY_TOKEN: 12, '<foresight_0>': 13, '<foresight_1>': 14,
             '<action_0>': 21, '<action_1>': 22}

    def encode(self, text, add_special_tokens=False):
        if text == 'history':
            return [5]
        pieces = re.split('(' + '|'.join(re.escape(k) for k in self.names) + ')', text)
        result = []
        for piece in pieces:
            result.extend([self.names[piece]] if piece in self.names else [7] * len(piece.split()))
        return result


class FixtureProcessor:
    def __init__(self):
        self.tokenizer = FixtureTokenizer()
        self.image_processor = self
        self.messages = None

    def __call__(self, images, return_tensors='pt'):
        values = [np.asarray(im).astype('float32').mean() / 255. for im in images]
        return BatchFeature({'pixel_values': torch.tensor(values).repeat_interleave(16)[:, None],
                             'image_grid_thw': torch.tensor([[1, 4, 4]] * len(images))})

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        rows, images = [], []
        for conversation in messages:
            ids = [8]
            for part in conversation[0]['content']:
                if part['type'] == 'image':
                    images.append(part['image'])
                    ids.extend([101, 102, 102, 102, 102, 103])
                else:
                    ids.extend(self.tokenizer.encode(part['text']))
            rows.append(ids + [9])
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        packed = self(images)
        packed['input_ids'] = torch.tensor(padded)
        packed['attention_mask'] = packed['input_ids'].ne(0).long()
        return packed


class FixtureVisual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(1, width)
        self.norm = nn.BatchNorm1d(width)
        self.config = SimpleNamespace(out_hidden_size=width, deepstack_visual_indexes=[0, 1, 2])
        self.spatial_merge_size = 2
        self.calls = 0

    def forward(self, pixel_values, grid_thw):
        self.calls += 1
        outputs, at = [], 0
        for grid in grid_thw:
            n = int(grid.prod())
            value = pixel_values[at:at+n].float().mean()
            at += n
            slots = torch.arange(n // 4, device=value.device).float()[:, None] * .01 + value
            outputs.append(self.norm(self.projection(slots)))
        result = torch.cat(outputs)
        return result, [result * scale for scale in (.1, .2, .3)]


class FixtureLanguage(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.calls = 0

    def forward(self, *, inputs_embeds, attention_mask, position_ids, visual_pos_masks,
                deepstack_visual_embeds, **kwargs):
        self.calls += 1
        value = inputs_embeds.clone()
        for stack in deepstack_visual_embeds:
            addition = torch.zeros_like(value).masked_scatter(visual_pos_masks[..., None], stack)
            value = value + addition
        valid = attention_mask[..., None].to(value.dtype)
        prefix = (value * valid).cumsum(1) / valid.cumsum(1).clamp_min(1)
        position = position_ids.float().sum(0)[..., None] * .001
        return SimpleNamespace(last_hidden_state=self.linear(prefix + position).tanh())


class FixtureCore(nn.Module):
    rope_model_class = Qwen3VLModel
    def __init__(self, width, image_id=102, video_id=104, vision_start=101):
        super().__init__()
        self.visual = FixtureVisual(width).requires_grad_(False)
        self.language_model = FixtureLanguage(width)
        self.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=image_id, video_token_id=video_id, vision_start_token_id=vision_start)

    def get_rope_index(self, *args, **kwargs):
        return self.rope_model_class.get_rope_index(self, *args, **kwargs)

    def get_image_features(self, pixel_values, image_grid_thw):
        values, deepstack = self.visual(pixel_values, image_grid_thw)
        return torch.split(values, (image_grid_thw.prod(-1) // 4).tolist()), deepstack


class FixtureVLM(nn.Module):
    def __init__(self, width, processor):
        super().__init__()
        if isinstance(processor, FixtureProcessor):
            vocabulary, image, video, start = 200, 102, 104, 101
        else:
            vocabulary = len(processor.tokenizer)
            image, video = processor.image_token_id, processor.video_token_id
            start = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.model = FixtureCore(width, image, video, start)
        self.embedding = nn.Embedding(vocabulary, width)
        self.config = SimpleNamespace(hidden_size=width, image_token_id=image, video_token_id=video)

    @property
    def device(self): return self.embedding.weight.device

    def get_input_embeddings(self): return self.embedding


class FixtureInterface(nn.Module):
    def __init__(self, width, processor):
        super().__init__()
        self.processor = processor
        self.model = FixtureVLM(width, processor)

    def build_qwenvl_inputs(self, images, instructions):
        messages = [[{'role': 'user', 'content': [{'type': 'image', 'image': im} for im in ims] +
            [{'type': 'text', 'text': instruction}]}] for ims, instruction in zip(images, instructions)]
        return self.processor.apply_chat_template(messages, tokenize=True, padding=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.model.device)


class FixtureParent(nn.Module):
    def __init__(self, processor=None):
        super().__init__()
        processor = processor or FixtureProcessor()
        self.qwen_vl_interface = FixtureInterface(16, processor)
        self.robot_history_token = ROBOT_HISTORY_TOKEN
        self.foresight_tokens, self.act_query_tokens = ['<foresight_0>', '<foresight_1>'], ['<action_0>', '<action_1>']
        tokenizer = processor.tokenizer
        self._special_token_ids = {name: tuple(tokenizer.encode(text, add_special_tokens=False)[0] for text in texts)
            for name, texts in {'history': [ROBOT_HISTORY_TOKEN], 'foresight': self.foresight_tokens,
                                'action': self.act_query_tokens}.items()}
        self.foresight_queries = nn.Parameter(torch.randn(2, 16))
        self.action_input_model = nn.Linear(4, 16)
        self.foresight_config = SimpleNamespace(readout_dim=16, planner_condition_mode='action_only')
        self.dino_head = DINOFeatureHead(16, 8, dim=16, heads=4)
        self.config = SimpleNamespace(datasets=SimpleNamespace(vla_data={}),
            framework=SimpleNamespace(action_model=SimpleNamespace(action_dim=4, action_horizon=8,
                repeated_diffusion_steps=8, num_inference_timesteps=10)))
        self.qwen_vl_interface.model.model.visual.eval()

    def amp(self): return nullcontext()
    def query_embeddings(self): return self.foresight_queries

    @staticmethod
    def _find_token_positions(ids, token_ids):
        result = []
        for row in ids:
            positions = []
            for token in token_ids:
                where = (row == token).nonzero().flatten()
                if len(where) != 1: raise ValueError('Ambiguous driving slot')
                positions.append(where[0])
            result.append(torch.stack(positions))
        return torch.stack(result)

    def _qwen_language_forward(self, ids, embeddings, attention, positions, visual, deepstack):
        image_mask = ids.eq(self.qwen_vl_interface.model.config.image_token_id)
        embeddings = embeddings.masked_scatter(image_mask[..., None], visual.to(embeddings.dtype))
        return self.qwen_vl_interface.model.model.language_model(inputs_embeds=embeddings,
            attention_mask=attention, position_ids=positions, visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack).last_hidden_state

    def encode_current(self, observations):
        instructions = [item['lang'] + ' ' + ROBOT_HISTORY_TOKEN + ''.join(self.foresight_tokens) +
                        ''.join(self.act_query_tokens) for item in observations]
        interface = self.qwen_vl_interface
        packed = interface.build_qwenvl_inputs([item['image'] for item in observations], instructions)
        ids, attention = packed['input_ids'], packed['attention_mask']
        with torch.no_grad():
            positions, _ = interface.model.model.get_rope_index(ids, packed['image_grid_thw'], attention_mask=attention)
            parts, deepstack = interface.model.model.get_image_features(packed['pixel_values'], packed['image_grid_thw'])
        slots = {name: self._find_token_positions(ids, values) for name, values in self._special_token_ids.items()}
        embeddings = interface.model.get_input_embeddings()(ids)
        states = torch.tensor(np.asarray([item['state'] for item in observations]))
        embeddings = replace_query_embeddings(embeddings, slots['history'], self.action_input_model(states))
        embeddings = replace_query_embeddings(embeddings, slots['foresight'], self.query_embeddings())
        hidden = self._qwen_language_forward(ids, embeddings, attention, positions, torch.cat(parts), deepstack)
        row = torch.arange(len(ids))[:, None]
        return {'W': hidden[row, slots['foresight']], 'action_queries': hidden[row, slots['action']]}


def observation(token='scene', long=False):
    return {'token': token, 'lang': 'Turn left at the next junction.' if long else 'Go.',
            'state': np.array([[.1, .2, 0., 1.]], dtype=np.float32),
            'image': [Image.new('RGB', (64, 32), (20 + i * 30, 40, 80)) for i in range(3)]}


def record(tmp_path, token='scene', missing=()):
    frames = []
    for t, seconds in enumerate((-1., -.5)):
        views = []
        for v, camera in enumerate(VIEWS):
            path = tmp_path / f'{token}_{t}_{v}.png'
            Image.new('RGB', (64, 36), (30 + t * 20, 20 + v * 40, 70)).save(path)
            valid = (t, v) not in missing
            views.append(dict(camera=camera, valid=valid, image=image_reference(path) if valid else None,
                intrinsics=np.eye(3).tolist() if valid else None, distortion=[] if valid else None,
                camera_to_current_planar_ego=np.eye(4).tolist() if valid else None,
                crop_xyxy=[0, 0, 64, 36] if valid else None,
                resize_affine=np.eye(3).tolist() if valid else None, sensor_timestamp_us=None))
        pose = np.eye(4); pose[0, 3] = seconds * 2
        frames.append(dict(requested_s=seconds, token=f'{token}_past{t}', timestamp_us=int(2000000 + seconds * 1e6),
            actual_offset_s=seconds, relative_ego_planar=pose.tolist(), views=views))
    result = dict(schema='st_causal_history_v1', token=token, log='fixture', current_timestamp_us=2000000,
        view_names=list(VIEWS), offsets_s=[-1., -.5], timestamp_tolerance_s=.05,
        source_raw_sha256='a' * 64, current_record_identity='b' * 64, frames=frames,
        preprocess='original center16:9 then1024x576 LANCZOS',
        missing_policy='masked_absent_no_current_or_future_substitution')
    result['identity'] = identity_hash(result)
    return result


def visual_input(all_missing=False):
    tokens = torch.randn(2, 2, 3, 6, 8)
    mask = torch.ones(2, 2, 3, 6, dtype=torch.bool)
    if all_missing: mask.zero_()
    else: mask[1, 0, 1] = False
    pose = torch.eye(4).expand(2, 2, 4, 4).clone()
    pose[:, 0, 0, 3] = -2
    return HistoryVisualInput(tokens, mask, mask.any(-1), torch.rand(2, 2, 3, 6, 2),
        torch.tensor([[-1., -.5], [-1.01, -.49]]), pose, ('a', 'b'))


def test_resampler_masking_allmissing_and_all_parameter_gradients():
    torch.manual_seed(1)
    model = HistoryResampler(8, 12, dim=16, heads=4)
    output = model(visual_input())
    assert output.tokens.shape == (2, 2, 3, 16, 12)
    assert torch.equal(output.tokens[~output.valid], torch.zeros_like(output.tokens[~output.valid]))
    output.tokens.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.norm() > 0 for p in model.parameters())
    model.zero_grad(set_to_none=True)
    empty = model(visual_input(all_missing=True))
    assert not empty.valid.any() and empty.tokens.count_nonzero() == 0
    empty.tokens.sum().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in model.parameters())


def test_history_visual_refuses_future_and_trainable_or_misaligned_features():
    valid = visual_input()
    for field, value in [('tokens', valid.tokens.requires_grad_()),
                         ('actual_dt_s', torch.tensor([[1., 2.], [1., 2.]])),
                         ('frame_valid', ~valid.frame_valid)]:
        fields = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
        fields['tokens'] = fields['tokens'].detach()
        fields[field] = value
        with pytest.raises(ValueError): HistoryVisualInput(**fields).validate()


def test_all_missing_and_left_padding_retain_current_function_and_real_image_layout():
    torch.manual_seed(2)
    parent = FixtureParent().eval()
    inputs = [ModelInputs(observation()), ModelInputs(observation('long', long=True))]
    before = {k: t.clone() for k, t in parent.state_dict().items()}
    expected = parent.encode_current([x.current for x in inputs])
    rng = torch.get_rng_state().clone()
    encoder = build_history_encoder(parent, initialization_seed=37)
    assert torch.equal(rng, torch.get_rng_state())
    actual = encoder.encode(parent, inputs)
    for key in actual: torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)
    batch = encoder.prepare(parent, inputs)
    assert len(batch.image_grid_thw) == 6 and len(batch.deepstack) == 3
    assert batch.input_ids.eq(102).sum() == 24
    assert not batch.attention_mask.gather(1, batch.memory_positions).any()
    assert not batch.input_ids.gather(1, batch.memory_positions).eq(102).any()
    assert encoder.last_diagnostics['history_images'] == 0 and encoder.last_diagnostics['visual_calls'] == 1
    for name, value in before.items(): assert torch.equal(parent.state_dict()[name], value), name


def test_current_visual_frozen_history_and_language_gradients_once(tmp_path):
    torch.manual_seed(3)
    parent = FixtureParent()
    encoder = build_history_encoder(parent, initialization_seed=37)
    parent.train()  # Outer training must not update visual buffers.
    visual = parent.qwen_vl_interface.model.model.visual
    before = {name: value.clone() for name, value in visual.state_dict().items()}
    inputs = [ModelInputs(observation(), record(tmp_path))]
    result = encoder.encode(parent, inputs)
    sum(value.square().mean() for value in result.values()).backward()
    assert visual.calls == 2 and not visual.training
    assert parent.qwen_vl_interface.model.model.language_model.calls == 1
    assert all(parameter.grad is None for parameter in visual.parameters())
    for name, value in before.items(): assert torch.equal(visual.state_dict()[name], value), name
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.norm() > 0 for p in encoder.parameters())
    assert parent.action_input_model.weight.grad.norm() > 0
    assert parent.foresight_queries.grad.norm() > 0
    assert parent.qwen_vl_interface.model.embedding.weight.grad.norm() > 0
    assert parent.qwen_vl_interface.model.model.language_model.linear.weight.grad.norm() > 0
    assert encoder.last_diagnostics['active_history_slots'] == 96


def test_partial_missing_camera_and_actual_native_grid_population(tmp_path):
    parent = FixtureParent().eval()
    inputs = [ModelInputs(observation(), record(tmp_path, missing=((0, 1), (1, 2))))]
    budget = build_history_encoder(parent, initialization_seed=5)
    packed = budget.prepare(parent, inputs)
    assert packed.memory.valid.sum() == 64 and len(packed.image_grid_thw) == 3
    native = build_history_encoder(parent, initialization_seed=5, mode='native_reference')
    packed_native = native.prepare(parent, inputs)
    assert packed_native.memory is None and len(packed_native.image_grid_thw) == 7
    assert packed_native.input_ids.eq(102).sum() == 7 * 4
    messages = parent.qwen_vl_interface.processor.messages
    texts = ' '.join(part['text'] for part in messages[0][0]['content'] if part['type'] == 'text')
    assert '-1.000000' in texts and '-0.500000' in texts and 'missing' in texts and 'yaw=' in texts


def test_source_label_whitelist_and_history_identity_reject_future_injection(tmp_path):
    bad = observation()
    bad['gt_trajectory'] = torch.ones(8, 3)
    with pytest.raises(ValueError): ModelInputs(bad)
    history = record(tmp_path)
    history['frames'][0]['actual_offset_s'] = .5
    history['frames'][0]['timestamp_us'] = 2500000
    history['identity'] = identity_hash({k: v for k, v in history.items() if k != 'identity'})
    with pytest.raises(ValueError, match='Future/current'):
        ModelInputs(observation(), history)
    encoder = build_history_encoder(FixtureParent(), initialization_seed=4)
    with pytest.raises(ValueError, match='validated ModelInputs'):
        encoder.encode(FixtureParent(), [dict(observation(), targets=torch.ones(1))])


def test_checkpoint_strict_restore_and_deployment_does_not_drop_history(tmp_path):
    parent = FixtureParent().eval()
    inputs = [ModelInputs(observation(), record(tmp_path))]
    first = build_history_encoder(parent, initialization_seed=42)
    second = build_history_encoder(parent, initialization_seed=42)
    second.load_state_dict(first.state_dict(), strict=True)
    # Match ZeRO's real named-buffer FP32 reconstruction, without discarding
    # identity as happens for arbitrary Module.extra_state.
    reconstructed = {name: value.float() for name, value in first.state_dict().items()}
    second.load_state_dict(reconstructed, strict=True)
    for key, value in first.encode(parent, inputs).items():
        torch.testing.assert_close(value, second.encode(parent, inputs)[key], rtol=0, atol=0)
    wrong = build_history_encoder(parent, initialization_seed=43)
    with pytest.raises(ValueError, match='contract changed'):
        wrong.load_state_dict(first.state_dict(), strict=True)
    from starVLA.model.framework.st_world_model import SpaceTimeStudent
    wrapper = SpaceTimeStudent.__new__(SpaceTimeStudent)
    nn.Module.__init__(wrapper)
    wrapper.history_encoder, wrapper.sparse_readout = first, nn.Linear(1, 1)
    wrapper.strip_auxiliary_heads()
    assert wrapper.history_encoder is first and not hasattr(wrapper, 'sparse_readout')


def test_parent_visual_must_already_be_frozen():
    parent = FixtureParent()
    parent.qwen_vl_interface.model.model.visual.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen visual'):
        build_history_encoder(parent, initialization_seed=42)


@pytest.mark.parametrize('implementation', ['installed', 'vendored'])
def test_actual_qwen_language_attention_allmissing_and_history_backward(tmp_path, implementation):
    if implementation == 'installed':
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    else:
        from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    torch.manual_seed(81)
    parent = FixtureParent().eval()
    parent.qwen_vl_interface.model.model.rope_model_class = (
        Qwen3VLModel if implementation == 'installed' else VendoredQwen3VLModel)
    config = Qwen3VLTextConfig(vocab_size=200, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=4096, pad_token_id=0, use_cache=False,
        rope_scaling={'rope_type': 'default', 'mrope_section': [2, 1, 1]})
    config._attn_implementation = 'sdpa'
    language = Qwen3VLTextModel(config).eval()
    parent.qwen_vl_interface.model.model.language_model = language
    encoder = build_history_encoder(parent, initialization_seed=81)
    missing = [ModelInputs(observation()), ModelInputs(observation('tail', long=True))]
    old = parent.encode_current([item.current for item in missing])
    new = encoder.encode(parent, missing)
    for key in old:
        torch.testing.assert_close(new[key], old[key], rtol=1e-5, atol=1e-6)
    real_history = [ModelInputs(observation(), record(tmp_path))]
    calls = []
    hook = language.register_forward_hook(lambda module, inputs, output: calls.append(1))
    encoded = encoder.encode(parent, real_history)
    (encoded['W'][..., 0].sum() + encoded['action_queries'][..., 1].sum()).backward()
    hook.remove()
    assert calls == [1]
    assert all(p.grad is not None and p.grad.isfinite().all() for p in encoder.parameters())
    assert language.layers[0].self_attn.q_proj.weight.grad.norm() > 0
    assert all(p.grad is None for p in parent.qwen_vl_interface.model.model.visual.parameters())


def test_mixed_history_tail_batch_absence_stays_zero_and_labels_are_not_inputs(tmp_path):
    parent = FixtureParent().eval()
    encoder = build_history_encoder(parent, initialization_seed=3)
    absent = ModelInputs(observation('absent'))
    present = ModelInputs(observation('present', long=True), record(tmp_path, 'present'))
    expected_absent = parent.encode_current([absent.current])
    first = encoder.encode(parent, [absent, present])
    for key in first:
        torch.testing.assert_close(first[key][0:1], expected_absent[key], rtol=1e-6, atol=1e-7)
    # Targets are a separate caller object. Arbitrary replacement neither
    # supplies a history slot nor enters either visual/LLM call signature.
    targets = {'future_gt': torch.randn(2, 8, 4), 'score': torch.randn(2)}
    targets.clear()
    second = encoder.encode(parent, [absent, present])
    for key in first: torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    tail = encoder.encode(parent, [present])
    for key in first: torch.testing.assert_close(first[key][1:2], tail[key], rtol=1e-6, atol=1e-7)


@pytest.fixture(scope='module')
def actual_qwen_processor():
    path = os.environ.get('ST_QWEN_PROCESSOR')
    if not path:
        pytest.skip('Set ST_QWEN_PROCESSOR to verified local Qwen assets for actual processor/MRoPE tests')
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    names = list(FixtureTokenizer.names)
    missing = [name for name in names if name not in processor.tokenizer.get_vocab()]
    if missing:
        processor.tokenizer.add_special_tokens({'additional_special_tokens': missing}, replace_additional_special_tokens=False)
    return processor


@pytest.mark.parametrize('rope_model', [Qwen3VLModel, VendoredQwen3VLModel], ids=['installed', 'vendored'])
def test_actual_qwen_processor_nine_images_preserves_real_layout_and_missing_masks(tmp_path, actual_qwen_processor, rope_model):
    # Actual installed Qwen image processor + get_rope_index, fixture visual and
    # language weights only. This is explicitly not a real S0-forward claim.
    parent = FixtureParent(actual_qwen_processor).eval()
    parent.qwen_vl_interface.model.model.rope_model_class = rope_model
    inputs = [ModelInputs(observation(), record(tmp_path))]
    budget = build_history_encoder(parent, initialization_seed=5)
    native = build_history_encoder(parent, initialization_seed=5, mode='native_reference')
    packed = budget.prepare(parent, inputs)
    reference = native.prepare(parent, inputs)
    assert len(packed.image_grid_thw) == 3 and len(reference.image_grid_thw) == 9
    assert torch.equal(packed.image_grid_thw, reference.image_grid_thw[-3:])
    assert packed.memory.tokens.shape == (1, 2, 3, 16, 16)
    assert len(reference.deepstack) == len(packed.deepstack) == 3
    for batch in (packed, reference):
        actual = int(batch.input_ids.eq(actual_qwen_processor.image_token_id).sum())
        assert actual == int((batch.image_grid_thw.prod(-1) // 4).sum())
        assert batch.position_ids.shape == (3, *batch.input_ids.shape)
    assert not packed.input_ids.gather(1, packed.memory_positions).eq(actual_qwen_processor.image_token_id).any()


@pytest.mark.parametrize('rope_model', [Qwen3VLModel, VendoredQwen3VLModel], ids=['installed', 'vendored'])
def test_actual_qwen_left_padding_allmissing_active_mrope_matches_current(actual_qwen_processor, rope_model):
    parent = FixtureParent(actual_qwen_processor).eval()
    parent.qwen_vl_interface.model.model.rope_model_class = rope_model
    inputs = [ModelInputs(observation('a')), ModelInputs(observation('b', long=True))]
    instructions = [item.current['lang'] + ' ' + parent.robot_history_token + ''.join(parent.foresight_tokens) +
                    ''.join(parent.act_query_tokens) for item in inputs]
    original = parent.qwen_vl_interface.build_qwenvl_inputs([item.current['image'] for item in inputs], instructions)
    positions, _ = parent.qwen_vl_interface.model.model.get_rope_index(original['input_ids'], original['image_grid_thw'],
        attention_mask=original['attention_mask'])
    encoder = build_history_encoder(parent, initialization_seed=5)
    batch = encoder.prepare(parent, inputs)
    for i in range(2):
        active_original, active_new = original['attention_mask'][i].bool(), batch.attention_mask[i].bool()
        assert torch.equal(batch.input_ids[i, active_new], original['input_ids'][i, active_original])
        assert torch.equal(batch.position_ids[:, i, active_new], positions[:, i, active_original])
    assert not batch.memory.valid.any()


def test_real_registered_images_processor_layout_only(actual_qwen_processor):
    """Real current/past pixels and metadata, NOT real pretrained S0 weights."""
    history_root, current_root = os.environ.get('ST_HISTORY_AUDIT_ROOT'), os.environ.get('ST_CURRENT_DATA_ROOT')
    if not history_root or not current_root:
        pytest.skip('Set audited real train history/current roots for CPU pixel/layout integration')
    from starVLA.dataloader.st_world_history import HistoryReader, file_sha256
    from starVLA.dataloader.foresight_dataset import current_observation
    manifest = json.loads((Path(history_root) / 'HISTORY_MANIFEST.json').read_text())
    reader = HistoryReader(history_root, expected_identity=manifest['identity'])
    assert manifest['split'] == 'train'
    rows = manifest['scenes'][:2]  # Fixed manifest order, never chosen by model results.
    inputs = []
    for row in rows:
        path = Path(current_root) / 'current' / (row['token'] + '.json')
        assert file_sha256(path) == row['current_record_sha256']
        current = json.loads(path.read_text())
        history = reader.read(row['token'], current_record=current)
        inputs.append(ModelInputs(current_observation(current), history))
    parent = FixtureParent(actual_qwen_processor).eval()
    parent.qwen_vl_interface.model.model.rope_model_class = VendoredQwen3VLModel
    budget = build_history_encoder(parent, initialization_seed=5)
    native = build_history_encoder(parent, initialization_seed=5, mode='native_reference')
    packed = budget.prepare(parent, inputs)
    reference = native.prepare(parent, inputs)
    assert len(packed.image_grid_thw) == 3 * len(inputs)
    assert budget.last_diagnostics['history_images'] == 6 * len(inputs)
    assert len(reference.image_grid_thw) == 9 * len(inputs)
    assert packed.memory.valid.sum() == 96 * len(inputs)
    assert len(packed.deepstack) == len(reference.deepstack) == 3
    for batch in (packed, reference):
        assert batch.position_ids.shape == (3, *batch.input_ids.shape)
        assert int(batch.input_ids.eq(actual_qwen_processor.image_token_id).sum()) == int(
            (batch.image_grid_thw.prod(-1) // 4).sum())
        assert (batch.slots['history'][:, -1] < batch.slots['foresight'][:, 0]).all()
        assert (batch.slots['foresight'][:, -1] < batch.slots['action'][:, 0]).all()
    assert not packed.input_ids.gather(1, packed.memory_positions).eq(actual_qwen_processor.image_token_id).any()
