"""Current-only CPU preprocessing, scoped to one real optimizer batch.

No visual embeddings, trainable outputs, future labels or teacher inputs are
cached. The original processor packs each actual microbatch, including padding.
Pinned buffers remain owned by the payload until its CUDA consumers finish.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import torch
import transformers


@dataclass(frozen=True)
class PreparedCurrent:
    instructions: tuple
    image_identity: tuple
    processor_identity: str
    inputs: dict
    position_ids: torch.Tensor
    slots: dict
    geometry_images: torch.Tensor
    calibration: dict
    pixel_valid: torch.Tensor

    def validate(self, examples, instructions, processor_identity):
        if (tuple(instructions) != self.instructions or
            tuple(tuple(id(image) for image in row['image']) for row in examples) != self.image_identity or
            processor_identity != self.processor_identity):
            raise ValueError('Prepared current input belongs to another microbatch/processor')

    def to(self, device):
        def transfer(value):
            return value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        return PreparedCurrent(self.instructions, self.image_identity, self.processor_identity,
            {k: transfer(v) for k, v in self.inputs.items()}, transfer(self.position_ids),
            {k: transfer(v) for k, v in self.slots.items()}, transfer(self.geometry_images),
            {k: transfer(v) for k, v in self.calibration.items()}, transfer(self.pixel_valid))


def processor_identity(model):
    processor = model.qwen_vl_interface.processor
    record = {'transformers': transformers.__version__,
        'processor_class': type(processor).__module__+'.'+type(processor).__name__,
        'processor': processor.to_dict(), 'chat_template': processor.chat_template,
        'image_processor': processor.image_processor.to_dict(),
        'tokenizer_init_kwargs': processor.tokenizer.init_kwargs,
        'camera_count': model.cameras,
        'tokenizer_vocab': processor.tokenizer.get_vocab(),
        'padding_side': processor.tokenizer.padding_side,
        'CoT_prompt': model.config.datasets.vla_data.get('CoT_prompt', None),
        'camera_order': 'existing dataset observation order; unchanged ROI',
        'special_token_ids': model._special_token_ids}
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def pin(value):
    return value.pin_memory() if torch.is_tensor(value) else value


class CurrentBatchCollator:
    """Called by the existing single ordered CPU prefetch coordinator."""
    def __init__(self, model, microbatch, collate):
        self.model, self.microbatch, self.collate = model, microbatch, collate
        self.identity = processor_identity(model)

    def __call__(self, rows):
        from starVLA.model.framework.DDPForesight import ROBOT_HISTORY_TOKEN
        observations, targets = self.collate(rows)
        from starVLA.model.modules.structured_world.queries import prepare_query_candidates
        targets['query_candidates'] = [item.pin_memory() for item in
            prepare_query_candidates(targets, grid=self.model.grid)]
        if not torch.isfinite(targets['ego']).all() or not targets['future_valid'].all():
            raise ValueError('Invalid native ego targets in CPU collator')
        for at in range(0, len(observations), self.microbatch):
            micro = observations[at:at+self.microbatch]
            instructions = [row['lang']+' '+ROBOT_HISTORY_TOKEN+
                ''.join(self.model.foresight_tokens)+''.join(self.model.act_query_tokens) for row in micro]
            messages = []
            for row, instruction in zip(micro, instructions):
                template = self.model.config.datasets.vla_data.get('CoT_prompt', None)
                prompt = template.replace('{instruction}', instruction) if template is not None else instruction
                messages.append([{'role': 'user', 'content':
                    [{'type': 'image', 'image': image} for image in row['image']]+
                    [{'type': 'text', 'text': prompt}]}])
            inputs = self.model.qwen_vl_interface.processor.apply_chat_template(
                messages, tokenize=True, padding=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt')
            if any(torch.is_tensor(v) and v.device.type != 'cpu' for v in inputs.values()):
                raise ValueError('Worker preprocessing unexpectedly allocated GPU input')
            with torch.no_grad():
                positions, _ = self.model.qwen_vl_interface.model.model.get_rope_index(
                    input_ids=inputs['input_ids'], image_grid_thw=inputs['image_grid_thw'],
                    video_grid_thw=inputs.get('video_grid_thw'), attention_mask=inputs['attention_mask'])
                slots = {name: self.model._find_token_positions(inputs['input_ids'], ids)
                         for name, ids in self.model._special_token_ids.items()}
            states = np.asarray([row['state'] for row in micro])
            if states.shape != (len(micro), 1, 4) or not np.isfinite(states).all():
                raise ValueError('Invalid allowed current ego state')
            payload = PreparedCurrent(tuple(instructions),
                tuple(tuple(id(image) for image in row['image']) for row in micro), self.identity,
                {k: pin(v) for k, v in inputs.items()}, pin(positions),
                {k: pin(v) for k, v in slots.items()},
                pin(torch.stack([row['geometry_images'] for row in micro])),
                {k: pin(torch.stack([row['calibration'][k] for row in micro]))
                 for k in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')},
                pin(torch.stack([row['geometry_pixel_valid'] for row in micro])))
            for row in micro:
                row['_prepared_current'] = payload
        return observations, {k: pin(v) for k, v in targets.items()}


def transfer_optimizer_batch(observations, targets, device):
    """One transfer per batched tensor; microbatches retain views and ownership."""
    for at, row in enumerate(observations):
        cpu = row['_prepared_current']
        if at == 0 or cpu is not observations[at-1]['_prepared_current_cpu']:
            moved = cpu.to(device)
        row['_prepared_current_cpu'] = cpu
        row['_prepared_current'] = moved
    return observations, {k: ([item.to(device, non_blocking=True) for item in v] if k == 'query_candidates'
                             else v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                          for k, v in targets.items()}
