"""Only frozen inputs/compute define payloads; changing LLM/Adam does not."""
from pathlib import Path
import hashlib
import inspect
import json
import platform
import torch
from ..contracts import require
from ..io import digest, file_hash, json_bytes
from ..training.runtime_identity import backend_flags
from .storage import SCHEMA


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(digest({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)}).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint_visual_hash(state):
    prefix = 'qwen_vl_interface.model.model.visual.'
    values = {k[len(prefix):]: v for k, v in state['model'].items() if k.startswith(prefix)}
    require(values, 'checkpoint has no actual frozen visual state')
    return tensor_state_hash(values)


def visual_identity(model, contract):
    import transformers
    visual = model.qwen_vl_interface.model.model.visual
    require(not visual.training and all(not p.requires_grad for p in visual.parameters()), 'visual not frozen eval')
    require(all(v.dtype == torch.float32 for v in visual.state_dict().values() if v.is_floating_point()),
            'visual parameters/buffers must be FP32')
    require(not torch.is_autocast_enabled(), 'identity must be built outside autocast')
    # A stochastic transform cannot be made cacheable by silently disabling it.
    require(contract['input']['image_transform'] == 'original center 16:9 crop; Lanczos1024x576; original Qwen processor',
            'unknown augmentation/transform: explicit new identity and audit needed')
    processor = model.qwen_vl_interface.processor
    tokenizer = processor.tokenizer
    value = {'schema': SCHEMA, 'visual_state_sha256': tensor_state_hash(visual.state_dict()),
        'visual_numeric_contract': 'legacy_fp32', 'compute_dtype': 'float32', 'storage_dtype': 'float32',
        'backend_flags': backend_flags(), 'visual_attention': visual.config._attn_implementation,
        'visual_config': visual.config.to_dict(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'transformers': transformers.__version__, 'python': platform.python_version(),
        'processor_source_hash': file_hash(inspect.getfile(type(processor))),
        'visual_source_hash': file_hash(inspect.getfile(type(visual))),
        'processor_config': processor.image_processor.to_dict(),
        'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()), 'special_token_ids': model._special_token_ids,
        'padding_side': tokenizer.padding_side, 'pad_token_id': tokenizer.pad_token_id,
        'image_token_id': model.qwen_vl_interface.model.config.image_token_id,
        'framework_source_hashes': contract['source_hashes'], 'prompt_hash': contract['prompt_hash'],
        'transform_hash': contract['transform_hash'], 'camera_time_hash': contract['camera_time_hash'],
        'views': contract['input']['views'], 'augmentation_identity': 'locked_no_augmentation',
        'deepstack_count': 3, 'normalizer_hash': contract['normalizer_hash'],
        'cache_boundary_source_hash': file_hash(Path(__file__).with_name('inputs.py')),
        'payload_schema_source_hash': file_hash(Path(__file__).with_name('storage.py'))}
    return json.loads(json_bytes(value))


def payload_key(image_hashes, instruction, identity):
    require(len(image_hashes) == 3 and all(len(x) == 64 for x in image_hashes), 'three real content hashes required')
    return digest({'image_sha256_ordered': image_hashes, 'instruction': instruction, 'identity': identity})
