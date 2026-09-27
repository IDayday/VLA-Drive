"""Diagnose trained BF16 native-prefix invariance; no labels or policy selection."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from tools.local_interaction_mask_v2.extract_current import load_foundation
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle, current_example
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.local_planner import predict_local


def full_precision(self, x):
    with torch.autocast('cuda', enabled=False):
        return self.up(self.down(x.float())) * self.scale


def fixed_chunks(self, x):
    shape = x.shape; values = x.reshape(-1, shape[-1]); count = len(values)
    values = F.pad(values, (0, 0, 0, (-count) % 512))
    result = torch.cat([self.up(self.down(block)) * self.scale for block in values.split(512)])
    return result[:count].reshape(*shape[:-1], -1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('foundation', 'public-qwen', 'dataset', 'visual-cache', 'output'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    world, metadata = load_foundation(args.foundation, args.public_qwen, args.visual_cache)
    spec = json.loads(Path(args.dataset).read_text()); token = json.loads(Path(spec['tokens']).read_text())[0]
    record = current_metadata_from_training_pickle(Path(spec['meta_root']) / (token + '.pkl'), token)
    example, observation = current_example(Path(spec['observations']) / (token + '.npz'), record)
    lm = world.baseline.qwen_vl_interface.model.model.language_model
    original = {name: module.forward for name, module in world.baseline.qwen_adapters.items()}
    rows = []; reference = None
    for mode in ('ordinary', 'fp32_adapter', 'chunked_adapter'):
        for name, module in world.baseline.qwen_adapters.items():
            module.forward = original[name] if mode == 'ordinary' else MethodType(full_precision if mode == 'fp32_adapter' else fixed_chunks, module)
        captures = {}; sequence = {}; tag = 'native'
        def capture_input(module, arguments, kwargs):
            values = kwargs['inputs_embeds']; sequence[tag] = values.detach().cpu()
        handles = [lm.register_forward_pre_hook(capture_input, with_kwargs=True)]
        def layer_hook(index):
            def hook(module, arguments):
                native_length = sequence['native'].shape[1]
                captures[(tag, index)] = arguments[0][:, native_length-16:native_length].detach().float().cpu()
            return hook
        handles += [layer.register_forward_pre_hook(layer_hook(i)) for i, layer in enumerate(lm.layers)]
        native = world.baseline.native_conditions([example]); tag = 'append'
        with torch.autocast('cuda', dtype=torch.bfloat16): appended, _ = world.encode_conditions([example])
        for handle in handles: handle.remove()
        # Causal isolation under the SAME sequence shape is independent of kernel rounding across shapes.
        def erase_tail(module, arguments, memory):
            return replace(memory, scene_memory=memory.scene_memory * 0., agent_memory=memory.agent_memory * 0.)
        handle = world.reader.register_forward_hook(erase_tail)
        with torch.autocast('cuda', dtype=torch.bfloat16): erased, _ = world.encode_conditions([example])
        handle.remove()
        first = predict_local(world.baseline.action_model, None, native, None, [token])['normalized_actions']
        second = predict_local(world.baseline.action_model, None, appended, None, [token])['normalized_actions']
        if reference is None: reference = appended.clone()
        delta = (native.float()-appended.float())
        row = {'mode': mode, 'token': token, 'native_tokens': native.shape[1], 'prefix_sequence_length': sequence['native'].shape[1],
            'prefix_embedding_max_error': float((sequence['native']-sequence['append'][:, :sequence['native'].shape[1]]).abs().max()),
            'condition_max_error': float(delta.abs().max()), 'condition_relative_l2': float(delta.norm()/native.float().norm()),
            'condition_magnitude': float(native.abs().max()), 'tail_erased_same_shape_error': float((appended-erased).abs().max()),
            'normalized_action_max_error': float((first-second).abs().max()),
            'xy_max_distance_m': float(((first[...,:2]-second[...,:2])*first.new_tensor([8.805105,2.277741])).norm(dim=-1).max()),
            'changed_adapter_vs_original_append_max_error': float((reference-appended).abs().max()),
            'layer_last16_prefix_max_error': [float((captures[('native', i)]-captures[('append', i)]).abs().max()) for i in range(len(lm.layers))]}
        rows.append(row); atomic_json(output / 'result.json', {'scope': 'numerical/causal prefix diagnostic, no deployment recipe change yet', 'foundation': metadata['foundation_sha256'], 'rows': rows})
        print(json.dumps(row), flush=True)
    atomic_json(output / 'status.json', {'status': 'complete'})


if __name__ == '__main__': main()
