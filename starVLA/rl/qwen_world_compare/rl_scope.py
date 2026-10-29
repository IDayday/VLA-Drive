"""Architecture-specific deployment LoRA scope with identity-safe installation."""
from collections import defaultdict

from torch import nn

from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear


def _attention_paths(root, prefix):
    paths = []
    for index, block in enumerate(root.blocks):
        for attention in ('self_attn', 'cross_attn'):
            for projection in ('q', 'k', 'v', 'o'):
                paths.append(f'{prefix}.blocks.{index}.{attention}.{projection}')
    return paths


def scope_targets(model):
    """Exact online projection objects, not substring matching or old layer counts."""
    architecture = model.architecture_id
    decision = model.decision
    if architecture == 'A':
        wanted = _attention_paths(decision.actor, 'decision.actor')
    elif architecture == 'B':
        wanted = _attention_paths(decision, 'decision')
    elif architecture == 'C':
        wanted = _attention_paths(decision.actor, 'decision.actor')
        wanted += _attention_paths(decision.world_stream, 'decision.world_stream')
        wanted += [f'decision.actor.world_kv_bridges.{group}.{projection}'
                   for group in range(len(decision.actor.world_kv_bridges))
                   for projection in ('k', 'v')]
    elif architecture == 'D':
        wanted = ['decision.'+path for path in decision.rl_attention_paths()]
    else:
        raise ValueError('Unregistered four-framework architecture')
    unique, seen = [], set()
    for path in wanted:
        projection = model.get_submodule(path)
        if not isinstance(projection, nn.Linear):
            raise ValueError('Expected an unwrapped deployment Linear: '+path)
        if id(projection) not in seen:
            seen.add(id(projection)); unique.append(path)
    if not unique:
        raise ValueError('An empty RL scope cannot update the policy')
    return tuple(unique)


def install_lora(model, *, paths=None, r=16, alpha=32., dropout=0.):
    """Wrap each selected Linear object once and retain every existing alias.

    Uses the unchanged source LoRALinear arithmetic/initialization.  The installer
    differs because shared P/A/R uses one deployment projection object at more
    than one logical role; recursive child-name replacement is insufficient.
    """
    if int(r) != 16 or float(alpha) != 32. or float(dropout) != 0.:
        raise ValueError('The four-framework main comparison fixes ordinary r16/alpha32/dropout0')
    paths = scope_targets(model) if paths is None else tuple(paths)
    targets = {id(model.get_submodule(path)): path for path in reversed(paths)}
    if any(not isinstance(model.get_submodule(path), nn.Linear) for path in paths):
        raise ValueError('LoRA requires the unwrapped IL topology')
    aliases = defaultdict(list)
    # Keep aliases rather than the default remove_duplicate=True traversal.
    for path, module in model.named_modules(remove_duplicate=False):
        if id(module) in targets:
            aliases[id(module)].append(path)
    wrappers, manifest = {}, []
    for object_id, canonical in targets.items():
        base = model.get_submodule(canonical)
        wrapper = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
        wrappers[object_id] = wrapper
        manifest.append(dict(path=canonical, aliases=aliases[object_id],
            in_features=base.in_features, out_features=base.out_features,
            r=wrapper.r, alpha=alpha, scaling=wrapper.scaling,
            parameters=wrapper.lora_A.numel()+wrapper.lora_B.numel(),
            dtype=str(wrapper.lora_A.dtype)))
    # Resolve before mutation so a shared parent's aliases cannot change lookup.
    locations = []
    for object_id, names in aliases.items():
        for path in names:
            parent_path, _, child = path.rpartition('.')
            locations.append((model.get_submodule(parent_path) if parent_path else model,
                              child, wrappers[object_id]))
    for parent, child, wrapper in locations:
        setattr(parent, child, wrapper)
    return tuple(manifest)


def trainable_manifest(actor):
    current = actor.action_expert
    names = {name: parameter for name, parameter in current.named_parameters() if parameter.requires_grad}
    expected_ids = {id(parameter) for module in current.modules() if isinstance(module, LoRALinear)
                    for parameter in (module.lora_A, module.lora_B)}
    if {id(parameter) for parameter in names.values()} != expected_ids:
        raise ValueError('RL parameter roles include a base/readout tensor or omit an adapter')
    if any(parameter.requires_grad for parameter in actor.ref_action_expert.parameters()):
        raise ValueError('Permanent IL reference must remain fully frozen')
    return dict(architecture_id=current.architecture_id,
        trainable_names=['action_expert.'+name for name in names],
        trainable_numel=sum(parameter.numel() for parameter in names.values()),
        adapters=list(actor.scope_manifest), base_frozen=True, reference_frozen=True,
        qwen_text_changes=current.architecture_id == 'D',
        current_prefix_recomputed_each_logprob_call=current.architecture_id in ('B', 'C', 'D'),
        reference='independent deep copy of the selected own-architecture IL parent',
        deployment='single clean-x K10 shift5 ODE; preserve LoRA wrappers')
