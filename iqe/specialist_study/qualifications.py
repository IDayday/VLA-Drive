"""Bounded real-data checks for this campaign, separate from scientific models."""
import argparse
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.losses import expert_il_terms
from iqe.training.trainer import train
from iqe.internal_baseline.bank import save_candidate, score_candidates
from .binding import scenes_by_role
from .config import load_config
from .data import ReusedContext, verified_base_scores
from .models import make_specialist, EMAExpert
from .objectives import SceneSampler
from .resource_guard import check
from .training import runtime, preload, trainer_config


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def expert_resume(config, device):
    """Eight diagnostic updates total: four continuous versus two plus two."""
    check(config, 'train-expert-cell')
    device = runtime(device)
    folder = config.root/'qualifications/expert_gpu_resume'
    output = folder/'REPORT.json'
    if output.exists():
        return read_json(output)
    manifest = read_json(config.root/'EXPERT_MATRIX_DATA.json')
    lookup = {s.scene_id: s for s in scenes_by_role(config)['expert_fit']}
    scenes = [lookup[i] for i in manifest['scene_ids'][:32]]
    require(len(scenes) == 32, '32 real eligible specialist scenes required')
    cache = preload(config, scenes)
    identities = [s.scene_id for s in scenes]
    model_state = make_specialist(config, True).state_dict()
    cfg = trainer_config(config, 'expert')
    cfg['max_optimizer_steps'] = 4  # diagnostic only; main 5000-step runs untouched
    cfg['warmup_steps'] = 1
    cfg['validation_steps'] = []
    cfg['checkpoint_every_steps'] = 2
    deps = {'diagnostic': 'four_real_updates_private_reader_raw_ema', 'Base': file_hash(config.root/'BASE_BINDING.json'),
            'scene_ids': digest(identities), 'not_a_scientific_expert': True}
    prev = read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['query_architecture']['prev_weight']

    def fetch(entries, dev):
        x = cache.batch([lookup[e['scene_id']] for e in entries], dev)
        return {'x': x, 'target': x['target_raw']}

    def loss(module, batch):
        return expert_il_terms(module(batch['x']), batch['target'], prev_weight=prev)

    def run(name, stop=None, resume=None):
        raw = make_specialist(config, True).to(device)
        raw.load_state_dict(model_state, strict=True)
        model = EMAExpert(raw)
        sampler = SceneSampler(identities, config.seed, deps['scene_ids'])
        train(model, loss, fetch, sampler, cfg, deps, folder/name, device=device, seed=config.seed,
              stop_after=stop, resume=resume, after_optimizer_step=lambda m, _: m.update(),
              denominator_function=lambda b: {'il': torch.tensor(len(b['target']), device=device).float()})
        return sampler.peek(128)

    continuous_ids = run('continuous')
    run('interrupted', stop=2)
    resumed_ids = run('interrupted', resume='auto')
    paths = [folder/name/'step_000004.pt' for name in ('continuous', 'interrupted')]
    a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    fields = ('module', 'optimizer', 'scheduler', 'sampler', 'rng_by_rank', 'consumed_sample_ids')
    equality = {key: equal_tree(a[key], b[key]) for key in fields}
    equality['next_sample_ids'] = continuous_ids == resumed_ids
    require(all(equality.values()), 'real GPU exact-resume mismatch')
    import json
    logs = [[json.loads(line) for line in (folder/name/'steps.jsonl').read_text().splitlines()]
            for name in ('continuous', 'interrupted')]
    loss_equal = [r['loss'] for r in logs[0]] == [r['loss'] for r in logs[1]]
    require(loss_equal, 'real GPU resumed losses changed')
    terminal_ids = run('interrupted', resume='auto')
    terminal = torch.load(paths[1], map_location='cpu', weights_only=False)
    require(terminal_ids == resumed_ids and equal_tree(terminal, b), 'terminal resume consumed/updated state')
    result = {'status': 'PASS', 'device': device, 'hardware': torch.cuda.get_device_name() if 'cuda' in device else 'CPU',
              'Base_binding_hash': deps['Base'], 'real_scene_ids': identities, 'continuous_updates': 4,
              'interrupted_updates': [2, 2], 'terminal_added_updates': 0, 'equality': equality, 'losses_equal': loss_equal,
              'checkpoint_hashes': [file_hash(p) for p in paths], 'production_dropout_preserved': True,
              'diagnostic_model_not_registered_as_candidate': True}
    atomic_json(output, result, immutable=True)
    return result


def official_invariance(config):
    """Fresh official calls, not reuse of the labels under investigation."""
    check(config, 'audit-target-opportunities')
    folder = config.root/'qualifications/official_reference'
    output = folder/'REPORT.json'
    if output.exists():
        return read_json(output)
    ids = read_json(config.root/'EXPERT_MATRIX_DATA.json')['scene_ids'][:2]
    lookup = {s.scene_id: s for s in scenes_by_role(config)['expert_fit']}
    scenes = [lookup[i] for i in ids]
    old = verified_base_scores(config, scenes)
    cache = ReusedContext(config)
    candidates, observations = [], []
    for s in scenes:
        values = cache.read(s)
        for kind in ('base_physical', 'target_physical'):
            candidates.append(save_candidate(config, s, 'qualification_'+kind, digest([s.scene_id, kind]), values[kind].numpy()))
            observations.append(s)
    tests = [('original', [0, 1, 2, 3], 2, 4), ('reordered', [3, 1, 0, 2], 3, 4),
             ('deleted_other', [0, 2], 1, 2), ('duplicated', [0, 0, 1, 2, 3], 1, 5),
             ('chunk_one', [0, 1, 2, 3], 1, 1), ('chunk_three', [0, 1, 2, 3], 2, 3)]
    truth, details = {}, []
    for name, order, workers, chunk in tests:
        root = folder/name
        root.mkdir(parents=True, exist_ok=True)
        for filename in ('SCORE_PROTOCOL.json', 'OFFLINE_BASE_CONTRACT.json', 'BASE_BINDING.json'):
            if not (root/filename).exists():
                shutil.copy2(config.root/filename, root/filename)
        c = replace(config, output_root=str(root), scoring_workers=workers)
        values = []
        for at in range(0, len(order), chunk):
            subset = order[at:at+chunk]
            values.extend(score_candidates(c, [observations[i] for i in subset], [candidates[i] for i in subset], name+str(at)))
        errors = []
        for index, value in zip(order, values):
            comparable = {'total': value.total_score_01, 'components': value.named_components, 'valid': value.component_valid_masks}
            if index not in truth:
                truth[index] = comparable
            require(truth[index] == comparable, 'official label depends on candidate ordering/pool/chunk')
            if index % 2 == 0:
                original = old[observations[index].scene_id]
                require(original['total_score_01'] == value.total_score_01 and original['named_components'] == value.named_components,
                        'fresh official score does not equal original Base bank')
            errors.append(0.)
        details.append({'name': name, 'order': order, 'workers': workers, 'chunk_size': chunk, 'max_abs_error': max(errors), 'labels': len(values)})
    result = {'status': 'PASS', 'real_scene_ids': ids, 'protocol_hash': read_json(config.root/'SCORE_PROTOCOL.json')['protocol_hash'],
              'GT_is_supervision_audit_only': True, 'independent_fresh_official_contexts': True, 'tests': details,
              'cached_Base_vs_fresh_official_exact': True, 'no_navtest_access': True}
    atomic_json(output, result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('expert-resume', 'official-reference'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = load_config(args.config)
    result = expert_resume(config, args.device) if args.command == 'expert-resume' else official_invariance(config)
    import json
    print(json.dumps(result))


if __name__ == '__main__':
    main()
