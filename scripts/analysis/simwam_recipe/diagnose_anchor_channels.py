"""Read-only channel gradients on sealed behavior; never update or resample.

The only counterfactual is the previously proposed x-weight .2 -> 1 with the
source weighted-average denominator retained. This also rescales y/heading.
It is a gradient diagnostic, not an optimizer step or a policy evaluation.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha


LOGICAL_WEIGHTS = (.2, 1., 1.)
PROPOSED_WEIGHTS = (1., 1., 1.)


def logical_errors(delta):
    """[G,S,H,4] -> [S,3]; heading is normalized sin/cos pair mean."""
    if delta.ndim != 4 or delta.shape[-1] != 4:
        raise ValueError('Expected candidate, stochastic step, horizon, native4')
    squared = delta.square().mean((0, 2))
    return torch.stack((squared[:, 0], squared[:, 1],
                        .5 * (squared[:, 2] + squared[:, 3])), dim=-1)


def component_losses(delta, weights=LOGICAL_WEIGHTS):
    weights = torch.as_tensor(weights, dtype=delta.dtype, device=delta.device)
    if weights.shape != (3,) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Expected three nonnegative logical weights')
    return 50 * logical_errors(delta) * weights / weights.sum()


def source_form_anchor(delta, x_weight=.2):
    """Literal H/D weighted reduction used by the locked source trainer."""
    weights = delta.new_tensor([x_weight, 1., .5, .5]).repeat(delta.shape[2], 1)
    return 50 * ((delta.square() * weights).sum((-1, -2)) / weights.sum()).mean(0)


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if denominator > 0 else None


def gradient_summary(basis):
    """basis=[PG, original x, original y, original heading] gradients."""
    basis = basis.double()
    pg = basis[0]
    original = basis[1:].sum(0)
    # .2/2.2 -> 1/3, and 1/2.2 -> 1/3. Do not silently keep denominator 2.2.
    multipliers = basis.new_tensor([11/3, 11/15, 11/15])
    proposed = (basis[1:] * multipliers[:, None]).sum(0)
    old_total, proposed_total = pg + original, pg + proposed
    return dict(
        PG_norm=float(pg.norm()),
        component_norms={k: float(v.norm()) for k, v in zip(('x', 'y', 'heading_pair'), basis[1:])},
        component_PG_cosines={k: cosine(pg, v) for k, v in zip(('x', 'y', 'heading_pair'), basis[1:])},
        original_anchor_norm=float(original.norm()),
        original_PG_anchor_cosine=cosine(pg, original),
        proposed_anchor_norm=float(proposed.norm()),
        proposed_PG_anchor_cosine=cosine(pg, proposed),
        original_total_norm=float(old_total.norm()),
        proposed_total_norm=float(proposed_total.norm()),
        total_direction_cosine=cosine(old_total, proposed_total),
        change_relative_to_original_total=(float((proposed_total-old_total).norm()/old_total.norm())
                                           if old_total.norm() > 0 else None),
        gram_order=['PG', 'x', 'y', 'heading_pair'], gram=(basis @ basis.T).tolist(),
    )


def qualify():
    """CPU arithmetic oracle: literal source vs channel decomposition."""
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU qualification must not initialize CUDA')
    records = []
    for dtype, atol, rtol in [(torch.float64, 1e-12, 1e-12), (torch.float32, 2e-6, 2e-6)]:
        for shape in [(1, 3, 8, 4), (8, 3, 8, 4)]:
            generator = torch.Generator().manual_seed(42)
            delta = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
            for x_weight, logical in [(.2, LOGICAL_WEIGHTS), (1., PROPOSED_WEIGHTS)]:
                direct = source_form_anchor(delta, x_weight)
                decomposed = component_losses(delta, logical).sum(-1)
                torch.testing.assert_close(direct, decomposed, atol=atol, rtol=rtol)
                gd = torch.autograd.grad(direct.sum(), delta, retain_graph=True)[0]
                gc = torch.autograd.grad(decomposed.sum(), delta, retain_graph=True)[0]
                torch.testing.assert_close(gd, gc, atol=atol, rtol=rtol)
                records.append(dict(dtype=str(dtype), shape=shape, x_weight=x_weight,
                                    loss_max_abs=float((direct-decomposed).abs().max()),
                                    gradient_max_abs=float((gd-gc).abs().max())))
    zero = torch.zeros(8, 3, 8, 4, requires_grad=True)
    assert torch.equal(component_losses(zero), torch.zeros(3, 3))
    assert cosine(torch.zeros(4), torch.ones(4)) is None
    errors = logical_errors(torch.ones(1, 3, 8, 4))
    assert torch.equal(errors, torch.ones(3, 3))
    # Check the proposed gradient exactly follows the advertised normalization.
    delta = torch.randn(8, 3, 8, 4, dtype=torch.float64, requires_grad=True)
    parts = component_losses(delta)
    combined = parts * delta.new_tensor([11/3, 11/15, 11/15])
    torch.testing.assert_close(combined.sum(-1), source_form_anchor(delta, 1.), atol=1e-12, rtol=1e-12)
    return dict(status='PASS', checks=records, zero_and_heading_pair='PASS',
                cuda_initialized=torch.cuda.is_initialized(),
                new_candidates=0, optimizer_updates=0)


def diagnose(args):
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.simwam_recipe.checkpoint import device_tree
    from starVLA.rl.simwam_recipe.contracts import require_dispatch, load_recipe
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.simwam_recipe.trainer_adapter import make_actor

    host = require_dispatch()
    configure_numerics()
    started = time.time()
    run, checkpoint = Path(args.run), Path(args.checkpoint)
    identity = json.loads((run/'identity.json').read_text())
    cfg = identity['recipe']
    # Still enforce the formal recipe; do not add a third training variant.
    cfg_path = Path('configs/flow_rl') / ('simwam_source_ddp.yaml' if cfg['variant'] == 'R_SOURCE'
                                        else 'simwam_canonical_ddp.yaml')
    if cfg != load_recipe(cfg_path):
        raise ValueError('Run recipe differs from the unchanged formal configuration')
    complete = json.loads((checkpoint/'COMPLETE').read_text())
    if file_sha(checkpoint/'files.json') != complete['files_sha256']:
        raise ValueError('Checkpoint seal changed')
    sealed = json.loads((checkpoint/'files.json').read_text())
    for name in ('optimizer.pt', 'rank0.pt', 'identity.json'):
        if file_sha(checkpoint/name) != sealed[name]:
            raise ValueError('Checkpoint file changed: '+name)
    state = torch.load(checkpoint/'rank0.pt', map_location='cpu', weights_only=False)['state']
    update = state['updates']
    if update != args.collection*4 or state['next_inner'] != 0:
        raise ValueError('Use the sealed milestone immediately after this four-inner collection')
    previous = json.loads(Path(args.previous_diagnostic).read_text())
    if (previous['run_identity'] != identity['sha256'] or previous['collection'] != args.collection
            or Path(previous['checkpoint']) != checkpoint):
        raise ValueError('Unmatched existing diagnostic population')
    world = identity['world']
    if world != 8:
        raise ValueError('Only the actual eight-scene formal batch is supported')
    actor, _, _ = make_actor(cfg)
    saved = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=False)
    restore_adapters(actor, saved['adapters'])
    params = [p for p in actor.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    loaded = time.time()
    # Per-step gradients, all ranks averaged including a possible tied rank.
    basis = torch.zeros(3, 4, count)
    values = torch.zeros(3, 3, dtype=torch.float64)
    direct_anchor_gradient = torch.zeros(3, count)
    batches = []
    direct_checks = []
    backward_calls = 0

    def gradient(scalar):
        nonlocal backward_calls
        backward_calls += 1
        result = torch.autograd.grad(scalar, params, retain_graph=True)
        return torch.cat([v.detach().float().flatten().cpu() for v in result])

    for rank in range(world):
        path = run/'behavior'/f'collection{args.collection:06}'/f'rank{rank}.pt'
        sha = file_sha(path)
        if sha != previous['batches'][rank]['sha256']:
            raise ValueError('Saved behavior differs from audited batch')
        row = device_tree(torch.load(path, map_location='cpu', weights_only=False), actor.device)
        if (row['behavior_version']['run'] != identity['sha256'] or
                row['behavior_version']['update'] != update-4 or
                row['stoch_idx'].tolist() != [7, 8, 9]):
            raise ValueError('Wrong behavior/step identity')
        logp, terms = actor.action_chain_logprobs(row['cond_bg'], row['chain'], row['timesteps'], row['deltas'],
                                                stoch_idx=row['stoch_idx'], return_terms=True)
        with torch.no_grad():
            _, ref = actor.action_chain_logprobs(row['cond_bg'], row['chain'], row['timesteps'], row['deltas'],
                                                stoch_idx=row['stoch_idx'], return_terms=True, use_ref=True)
        ratio = (logp-row['old_logp']).exp()
        adv = row['advantages'].clamp(-5, 5)[:, None]
        mask = row['adv_mask'][:, None]
        pg = (torch.maximum(-ratio*adv, -ratio.clamp(.995, 1.005)*adv)*mask).sum(0)/mask.sum().clamp_min(1)
        delta = terms['mean']-ref['mean']
        parts = component_losses(delta)
        direct = source_form_anchor(delta)
        torch.testing.assert_close(parts.sum(-1), direct, atol=1e-8, rtol=2e-5)
        rank_basis = torch.zeros_like(basis) if rank == 0 else None
        for j in range(3):
            for component, scalar in enumerate([pg[j], *parts[j]]):
                g = gradient(scalar/world)
                basis[j, component] += g
                if rank_basis is not None:
                    rank_basis[j, component] = g
            direct_anchor_gradient[j] += gradient(direct[j]/world)
        values += parts.detach().double().cpu()/world
        if rank == 0:
            # Actual unchanged upstream _policy_loss is the independent oracle.
            actual_total, metrics = actor(row)
            torch.testing.assert_close(actual_total, pg.mean()+parts.sum(-1).mean(), atol=1e-7, rtol=2e-5)
            actual_grad = gradient(actual_total/world)
            reconstructed = rank_basis.sum(1).mean(0)
            torch.testing.assert_close(actual_grad, reconstructed, atol=1e-7, rtol=3e-4)
            direct_checks.append(dict(rank=rank, source_total=float(actual_total),
                                     reconstructed_total=float(pg.mean()+parts.sum(-1).mean()),
                                     total_gradient_max_abs=float((actual_grad-reconstructed).abs().max())))
        batches.append(dict(rank=rank, sha256=sha, behavior=row['behavior_version'],
                            PG_mask_fraction=float(row['adv_mask'].mean())))
        del logp, terms, ref, pg, parts, direct, delta, ratio, row

    torch.testing.assert_close(basis[:, 1:].sum(1), direct_anchor_gradient, atol=1e-7, rtol=3e-4)
    # Neither parameter updates nor .grad accumulation is part of this readout.
    for name, param in actor.named_parameters():
        if param.requires_grad and not torch.equal(param.detach().cpu(), saved['adapters'][name]):
            raise ValueError('Read-only analysis changed adapter: '+name)
        if param.grad is not None:
            raise ValueError('Unexpected retained parameter gradient: '+name)
    steps = []
    for j, index in enumerate((7, 8, 9)):
        summary = gradient_summary(basis[j])
        # Old diagnostics reduced these 4.7M-element vectors in FP32. Compare
        # like-for-like; the new FP64 norms/Gram matrix avoid that reduction
        # error and must not be mistaken for a change in the actor gradient.
        legacy_norms = dict(PG=float(basis[j, 0].norm()),
                            anchor=float(direct_anchor_gradient[j].norm()))
        for got, old_key in [(legacy_norms['PG'], 'PG_gradient_norm'),
                             (legacy_norms['anchor'], 'beta50_anchor_gradient_norm')]:
            torch.testing.assert_close(torch.tensor(got), torch.tensor(previous['steps'][j][old_key]),
                                       atol=1e-7, rtol=3e-4)
        steps.append(dict(step=index, **summary,
                          old_diagnostic_FP32_norms=legacy_norms,
                          original_beta50_anchor_loss_components=values[j].tolist()))
    ended = time.time()
    return dict(schema='simwam_anchor_channel_diagnostic_v1', status='PASS',
                source_file_sha256=file_sha(__file__), host=host,
                run_identity=identity['sha256'], checkpoint=str(checkpoint),
                checkpoint_files_sha256=complete['files_sha256'], collection=args.collection,
                previous_diagnostic_sha256=file_sha(args.previous_diagnostic),
                steps=steps, all_steps_mean=gradient_summary(basis.mean(0)), batches=batches,
                original_logical_weights=LOGICAL_WEIGHTS, proposed_logical_weights=PROPOSED_WEIGHTS,
                effective_component_multipliers=[11/3, 11/15, 11/15],
                direct_upstream_total_check=direct_checks,
                direct_anchor_gradient_max_abs=float((basis[:, 1:].sum(1)-direct_anchor_gradient).abs().max()),
                parameters_unchanged=True, optimizer_updates=0, new_candidates=0, new_scoring_calls=0,
                cost=dict(started_unix=started, ended_unix=ended, loading_seconds=loaded-started,
                          analysis_seconds=ended-loaded, worker_wall_seconds=ended-started,
                          current_velocity_calls=actor._velocity_calls,
                          reference_velocity_calls=actor._reference_calls, autograd_calls=backward_calls,
                          GPU_time_semantics='single shared-GPU worker wall, not exclusive compute time'),
                limitations=['Only one saved eight-scene batch per milestone; not population inference.',
                             'Unpreconditioned gradients are not Adam steps or predicted policy gains.',
                             'No deployment rollout, new score, or counterfactual training was run.',
                             'The proposed x-weight change weakens normalized y/heading coefficients.',
                             'Partial stochastic-step/clamped-probability source proxy is unchanged.'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--checkpoint')
    parser.add_argument('--collection', type=int)
    parser.add_argument('--previous-diagnostic')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing diagnostic: '+str(output))
    if not args.qualify and not all([args.run, args.checkpoint, args.collection, args.previous_diagnostic]):
        parser.error('A real diagnostic needs run/checkpoint/collection/previous-diagnostic')
    result = qualify() if args.qualify else diagnose(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(dict(status=result['status'], output=str(output), optimizer_updates=0, new_candidates=0)))


if __name__ == '__main__':
    main()
