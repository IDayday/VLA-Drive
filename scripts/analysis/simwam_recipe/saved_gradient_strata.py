"""Finite, read-only attribution of saved-batch loss gradients by SFT stratum.

Contributions retain the source world-size divisor. They are gradients of the
post-inner checkpoint loss proxy, not Adam updates or causal safety benefits.
"""
import itertools
import json
from pathlib import Path

import torch

from scripts.analysis.simwam_recipe.analyze_training_safety_signal import safety_type, validate_saved
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.simwam_recipe.data_stream import make_scene_stream


def vector_summary(pg, anchor):
    pg, anchor = pg.double(), anchor.double()
    pn, an = pg.norm(), anchor.norm()
    return dict(PG_norm=float(pn), beta50_anchor_norm=float(an),
                cosine=float(torch.dot(pg, anchor)/(pn*an)) if pn > 0 and an > 0 else None,
                total_norm=float((pg+anchor).norm()),
                anchor_to_PG=float(an/pn) if pn > 0 else None)


def summarize_strata(basis, counts):
    """[safe/unsafe, PG/anchor, step, parameter], already divided by world."""
    if basis.ndim != 4 or basis.shape[:2] != (2, 2) or sum(counts) != 8:
        raise ValueError('Expected two source strata in the actual world8 batch')
    if not torch.isfinite(basis).all() or any(n < 0 for n in counts):
        raise ValueError('Nonfinite gradient or negative population')
    values = basis.double().mean(2)
    pg, anchor = values.sum(0)
    norm2 = torch.dot(pg, pg)
    strata = {}
    for i, key in enumerate(('SFT_safe', 'SFT_unsafe')):
        g, h = values[i]
        strata[key] = dict(groups=counts[i], **vector_summary(g, h),
            dot_fraction_of_total_PG=float(torch.dot(g, pg)/norm2) if norm2 > 0 else None,
            mean_per_scene_PG_norm=float(g.norm()*8/counts[i]) if counts[i] else None)
    return dict(all_steps_mean=vector_summary(pg, anchor), strata=strata,
        step_summaries=[vector_summary(*basis[:, :, j].double().sum(0)) for j in range(basis.shape[2])],
        interpretation='Raw source loss gradients at the checkpoint after inner4; not optimizer displacement. '
                       'Stratum contributions use divisor8 including tied ranks. Dot fractions may be negative '
                       'or exceed1 under cancellation; they are not causal safety percentages.')


class SavedGradientStrata:
    def __init__(self, registration, run, checkpoint, collection, identity, count):
        self.path = Path(registration)
        self.reg = json.loads(self.path.read_text())
        matches = [j for j in self.reg['jobs'] if j['run'] == str(run) and
                   j['checkpoint'] == str(checkpoint) and j['collection'] == collection]
        if len(matches) != 1:
            raise ValueError('Batch is outside the fixed gradient-analysis registration')
        self.job = matches[0]
        if identity['sha256'] != self.job['run_identity'] or file_sha(run/'identity.json') != self.job['identity_sha256']:
            raise ValueError('Run identity changed')
        if file_sha(checkpoint/'files.json') != self.job['checkpoint_files_sha256']:
            raise ValueError('Checkpoint changed after registration')
        for key in ('hard_manifest', 'population'):
            if file_sha(self.reg[key]['path']) != self.reg[key]['sha256']:
                raise ValueError('Registered data changed')
        self.population = json.loads(Path(self.reg['population']['path']).read_text())
        manifest = json.loads(Path(self.reg['hard_manifest']['path']).read_text())
        self.types = {r['token']: safety_type(r['metrics']) for r in manifest['rows']}
        held = set(self.population['dev_tokens']) | set(self.population['navtest_tokens'])
        if not set(self.types) <= set(self.population['train_tokens']) or set(self.types) & held:
            raise ValueError('Gradient audit must use TRAIN only')
        if {self.population['token_logs'][t] for t in self.types} & {self.population['token_logs'][t] for t in held}:
            raise ValueError('Held-out log overlap')
        self.stream = make_scene_stream(identity['recipe'], self.reg['hard_manifest']['path'], 8)
        self.run, self.identity, self.collection = run, identity, collection
        self.basis = torch.zeros(2, 2, 3, count)
        self.counts, self.rank_strata, self.evidence = [0, 0], {}, []
        self.oracle = torch.zeros(count)
        self.oracle_max_abs = []

    def validate(self, rank, saved):
        path = self.run/'behavior'/f'collection{self.collection:06d}'/f'rank{rank}.pt'
        if file_sha(path) != self.job['behavior_sha256'][rank]:
            raise ValueError('Saved batch changed after registration')
        with (self.run/f'metrics_rank{rank}.jsonl').open() as f:
            row = json.loads(next(itertools.islice(f, self.collection*4-1, None)))
        if row['updates'] != self.collection*4 or row['inner_completed'] != 4:
            raise ValueError('Expected completed inner4 journal')
        digest = validate_saved(saved, row, rank, self.identity, self.population, self.stream,
                                reward_summary_device=torch.device('cuda'))
        if saved['behavior_version']['update'] != self.collection*4-4:
            raise ValueError('Wrong old behavior boundary')
        stratum = int(self.types[saved['token']] != 'SFT_single_seed_safe')
        self.rank_strata[rank] = stratum
        self.counts[stratum] += 1
        self.evidence.append(dict(rank=rank, SFT_stratum=('SFT_safe', 'SFT_unsafe')[stratum],
            chain_identity=digest, global_std=float(saved['global_std']),
            reward_summary_backend='Exact CUDA FP32 reproduction of journal',
            CPU_vs_logged_reward_mean_abs_error=abs(float(saved['rewards'].mean())-row['exploration']['train_reward_mean']),
            PG_mask_fraction=float(saved['adv_mask'].mean()), exploration=row['exploration']))

    def add(self, rank, kind, step, gradient):
        self.basis[self.rank_strata[rank], kind, step] += gradient

    def finish(self, pg, anchor):
        torch.testing.assert_close(self.basis.sum(0), torch.stack((pg, anchor)), atol=1e-7, rtol=3e-4)
        reconstructed = self.basis.double().sum((0, 1)).mean(0)
        torch.testing.assert_close(reconstructed, self.oracle.double(), atol=1e-7, rtol=3e-4)
        return dict(registration_sha256=file_sha(self.path), helper_sha256=file_sha(__file__),
            **summarize_strata(self.basis, self.counts), evidence=self.evidence,
            source_loss_gradient_oracle_max_abs_by_rank=self.oracle_max_abs,
            summed_source_gradient_max_abs=float((reconstructed-self.oracle.double()).abs().max()),
            source_loss_oracle='Unchanged actor.forward/_policy_loss evaluated on all8 saved groups; no optimizer step')


def qualify():
    """Independent autograd oracle including tied ranks and empty strata."""
    checks = []
    for unsafe_count in (0, 2, 8):
        p = torch.tensor([.3, -.7, .5, 1.2], dtype=torch.float64, requires_grad=True)
        basis = torch.zeros(2, 2, 3, 4, dtype=torch.float64)
        losses = []
        for rank in range(8):
            label = int(rank < unsafe_count)
            for step in range(3):
                v = p.new_tensor([rank+1, step+1, -2, .5])
                # One tied rank has zero PG but retains its reference anchor.
                pg = (p.dot(v)-.1).square() * float(rank != 0)
                anchor = 50*(p-p.new_tensor([.1, .2, .3, .4])).square().mean()*(step+1)
                losses.append((pg+anchor)/24)
                for kind, scalar in enumerate((pg, anchor)):
                    basis[label, kind, step] += torch.autograd.grad(scalar/8, p, retain_graph=True)[0]
        direct = torch.autograd.grad(sum(losses), p)[0]
        torch.testing.assert_close(basis.sum((0, 1)).mean(0), direct, atol=1e-12, rtol=1e-12)
        result = summarize_strata(basis, [8-unsafe_count, unsafe_count])
        assert abs(result['all_steps_mean']['total_norm']-float(direct.norm())) < 1e-12
        assert abs(sum(s['dot_fraction_of_total_PG'] for s in result['strata'].values())-1) < 1e-12
        checks.append(dict(unsafe_groups=unsafe_count, source_rank_divisor=8, tied_anchor_preserved=True))
    cancellation = torch.zeros(2, 2, 3, 1)
    cancellation[0, 0] = 2; cancellation[1, 0] = -1
    result = summarize_strata(cancellation, [6, 2])
    assert result['strata']['SFT_safe']['dot_fraction_of_total_PG'] == 2
    assert result['strata']['SFT_unsafe']['dot_fraction_of_total_PG'] == -1
    zero = summarize_strata(torch.zeros_like(cancellation), [8, 0])
    assert zero['all_steps_mean']['cosine'] is None
    assert zero['strata']['SFT_unsafe']['mean_per_scene_PG_norm'] is None
    invalid = cancellation.clone(); invalid[0, 0, 0, 0] = float('nan')
    try: summarize_strata(invalid, [6, 2])
    except ValueError: pass
    else: raise AssertionError('Nonfinite gradient admitted')
    return dict(status='PASS', independent_autograd_cases=checks,
                cancellation_and_empty_and_nonfinite='PASS', source_sha256=file_sha(__file__),
                new_candidates=0, new_scores=0, optimizer_updates=0)


if __name__ == '__main__':
    import argparse
    from starVLA.rl.flow_grpo.transactions import atomic_json
    parser = argparse.ArgumentParser(description='CPU qualification of the gradient readout only')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    if torch.cuda.is_initialized(): raise RuntimeError('CPU qualification must not initialize CUDA')
    result = qualify(); atomic_json(output, result); print(json.dumps(result))
