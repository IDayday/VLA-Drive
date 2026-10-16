"""Compare a fixed early scene stream; never substitute it for ODE evaluation."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import torch

from scripts.analysis.simwam_recipe.summarize_training import summarize_window
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--control', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--window-updates', type=int,
                   help='Optional fixed, non-overlapping complete-collection windows')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.updates < 4 or args.updates % 4:
        raise ValueError('Use complete collections, four updates each')
    if args.window_updates is not None and (args.window_updates < 4 or args.window_updates % 4
                                            or args.updates % args.window_updates):
        raise ValueError('Fixed windows must evenly partition the complete prefix')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    roots = [Path(args.control), Path(args.candidate)]
    identities = [json.loads((r/'identity.json').read_text()) for r in roots]
    for key in ('SFT_sha256','hard_manifest_sha256','projected_condition_identity','score_protocol'):
        if identities[0][key] != identities[1][key]:
            raise ValueError('Unmatched training identity: '+key)
    if any(x['world'] != 8 for x in identities):
        raise ValueError('Eight-rank source reduction required')
    journals = []
    chain_coverage = []
    for root, identity in zip(roots, identities):
        ranks = []
        coverage = dict(checked_group_digests=0, legacy_groups_without_logged_digest=0)
        for rank in range(8):
            with (root/f'metrics_rank{rank}.jsonl').open() as stream:
                rows = [json.loads(line) for line in itertools.islice(stream, args.updates)]
            if [r['updates'] for r in rows] != list(range(1,args.updates+1)):
                raise ValueError('Incomplete or duplicate prefix')
            validate_rows(rows, args.updates)
            for row in rows:
                if (row['behavior']['run'] != identity['sha256']
                    or row['collection'] != (row['updates']-1)//4+1
                    or row['inner_completed'] != (row['updates']-1)%4+1):
                    raise ValueError('Invalid behavior/inner identity')
            for start in range(0, args.updates, 4):
                group = rows[start:start+4]
                present = ['chain_identity' in row for row in group]
                if any(present) and not all(present):
                    raise ValueError('Chain digest disappeared within one collection')
                if all(present):
                    if any(row['chain_identity'] != group[0]['chain_identity'] for row in group):
                        raise ValueError('Chain digest changed during inner reuse')
                    coverage['checked_group_digests'] += 1
                else:
                    coverage['legacy_groups_without_logged_digest'] += 1
                for row in group:
                    if row['behavior']['update'] != start:
                        raise ValueError('Behavior does not precede all four inner updates')
                    if any(row[k] != group[0][k] for k in ('token', 'behavior', 'exploration')):
                        raise ValueError('Collection changed during inner reuse')
            ranks.append(rows)
        journals.append(ranks)
        chain_coverage.append(coverage)
    roots_equal = []; prefixes_equal = []
    for rank in range(8):
        for before, after in zip(journals[0][rank], journals[1][rank]):
            if before['token'] != after['token']:
                raise ValueError('Training scene order differs')
        if journals[0][rank][0]['behavior']['adapter_sha256'] != journals[1][rank][0]['behavior']['adapter_sha256']:
            raise ValueError('Initial LoRA differs')
        rows = [torch.load(r/'behavior/collection000001'/f'rank{rank}.pt',
                           map_location='cpu',weights_only=False,mmap=True) for r in roots]
        if not torch.equal(rows[0]['cond_bg']['context'], rows[1]['cond_bg']['context']):
            raise ValueError('Initial condition differs')
        roots_equal.append(torch.equal(rows[0]['chain'][:,0], rows[1]['chain'][:,0]))
        prefixes_equal.append(torch.equal(rows[0]['chain'][:,:7], rows[1]['chain'][:,:7]))
    if not all(roots_equal) or not all(prefixes_equal):
        raise ValueError('Initial root or common six-step ODE prefix differs')
    summaries = [summarize_window(r,0,args.updates) for r in journals]
    result = dict(schema='simwam_matched_training_prefix_v1',status='MATCHED_TRAINING_PREFIX_ONLY',
        captured_utc=datetime.now(timezone.utc).isoformat(),analysis_identity=file_sha(__file__),
        updates=args.updates,run_identities=[i['sha256'] for i in identities],
        variants=[i['recipe']['variant'] for i in identities],
        chain_digest_journal_coverage=chain_coverage,
        same_scene_order=True,same_initial_LoRA=True,same_initial_condition=True,
        first_collection_root_and_common_ODE_prefix_exact=True,
        scene_order_sha256=fingerprint([[x['token'] for x in rank[::4]] for rank in journals[0]]),
        control=summaries[0],candidate=summaries[1],
        random_policy_delta={k:summaries[1]['stochastic_policy_mean'][k]-summaries[0]['stochastic_policy_mean'][k]
                             for k in summaries[0]['stochastic_policy_mean']},
        new_inference_or_scoring=0,
        limits=['This is a fixed prefix of the registered hard TRAIN population, not an unbiased full-TRAIN estimate.',
                'After the first collection, models have different learned weights; this is the complete window-recipe effect.',
                'Legacy journals without chain digests are counted explicitly; matching behavior/reward fields alone do not independently verify every stored chain.',
                'Random G8 means are not deployed single-ODE scores; no checkpoint/hyperparameter selection from this readout.'])
    if args.window_updates is not None:
        result['schema'] = 'simwam_matched_training_prefix_v2'
        result['windows'] = []
        for start in range(0, args.updates, args.window_updates):
            end = start+args.window_updates
            control, candidate = [summarize_window(r, start, end) for r in journals]
            transitions = {}
            for flag in ('all_full', 'all_zero', 'tied', 'zero_PG', 'progress_only', 'safety_differing'):
                counts = {'both_false': 0, 'control_only': 0, 'candidate_only': 0, 'both_true': 0}
                for before, after in zip(journals[0], journals[1]):
                    for a, b in zip(before[start:end:4], after[start:end:4]):
                        va, vb = bool(a['exploration'][flag]), bool(b['exploration'][flag])
                        key = ('both_true' if vb else 'control_only') if va else ('candidate_only' if vb else 'both_false')
                        counts[key] += 1
                if sum(counts.values()) != (end-start)*2:
                    raise ValueError('Unpaired group transition denominator')
                transitions[flag] = counts
            result['windows'].append(dict(control=control, candidate=candidate,
                paired_scene_group_flags=transitions,
                scope='Identical scene exposures, different policy states and sampler; not same-trajectory intervention.'))
    result['analysis_body_seconds'] = time.monotonic()-started
    result['optimizer_updates_from_analysis'] = 0
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(status=result['status'],updates=args.updates,delta=result['random_policy_delta'])))


if __name__ == '__main__':
    main()
