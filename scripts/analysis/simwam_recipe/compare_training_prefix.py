"""Compare a fixed early scene stream; never substitute it for ODE evaluation."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import torch

from scripts.analysis.simwam_recipe.summarize_training import summarize_window
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--control', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.updates < 4 or args.updates % 4:
        raise ValueError('Use complete collections, four updates each')
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
    for root, identity in zip(roots, identities):
        ranks = []
        for rank in range(8):
            with (root/f'metrics_rank{rank}.jsonl').open() as stream:
                rows = [json.loads(line) for line in itertools.islice(stream, args.updates)]
            if [r['updates'] for r in rows] != list(range(1,args.updates+1)):
                raise ValueError('Incomplete or duplicate prefix')
            for row in rows:
                if (row['behavior']['run'] != identity['sha256']
                    or row['collection'] != (row['updates']-1)//4+1
                    or row['inner_completed'] != (row['updates']-1)%4+1):
                    raise ValueError('Invalid behavior/inner identity')
            ranks.append(rows)
        journals.append(ranks)
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
        same_scene_order=True,same_initial_LoRA=True,same_initial_condition=True,
        first_collection_root_and_common_ODE_prefix_exact=True,
        scene_order_sha256=fingerprint([[x['token'] for x in rank[::4]] for rank in journals[0]]),
        control=summaries[0],candidate=summaries[1],
        random_policy_delta={k:summaries[1]['stochastic_policy_mean'][k]-summaries[0]['stochastic_policy_mean'][k]
                             for k in summaries[0]['stochastic_policy_mean']},
        new_inference_or_scoring=0,
        limits=['Eight initial scenes and the following small fixed TRAIN prefix are not an unbiased population estimate.',
                'After the first collection, models have different learned weights; this is the complete window-recipe effect.',
                'Random G8 means are not deployed single-ODE scores; no checkpoint/hyperparameter selection from this readout.'])
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(status=result['status'],updates=args.updates,delta=result['random_policy_delta'])))


if __name__ == '__main__':
    main()
