"""Read an equal, complete optimizer-update prefix; no driving-score claim."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def multiplier(u):
    return .001+.999*u/1000 if u <= 1000 else .1+.9*(1+math.cos(math.pi*(u-1000)/49000))/2


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    args = parser.parse_args()
    if args.updates < 4 or args.updates % 4:
        parser.error('Use a complete four-inner prefix')
    registry = json.loads((args.root/'RUNS.json').read_text())
    output = args.root/'analysis'/f'production_prefix_u{args.updates:06}'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    baseline_scene_order = None
    for run in registry['runs']:
        base = Path(run['root'])
        identity = json.loads((base/'identity.json').read_text())
        groups = identity['actual_optimizer']['parameter_groups']
        peaks = {g['name']: g['lr'] for g in groups}
        assert sum(g['numel'] for g in groups) == 52707840
        assert identity['recipe']['lr_warmup_steps'] == 1000
        assert identity['recipe']['max_steps'] == 50000
        assert identity['recipe']['research']['campaign_root'] == str(args.root)
        rows = []
        scene_order = []
        receipts = []
        for rank in range(8):
            path = base/f'metrics_rank{rank}.jsonl'
            with path.open('rb') as handle:
                lines = [handle.readline() for _ in range(args.updates)]
            assert all(lines), f'Incomplete rank {rank} prefix'
            current = [json.loads(line) for line in lines]
            assert [r['updates'] for r in current] == list(range(1, args.updates+1))
            for row in current:
                u = row['updates']
                assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
                assert set(row['optimizer_group_lrs_used']) == set(peaks)
                for name, peak in peaks.items():
                    assert abs(row['optimizer_group_lrs_used'][name]-peak*multiplier(u-1)) < 1e-15
                    assert abs(row['optimizer_group_lrs'][name]-peak*multiplier(u)) < 1e-15
            data = b''.join(lines)
            archive = output/f"{run['id']}_rank{rank}.jsonl.gz"
            archive.write_bytes(gzip.compress(data, mtime=0))
            receipts.append(dict(rank=rank, journal=str(path), prefix_sha256=digest(data),
                archived_prefix=str(archive), archive_sha256=digest(archive.read_bytes())))
            rows.extend(current)
            scene_order.extend((rank, r['collection'], r['token'])
                for r in current if r['inner_completed'] == 1)
        if baseline_scene_order is None:
            baseline_scene_order = scene_order
        assert scene_order == baseline_scene_order, 'Actual scene order differs across arms'
        assert len({x[2] for x in scene_order}) == len(scene_order), 'Repeated first-pass scene'
        fresh = [r for r in rows if r['inner_completed'] == 1]
        assert len(fresh) == args.updates//4*8
        summation = lambda key: sum(r['metrics'].get(key, 0) for r in fresh)
        quality = dict(groups=len(fresh), candidates=len(fresh)*8,
            strict_high=summation('quality_strict_high_count'),
            feasible=summation('quality_feasible_count'),
            positive_high=summation('quality_positive_high_count'),
            negative_high=summation('quality_negative_high_count'))
        norm = [r['gradient_norm'] for r in rows]
        blocked = {str(i): sum(r['metrics'].get('ppo_blocked_fraction', 0) for r in rows
            if r['inner_completed'] == i)/(args.updates//4*8) for i in range(1, 5)}
        results.append(dict(arm=run['id'], status='PASS_ACTUAL_8RANK_PREFIX',
            identity_sha256=identity['sha256'], optimizer_updates=args.updates,
            metric_rows=len(rows), actual_trainable_numel=sum(g['numel'] for g in groups),
            optimizer_peak_lrs=peaks, all_used_and_next_LRs_match_fixed_schedule=True,
            same_actual_scene_order_as_control=True, first_pass_unique_scenes=len(scene_order),
            all_loss_and_gradients_finite=True, gradient_norm_mean=sum(norm)/len(norm),
            gradient_norm_max=max(norm), gradient_above_clip1_rows=sum(x > 1 for x in norm),
            ppo_sign_blocked_mean_by_inner=blocked, collection_quality=quality,
            combined_branches={key: summation('combined_'+key+'_groups')
                for key in ('dense', 'saturated', 'inversion')}, receipts=receipts))
    result = dict(schema='world_standard_actual_training_prefix_review_v1',
        status='PASS_ACTUAL_4ARMS_8RANK', at_unix=time.time(),
        root=str(args.root), updates=args.updates, results=results,
        interpretation='Actual production update/schedule/signal check; generated quality and clipping are process observations, not NAVTEST gains or convergence evidence.',
        complete_NAVTEST_results=0, performance_improvement_claim=False,
        gradient_rows_repeat_the_global_DDP_norm_across_ranks=True,
        collection_counts_use_inner1_only=True)
    path = args.root/'analysis'/f'STANDARD_PRODUCTION_PREFIX_u{args.updates:06}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}))
    for row in results:
        print(json.dumps({k:v for k,v in row.items() if k != 'receipts'}))


if __name__ == '__main__':
    main()
