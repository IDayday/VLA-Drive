"""Account for a completed A-only S0 profile and its native resume boundary.

The 275-scene prefix deliberately exercises a 19-scene tail. Estimate the full
A run from full and tail batches separately, rather than pretending every
profile update contains 32 scenes. This is a cost/continuation check, not a
planning-quality test or an execution-precision acceptance experiment.
"""
import argparse
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, read, source_sha


def rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def percentile(values, fraction):
    values = sorted(values)
    at = (len(values) - 1) * fraction
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (values[high] - values[low]) * (at - low)


def timing(values):
    return dict(count=len(values), mean_seconds=statistics.mean(values),
                median_seconds=statistics.median(values), p90_seconds=percentile(values, .9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'registration', 'population-contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--warmup-after-resume', type=int, default=20)
    parser.add_argument('--cpu-receipt', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.warmup_after_resume < 0:
        raise ValueError('Warmup cannot be negative')
    resources = check_host()
    if args.output.exists():
        raise FileExistsError('Preserve earlier profile evidence')
    registered = read(args.registration)
    contract = read(args.population_contract)
    meta, status = read(args.run / 'identity.json'), read(args.run / 'status.json')
    steps = rows(args.run / 'steps.jsonl')
    if (status['status'] != 'COMPLETE' or status['completed'] != registered['updates'] or
            meta['scope'] != 'profile' or meta['source_sha'] != registered['training_source_sha'] or
            meta['data']['population_contract'] != contract or
            registered['population_contract_sha256'] != digest(args.population_contract)):
        raise ValueError('The exact registered A-only profile has not completed')
    for key in ('world_size', 'global_batch', 'micro_batch', 'scene_count', 'updates'):
        if meta[key] != registered[key]:
            raise ValueError('Changed profile execution contract: ' + key)
    initialization = read(args.run / 'initialization.json')
    if initialization['driving_weights_loaded'] or initialization['future_teacher_in_model']:
        raise ValueError('Profile must begin from generic Qwen and random driving modules')
    if (meta['schedule']['horizon'] != registered['schedule_updates'] or
            meta['schedule']['warmup'] != registered['warmup']):
        raise ValueError('The profile did not use the registered full-run scheduler')
    if [r['update'] for r in steps] != list(range(1, registered['updates'] + 1)):
        raise ValueError('Missing or duplicate optimizer updates')
    world, batch, size = meta['world_size'], meta['global_batch'], meta['scene_count']
    batches_per_epoch = math.ceil(size / batch)
    exposure = 0
    for row in steps:
        epoch, offset = divmod(row['update'] - 1, batches_per_epoch)
        actual_batch = min(batch, size - offset * batch)
        exposure += actual_batch
        if (row['epoch'] != epoch or row['offset'] != offset + 1 or row['exposure'] != exposure or
                row['counts']['ego_scenes'] != actual_batch):
            raise ValueError('Incorrect scene exposure/data position at update ' + str(row['update']))
        if not row['fp32_master_update_verified'] or row['sampled_changed_elements'] <= 0:
            raise ValueError('No verified FP32-master update')
        expected_losses = {'ego_fm', 'current_dino', 'future_clip', 'interaction'}
        if not expected_losses <= set(row['losses']) or not all(math.isfinite(v) for v in row['losses'].values()):
            raise ValueError('Incomplete or nonfinite full S0 objective')
        if len(row['per_rank_profile']) != world or not math.isfinite(row['grad_norm']):
            raise ValueError('Missing rank timing or invalid gradient norm')
        if any(len(rank) != 5 or not all(math.isfinite(v) and v >= 0 for v in rank)
               for rank in row['per_rank_profile']):
            raise ValueError('Invalid native timing/memory record')
        row['actual_batch_scenes'] = actual_batch
    if exposure != status['exposure'] or exposure != registered['expected_exposure']:
        raise ValueError('Final exposure differs from registration')
    pause_update = min(registered['milestones'])
    boundaries = {}
    for update in (pause_update, registered['updates']):
        folder = args.run / 'checkpoints' / f'milestone_{update:06d}'
        saved = read(folder / 'COMPLETE.json')
        row = steps[update - 1]
        if (saved['identity'] != meta['identity'] or
                any(saved[k] != row[k] for k in ('epoch', 'offset', 'exposure')) or
                saved['completed'] != update or saved['scheduler']['completed'] != update):
            raise ValueError('Native checkpoint boundary differs from actual step records')
        rng_files = [folder / f'rng_rank{rank}.pt' for rank in range(world)]
        optimizers = list(folder.glob('*optim_states.pt'))
        if not all(p.is_file() for p in rng_files) or len(optimizers) != world:
            raise ValueError('Native checkpoint lacks a rank RNG or optimizer partition')
        boundaries[str(update)] = dict(complete_sha256=digest(folder / 'COMPLETE.json'),
            RNG_sha256=[digest(p) for p in rng_files], optimizer_partitions=len(optimizers),
            payload_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()))
    steady = [row for row in steps if row['update'] > pause_update + args.warmup_after_resume]
    if len(steady) < 40:
        raise ValueError('Too few registered stable updates for this cost check')
    periods = [row['ended_unix'] - steps[row['update'] - 2]['ended_unix'] for row in steady]
    if min(periods) <= 0:
        raise ValueError('Non-monotonic training clock')
    rank_seconds = [max(rank[0] for rank in row['per_rank_profile']) for row in steady]
    grouped = {kind: [period for row, period in zip(steady, periods)
                      if (row['actual_batch_scenes'] == batch) == (kind == 'full')]
               for kind in ('full', 'tail')}
    if not all(grouped.values()):
        raise ValueError('Profile did not exercise both full and tail optimizer batches')
    formal_size = len(read(contract['index']))
    formal_updates = registered['schedule_updates']
    formal_epoch_updates = math.ceil(formal_size / batch)
    epochs, remainder = divmod(formal_updates, formal_epoch_updates)
    tail_updates = epochs if formal_size % batch else 0
    expected_exposure = epochs * formal_size + remainder * batch
    estimate_seconds = ((formal_updates - tail_updates) * statistics.mean(grouped['full']) +
                        tail_updates * statistics.mean(grouped['tail']))
    checkpoint_costs = rows(args.run / 'checkpoint_costs.jsonl')
    inference = read(args.run / 'inference_profile.json')
    cpu_receipts = []
    for path in args.cpu_receipt:
        receipt = read(path)
        if receipt['exit_code'] != 0:
            raise ValueError('A profile invocation did not terminate successfully')
        cpu_receipts.append(dict(path=str(path), sha256=digest(path), **receipt))
    startup_memory = read(args.run / 'startup_memory.json')
    report = dict(schema='A_only_native_profile_accounting_v1', passed=True,
        profile_identity=meta['identity'], training_source_sha=meta['source_sha'],
        report_source_sha=source_sha(ROOT), profile_registration_sha256=digest(args.registration),
        actual_updates=len(steps), scene_exposure=exposure, world_size=world,
        resume_update=pause_update, post_resume_warmup_updates=args.warmup_after_resume,
        stable_updates=len(steady), slowest_rank_optimizer_timing=timing(rank_seconds),
        wall_optimizer_periods=timing(periods), actual_global_scenes_per_second=sum(r['actual_batch_scenes'] for r in steady)/sum(periods),
        full_batch_periods=timing(grouped['full']), tail_batch_periods=timing(grouped['tail']),
        per_rank_peak_allocated_bytes=[max(row['per_rank_profile'][rank][2] for row in steps) for rank in range(world)],
        per_rank_peak_reserved_bytes=[max(row['per_rank_profile'][rank][3] for row in steps) for rank in range(world)],
        startup_memory=startup_memory,
        checkpoint_boundaries=boundaries, checkpoint_costs=checkpoint_costs,
        actual_invocation_costs=dict(receipts=cpu_receipts,
            waited_child_CPU_seconds=sum(v['user_seconds'] + v['system_seconds'] for v in cpu_receipts),
            wall_seconds_including_startup_save_and_inference=sum(v['wall_seconds'] for v in cpu_receipts),
            allocated_GPU_hours_including_startup_save_and_inference=sum(v['wall_seconds'] for v in cpu_receipts)*world/3600,
            complete=bool(cpu_receipts),
            note='External waited-child CPU; do not add nested per-worker or native GPU meters again'),
        formal_projection=dict(scenes=formal_size, updates=formal_updates, expected_exposure=expected_exposure,
            complete_epochs=epochs, partial_epoch_full_batches=remainder, tail_updates=tail_updates,
            optimizer_wall_seconds=estimate_seconds, optimizer_GPU_hours=estimate_seconds*world/3600,
            caveat='Prefix cost estimate, not a full-population throughput guarantee; loading, final saves and evaluation separately metered'),
        precision=meta['precision'], BF16_inference_profile=inference,
        limitation='80-update engineering/cost check; no new planning evidence, no claim of bitwise uninterrupted numerical replay',
        resources=resources)
    atomic_json(args.output, report)
    print({k: report[k] for k in ('passed', 'actual_updates', 'scene_exposure', 'stable_updates', 'formal_projection')})


if __name__ == '__main__':
    main()
