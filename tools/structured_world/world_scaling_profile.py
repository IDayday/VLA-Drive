"""Admission for a finite fresh 8/16-rank throughput probe, never resumption.

Reuses the trainer, optimizer and BoundaryTimer. There is deliberately no
ZeRO repartitioning or claim that changing rank-local RNG preserves training.
"""


def validate_args(args):
    if not args.world_scaling_profile:
        return
    if not (args.scope == 'profile' and args.use_formal_schedule
            and args.benchmark_no_end_save and 0 < args.debug_updates <= 220
            and not args.resume and not args.resume_origin_run
            and not args.pilot_origin_run and not args.pilot_origin_tag
            and not args.capture_updates and not args.capture_storage_budget
            and not args.debug_scenes and not args.stop_after
            and args.geometry and args.geometry_identity):
        raise ValueError('World scaling requires a fresh finite full-population profile, '
                         'common geometry/formal schedule, no resume/captures/end save')
    if args.execution_mode not in ('io_preserving_v1', 'engineering_host_fast'):
        raise ValueError('Do not combine world scaling with math/precision candidates')


def validate_ranks(args, *, rank, world, local, local_world):
    if not args.world_scaling_profile:
        if world != 8 or args.micro_batch not in (1, 2, 4):
            raise ValueError('One complete eight-GPU host and registered microbatch required')
        return
    validate_args(args)
    if (world not in (8, 16) or local_world != 8 or not 0 <= rank < world
            or not 0 <= local < 8 or rank % 8 != local
            or args.micro_batch != 32 // world):
        raise ValueError('World scaling requires full8 nodes: 8x4 or 16x2, global32, '
                         'one unchanged optimizer batch without extra accumulation')


def contract(args, world):
    if not args.world_scaling_profile:
        return None
    return {
        'schema': 'structured_world_scaling_profile_v1',
        'numerical_level': 'E3_distributed_execution',
        'purpose': 'finite full-model throughput screen, not algorithm/quality acceptance',
        'world_size': world, 'local_world_size': 8, 'nodes': world // 8,
        'global_batch': 32, 'microbatch': args.micro_batch,
        'optimizer_updates_limit': args.debug_updates,
        'initialization': 'common general Qwen/random driving modules and shared geometry',
        'native_checkpoint_restored': False, 'formal_authorized': False,
        'RNG': 'unchanged rank-seeded generators; assignment to scenes differs across world sizes',
        'SyncBN': 'same global population; rank partition and reduction rounding differ',
        'ZeRO': 'fresh optimizer in each profile; no eight-rank state loaded on sixteen ranks',
        'save_policy': 'no initial/milestone/end checkpoint; explicit controlled stop retains trainer safety',
        'acceptance_limit': 'throughput cannot establish lossless migration or planning noninferiority',
    }
