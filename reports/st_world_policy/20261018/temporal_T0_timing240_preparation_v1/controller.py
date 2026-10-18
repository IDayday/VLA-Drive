#!/usr/bin/env python3
"""Fixed fresh T0 timing240 binding; independent reviews required before prepare/run.

Reuses the pinned capacity supervisor and original native trainer. No optimizer,
model, sampler, checkpoint or scientific loss implementation is replaced.
"""
import hashlib
import json
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
CAPACITY = HERE.parent / 'temporal_T0_profile4_actual_binding_v1/controller.py'
CAPACITY_SHA = 'e2e6fd89c9c18ef12a3c067a1c3366c1868acafb72e8be6b07f7980ced3790b9'
INPUT_SHA = 'f644c9c46be5e5918db909446674f2e3b01d443314b45e53c8fced765718dc2a'
EVALUATION_RESERVE = 4 << 30
SETTLEMENT = Path('/mnt/project/VLA-Drive-st-spatial20k-efficiency-20261018/reports/st_world_policy/20261018/T0_capacity_settlement_and_timing_budget_v1/SETTLEMENT_AND_PROPOSAL.json')
SETTLEMENT_SHA = 'b99b5ec90bf789c48971095385ac8ace8c37e661905b2d848bff76aac9a93f27'


def adapted_capacity(raw):
    """Exact-count substitutions are the entire invocation adaptation surface."""
    changes = [
        ("BASE=ROOT/'profile/temporal_global32_qualification_v1/T0_capacity4'",
         "BASE=ROOT/'profile/temporal_global32_qualification_v1/T0_timing240'", 1),
        ("INPUT=BASE/'INPUTS.json'", "INPUT=HERE/'prepared_inputs/INPUTS.json'", 1),
        ("INPUT_SHA='940f30904a84277dff8e27b4813100758bd114f27600676ce4cc3b1d2ef79d51'",
         "INPUT_SHA='" + INPUT_SHA + "'", 1),
        ("FACTORY=BASE/'metadata/controls/RESULT.json'",
         "FACTORY=ROOT/'profile/temporal_global32_qualification_v1/T0_capacity4/metadata/controls/RESULT.json'", 1),
        ("RUN_ID='T0_fulltrain_seed42_profile4_world4_micro1_global32_v1'",
         "RUN_ID='T0_fulltrain_seed42_profile240_world4_micro1_global32_v1'", 1),
        ('stop_after=4,', 'stop_after=240,', 1),
        ('cumulative4 updates/128 exposures', 'cumulative240 updates/7680 exposures', 1),
        ('capacity only; no formal permission', 'timing only; no formal permission', 1),
        ("\"'max_completed_updates':4,'max_total_scene_exposures':128,'sample_limit':101592\"",
         "\"'max_completed_updates':240,'max_total_scene_exposures':7680,'sample_limit':101592\"", 1),
        ("\"result.get('completed')==4 and result.get('exposure')==128\"",
         "\"result.get('completed')==240 and result.get('exposure')==7680\"", 1),
        ("\"'updates':4,'total_exposures':128,'world_size':4,'sample_limit':101592\"",
         "\"'updates':240,'total_exposures':7680,'world_size':4,'sample_limit':101592\"", 1),
        ('do not claim4 completed', 'do not claim240 completed', 1),
        ("'/root/temporal_profile_prepare'", "'/root/temporal_t0_timing_prepare'", 2),
        ('RESERVE=80<<30', 'RESERVE=40<<30', 1),
        ('checkpoint_reservation_bytes=80<<30', 'checkpoint_reservation_bytes=40<<30', 1),
        ("rt['save_every']==4", "rt['save_every']==240", 1),
        ("rt['profile_mode']=='none'", "rt['profile_mode']=='timing'", 1),
        ('Fresh T0 capacity only; no resume admitted', 'Fresh T0 timing only; no resume admitted', 1),
        ('Changed capacity runtime', 'Changed timing runtime', 1),
        ('st_T0_global32_profile4_resource_admission_v1', 'st_T0_global32_timing240_resource_admission_v1', 1),
        ('exact four-step capacity', 'exact fresh240 timing', 1),
        ('actual checkpoint/cache/metadata obligations + 1TiB floor + own remaining80GiB',
         'actual checkpoint/cache/metadata obligations + 1TiB floor + own remaining40GiB plus retained evaluation4GiB', 1),
        ('inclusive80GiB', 'inclusive40GiB', 1),
    ]
    for old, new, count in changes:
        if raw.count(old) != count:
            raise ValueError('Pinned capacity adaptation changed: ' + old)
        raw = raw.replace(old, new)
    return raw


def controller():
    if CAPACITY.stat().st_size > 32 << 10:
        raise ValueError('Only the bounded pinned capacity supervisor is inherited')
    raw = CAPACITY.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CAPACITY_SHA:
        raise ValueError('Capacity source identity changed')
    module = ModuleType('fixed_T0_timing240_binding')
    module.__file__ = str(HERE / 'controller.py')
    exec(compile(adapted_capacity(raw.decode()), str(CAPACITY), 'exec'), module.__dict__)
    c = module.controller()
    # Preserve every scientific field, not just the quick scalar checks in the
    # capacity wrapper. The native validator additionally resolves the teacher.
    original_recipe = c.validate_recipe
    expected_cfg = c.read(c.read(c.INPUT)['pins']['config']['path'])
    def validate_recipe(cfg):
        original_recipe(cfg)
        c.require(cfg == expected_cfg, 'Timing changes a field outside the locked metadata recipe')
    c.validate_recipe = validate_recipe
    original_admission = c.admission
    def admission(path, sha, action):
        approved, pin = original_admission(path, sha, action)
        c.require(approved.get('retained_evaluation_reservation_bytes') == EVALUATION_RESERVE,
                  'Independent C-evaluation4GiB obligation must remain reserved')
        c.require(approved.get('maximum_new_checkpoint_directories') == 1
                  and approved.get('capacity_reservation_settlement') == dict(path=str(SETTLEMENT), sha256=SETTLEMENT_SHA),
                  'One checkpoint and original closed capacity reservation required')
        c.require(c.digest(SETTLEMENT) == SETTLEMENT_SHA, 'Capacity settlement changed')
        settled = c.read(SETTLEMENT)
        c.require(settled.get('status') == 'SETTLED_NO_FURTHER_EXECUTION'
                  and settled.get('remaining_future_output_commitment_bytes') == 0
                  and settled.get('files_preserved') is True
                  and settled.get('no_more_resume_or_retry') is True
                  and settled.get('scientific_checkpoint_gross_bytes') == 4143 << 30,
                  'No old checkpoint or scientific reservation may be removed')
        review_pin = approved.get('correctness_review', {})
        c.require(set(review_pin) == {'path', 'sha256'} and c.digest(review_pin['path']) == review_pin['sha256'],
                  'Actual independent correctness review required')
        review = c.read(review_pin['path'])
        expected = dict(schema='st_T0_timing240_correctness_review_v1', status='PASS',
                        source_sha=module.SOURCE, controller_sha256=c.digest(HERE / 'controller.py'),
                        launcher_sha256=c.digest(HERE / 'launch_once.sh'), inputs_sha256=INPUT_SHA)
        c.require(all(review.get(k) == v for k, v in expected.items()), 'Correctness review does not bind actual timing pins')
        c.require(isinstance(review.get('reviewer'), str) and review['reviewer'] not in
                  ('', '/root', '/root/temporal_t0_timing_prepare'), 'Independent correctness author required')
        return approved, pin
    c.admission = admission
    original_requirement = c.storage_requirement
    c.storage_requirement = lambda used: original_requirement(used) + EVALUATION_RESERVE
    original_storage = c.storage
    def storage():
        slots = c.OUT / 'runs' / c.RUN_ID / 'checkpoints'
        if slots.exists():
            c.require(not slots.is_symlink(), 'Checkpoint directory may not alias another run')
            dirs = [p for p in slots.iterdir() if p.is_dir()]
            c.require(len(dirs) <= 1, 'Only one checkpoint directory, including partial writes, is admitted')
        value = original_storage()
        value['retained_evaluation_reservation_bytes'] = EVALUATION_RESERVE
        return value
    c.storage = storage
    # The inherited common() closes over its original scalar validator. Require
    # full config equality here too before it can produce or dispatch a run.
    original_common = c.common
    def common():
        inputs, cfg = original_common()
        validate_recipe(cfg)
        c.require(inputs.get('status') == 'TIMING_METADATA_BOUND_NOT_ADMITTED'
                  and inputs.get('timing_plan') == dict(warmup=20, steady=100, windows=2, updates=240),
                  'Timing window identity mismatch')
        return inputs, cfg
    c.common = common
    return c


if __name__ == '__main__':
    controller().main()
