"""Finite history metadata cost pilot; original math, no teacher/cache/trainer.

Must run through metadata_storage and an independently admitted owned CPU
supervisor. No direct admission, ledger migration, full-scale or launch API.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

from tools.st_world_policy import prepare_history as original
from tools.st_world_policy import prepare_future_metadata as bounded
from tools.st_world_policy import metadata_storage as storage
from starVLA.dataloader import st_world_history as history

METADATA_STORAGE_CALLBACK_PROTOCOL = 'st_metadata_storage_check_v1'
METADATA_STORAGE_CALLBACK_DEPENDENCIES = (
    'tools.st_world_policy.prepare_history',
    'tools.st_world_policy.prepare_future_metadata',
    'starVLA.dataloader.st_world_history',
    'starVLA.dataloader.st_world_metadata_index',
    'starVLA.dataloader.st_world_future_provenance',
    'tools.st_world_policy.prepare_current_teacher',
)
MiB=1<<20
LIMITS=dict(max_seconds=900,max_rss_bytes=4<<30,max_metadata_bytes=32*MiB,
            max_source_read_bytes=16<<30,read_mib_per_second=32)


def validate(args):
    if args.action!='build' or args.pin_selection or not args.scene_list:
        raise ValueError('Exact existing scene list; build only, no population reselection')
    if args.target_times!=[0.] or args.mode!='registered' or not 1<=args.max_scenes<=256:
        raise ValueError('Current-only registered cost pilot <=256; full build not yet admitted')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not set(os.sched_getaffinity(0))<= {6,7}:
        raise ValueError('CPU6,7 only; CUDA must be hidden')
    if storage._ACTIVE_GUARD is None:
        raise ValueError('Explicit original-ledger metadata admission required')
    guard=storage._ACTIVE_GUARD
    if (guard.limits['max_disk_bytes']>LIMITS['max_metadata_bytes'] or
            guard.limits['max_seconds']>LIMITS['max_seconds'] or
            guard.limits['max_rss_bytes']>LIMITS['max_rss_bytes']):
        raise ValueError('Registration exceeds pilot cap')


def run(args):
    validate(args)
    budget=bounded.PreparationBudget(Path(args.output),LIMITS)
    old_reference=history.image_reference
    def reference(path):
        # Conservative header allowance: hash charges actual chunks separately.
        # PIL header access can never consume more than one whole file here.
        budget.read_allowance(Path(path).stat().st_size,'image_header_upper_bound')
        return old_reference(path)
    def read_json(path):
        with bounded.BudgetedReader(path,budget) as stream:
            return json.loads(stream.read())
    def write(path,value,*,immutable=False):
        if not immutable: raise ValueError('Only immutable history artifacts')
        return bounded.write_json(path,value,budget)
    with ExitStack() as stack:
        for module,name,value in (
            (original,'file_sha256',lambda p:bounded.budget_sha(p,budget)),
            (history,'file_sha256',lambda p:bounded.budget_sha(p,budget)),
            (original,'image_reference',reference),(history,'image_reference',reference),
            (original,'read_json',read_json),
            (original,'open_raw',lambda p:bounded.BudgetedReader(p,budget)),
            (original,'atomic_json',write)):
            stack.enter_context(patch.object(module,name,value))
        result=original.run(args)
    bounded.write_json(Path(args.output)/'RESOURCE_COST.json',budget.snapshot(),budget)
    return dict(result,history_resource_cost=budget.snapshot(),full_scale_qualified=False)


def parser():
    p=original.parser()
    p.add_argument('action',choices=['build'])
    return p


if __name__=='__main__':
    args=parser().parse_args()
    print(json.dumps(run(args),sort_keys=True))
