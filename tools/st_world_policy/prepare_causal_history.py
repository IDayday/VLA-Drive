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
    'tools.st_world_policy.history_shards',
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
    layout=getattr(args,"record_layout","files")
    if args.action!='build' or args.pin_selection or not args.scene_list:
        raise ValueError('Exact existing scene list; build only, no population reselection')
    if args.target_times!=[0.] or args.mode!='registered' or args.max_scenes<1 or (layout=='files' and args.max_scenes>256):
        raise ValueError('Current-only registered cost pilot <=256; full build not yet admitted')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be hidden')
    if storage._ACTIVE_GUARD is None:
        raise ValueError('Explicit original-ledger metadata admission required')
    guard=storage._ACTIVE_GUARD
    if layout=='packed_log_v1':
        reg=guard.registration
        if reg.get('record_layout')!=layout or reg.get('inventory_strategy')!='immutable_history_shards_v1' or reg.get('max_scenes')!=args.max_scenes:raise ValueError('Exact packed population/strategy registration required')
        cpus=reg.get('cpu_affinity')
        if not isinstance(cpus,list) or not 1<=len(cpus)<=2 or not set(os.sched_getaffinity(0))<=set(cpus):raise ValueError('Explicit CPU-only topology required')
        for key in ('max_source_read_bytes','read_mib_per_second','max_history_shards'):
            if type(reg.get(key)) is not int or reg[key]<=0:raise ValueError('Explicit packed IO/shard budgets required')
        if reg['read_mib_per_second']>32 or reg['max_source_read_bytes']>2*(1<<40):raise ValueError('Packed IO budget exceeds supported bounds')
        return
    if guard.registration.get('record_layout') not in (None,'files') or guard.registration.get('inventory_strategy') is not None:
        raise ValueError('Files command cannot claim packed layout or census strategy')
    if not set(os.sched_getaffinity(0))<= {6,7}:raise ValueError('Pilot CPU6,7 only')
    if (guard.limits['max_disk_bytes']>LIMITS['max_metadata_bytes'] or
            guard.limits['max_seconds']>LIMITS['max_seconds'] or
            guard.limits['max_rss_bytes']>LIMITS['max_rss_bytes']):
        raise ValueError('Registration exceeds pilot cap')


def run(args):
    validate(args)
    limits=dict(LIMITS);packed=None
    if args.record_layout=='packed_log_v1':
        reg=storage._ACTIVE_GUARD.registration
        limits.update(max_seconds=reg['limits']['max_seconds'],max_rss_bytes=reg['limits']['max_rss_bytes'],max_metadata_bytes=reg['limits']['max_disk_bytes'],max_source_read_bytes=reg['max_source_read_bytes'],read_mib_per_second=reg['read_mib_per_second'])
    budget=bounded.PreparationBudget(Path(args.output),limits)
    if args.record_layout=='packed_log_v1':
        from tools.st_world_policy.history_shards import HistoryShards
        packed=HistoryShards(args.output,budget,max_shards=reg['max_history_shards'])
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
        if packed is not None:
            if Path(path).parent==Path(args.output)/'history':return packed.add(path,value)
            if Path(path).name=='HISTORY_MANIFEST.json':
                transformed=packed.manifest(value)
                # The original builder uses this identity in its final receipt.
                value.clear();value.update(transformed)
        return bounded.write_json(path,value,budget)
    with ExitStack() as stack:
        for module,name,value in (
            (original,'file_sha256',lambda p:packed.digest(p) if packed is not None else bounded.budget_sha(p,budget)),
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
    p.add_argument('--record-layout',choices=['files','packed_log_v1'],default='files')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    print(json.dumps(run(args),sort_keys=True))
