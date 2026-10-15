"""Read-only accounting of the first full A-only training epoch.

Uses native optimizer records, without model loading, checkpoint selection,
additional evaluation, or changes to the running trainer. Sampler membership is
reconstructed from its pinned source; it is not a per-file IO trace.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,identity,read,source_sha
from tools.decision_local.summarize_a_base_profile import timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','registration','population-contract','legacy-source','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();resources=check_host()
    if a.output.exists():raise FileExistsError('Preserve first-epoch evidence')
    reg=read(a.registration);meta=read(a.run/'identity.json');pop=read(a.population_contract)
    if (meta['scope']!='formal' or meta['source_sha']!=reg['training_source_sha'] or
        meta['registration_sha256']!=digest(a.registration) or meta['data']['population_contract']!=pop or
        digest(a.population_contract)!=reg['population_contract_sha256']):
        raise ValueError('Not the registered formal A-only base')
    for key in ('scene_count','global_batch','micro_batch','world_size','updates'):
        if meta[key]!=reg[key]:raise ValueError('Changed formal execution contract: '+key)
    size,batch,world=reg['scene_count'],reg['global_batch'],reg['world_size']
    n=math.ceil(size/batch)
    complete_lines=[x for x in (a.run/'steps.jsonl').read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]
    if len(complete_lines)<n+1:raise ValueError('First epoch and following update have not completed')
    lines=complete_lines[:n+1];rows=[json.loads(x) for x in lines]
    if [r['update'] for r in rows]!=list(range(1,n+2)):raise ValueError('Missing/duplicate update')
    exposure=0
    for row in rows:
        epoch,offset=divmod(row['update']-1,n);actual=min(batch,size-offset*batch);exposure+=actual
        if (row['epoch']!=epoch or row['offset']!=offset+1 or row['exposure']!=exposure or
            row['counts']['ego_scenes']!=actual or not row['fp32_master_update_verified'] or
            row['sampled_changed_elements']<=0 or not math.isfinite(row['grad_norm'])):
            raise ValueError('Invalid exposure, native cursor or master update')
        if (not {'ego_fm','current_dino','future_clip','interaction'}<=row['losses'].keys() or
            not all(math.isfinite(x) for x in row['losses'].values()) or len(row['per_rank_profile'])!=world):
            raise ValueError('Missing/nonfinite complete objective or rank profile')
    legacy=a.legacy_source
    if source_sha(legacy)!=reg['legacy_source_sha']:raise ValueError('Changed native sampler runtime')
    sampler=legacy/'tools/ddpolicy_vehicle/training_state.py'
    spec=importlib.util.spec_from_file_location('_native_epoch_sampler',sampler)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    batches=module.epoch_batches(size,batch,meta['config']['seed'],0)
    indices=[i for b in batches for i in b];index=read(pop['index'])
    if (len(index)!=size or sorted(indices)!=list(range(size)) or identity(index)!=pop['index_identity'] or
        meta['selected_index_hash']!=pop['index_identity']):raise ValueError('Invalid one-epoch population')
    stable=rows[20:n];periods=[row['ended_unix']-rows[row['update']-2]['ended_unix'] for row in stable]
    if any(x<=0 for x in periods):raise ValueError('Non-monotonic optimizer clock')
    loss_windows={}
    for name,window in [('first50',rows[:50]),('last50',rows[n-50:n])]:
        loss_windows[name]={k:statistics.mean(row['raw_losses'][k] for row in window)
            for k in ('ego_fm','current_dino','future_clip','interaction')}
    report=dict(schema='A_only_first_epoch_observation_v1',source_sha=source_sha(ROOT),
        training_identity=meta['identity'],training_source_sha=meta['source_sha'],
        registration_sha256=digest(a.registration),population_contract_sha256=digest(a.population_contract),
        steps_prefix_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),prefix_records_used=n+1,
        complete_epoch_scenes=size,complete_epoch_updates=n,
        tail_scenes=len(batches[-1]),tail_rank_sizes=[len(batches[-1][rank::world]) for rank in range(world)],
        epoch_boundary=[{k:row[k] for k in ('update','epoch','offset','exposure','counts','lr')} for row in rows[n-2:n+1]],
        exact_exposure_and_native_cursor=True,all_four_losses_finite=True,all_sampled_FP32_master_updates_verified=True,
        sampler_source_sha256=digest(sampler),sampler_unique_indices=len(set(indices)),
        sampler_coverage_scope='Pinned deterministic sampler reconstruction and native count/cursor checks, not an independent file-access trace',
        stable_warmup_excluded_updates=20,wall_optimizer_periods=timing(periods),
        slowest_rank_data_seconds=timing([max(rank[1] for rank in row['per_rank_profile']) for row in stable]),
        per_rank_peak_allocated_bytes=[max(row['per_rank_profile'][rank][2] for row in rows[:n]) for rank in range(world)],
        per_rank_peak_reserved_bytes=[max(row['per_rank_profile'][rank][3] for row in rows[:n]) for rank in range(world)],
        raw_loss_windows=loss_windows,resources=resources,
        limitations=['Only the first formal training epoch; not a completed base or planning efficacy result',
            'Loss windows contain different training scenes and different warmup stages, not a paired evaluation',
            'Observed periods include native logging/host gaps; no checkpoint has yet occurred in this prefix',
            'No base selection, B/dev/Navtest input or new model execution'])
    atomic_json(a.output,report)
    print(json.dumps({k:report[k] for k in ('complete_epoch_scenes','complete_epoch_updates','tail_scenes','wall_optimizer_periods')}))


if __name__=='__main__':main()
