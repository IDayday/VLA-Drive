"""Summarize actual finite B/D profiling evidence; never dispatch or modify runs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics


def build(root):
    def load(name):return json.loads((root/name).read_text())
    production={r['architecture']:r for r in load('FINAL_PRODUCTION_WINDOW.json')}
    volumes={r['arch']:r for r in load('GRADIENT_VOLUME.json')}
    rows=[]
    for arch in ('B','D'):
        p=production[arch];window=p['last64'];micro=load(arch+'_MICRO_RESULT_attempt002/RESULT.json')
        worker=load(arch+'_MICRO_WORKER_attempt002/RESULT.json');snapshot=load(arch+'_SNAPSHOT.json')
        seconds=statistics.mean(r['slowest_rank_seconds']for r in window)
        if len(window)!=64 or len(set(p['rank_last_updates']))!=1:raise ValueError('Incomplete production window')
        if not load(arch+'_MODEL_HASH_VERIFIED.json')['identical']:raise ValueError('Model snapshot hash mismatch')
        rows.append(dict(architecture=arch,production_updates=p['rank_last_updates'][0],world_size=8,
            sampled_at_utc=datetime.fromtimestamp(p['at_unix'],timezone.utc).isoformat(),
            production_window_updates=[window[0]['updates'],window[-1]['updates']],
            production_mean_update_seconds=seconds,data_wait_mean_seconds=statistics.mean(r['data_seconds']for r in window),
            remaining_active_training_hours=(100000-p['rank_last_updates'][0])*seconds/3600,
            eta_excludes='checkpoint/export/evaluation/contention/downtime; extrapolation, not a completion promise',
            production_micro=snapshot['identity']['execution']['micro'],
            production_accumulation=snapshot['identity']['execution']['accum'],
            optimizer_global_steps=p['rank0_last']['optimizer_global_steps'],optimizer_skipped_steps=p['rank0_last']['optimizer_skipped_steps'],
            FP32_gradient_GiB=volumes[arch]['GiB'],trainable_parameters=volumes[arch]['trainable_parameters'],
            layout_probe=dict(snapshot_update=snapshot['updates'],model_sha256=snapshot['model_sha256'],
                old_micro=micro['old_micro'],new_micro=micro['new_micro'],status=micro['status'],
                rank_slice_speedup=micro['rank_slice_speedup'],rank_slice_time_reduction=1-1/micro['rank_slice_speedup'],
                median_seconds=micro['median_rank_slice_seconds'],gradient=micro['gradient_comparison'],
                thresholds=micro['thresholds'],parameters_unchanged=micro['all_parameters_unchanged'],
                RNG_unchanged=all(r['RNG_unchanged']for r in micro['rows']),
                peak_allocated_GiB=max(r['peak_allocated_GiB']for r in micro['rows']),
                worker_wall_seconds=worker['end_unix']-worker['start_unix'],optimizer_updates=0,scorer_calls=0,
                production_migrated=False,caveat=micro['caveat'])))
    return dict(schema='bd_parallel_profile_summary_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        scientific_source='5a2d5b50335e85fd84049ab1f1e5ebe34f537b4e',
        formal_scientific_files_unchanged=load('SOURCE_AUDIT.json')['all_identical'],
        rows=rows,TCP={a:load('TCP_PARALLEL_'+a+'_CLIENT.json')['rows']for a in('B','D')},
        NCCL=dict(ranks=[load('NCCL/rank0.json'),load('NCCL/rank1.json')],
            transport='Socket',GPU_Direct_RDMA=False,scope='Two spare hosts, one GPU each; not B/D sixteen-rank model training'),
        RDMA=dict(user_authorized=True,platform_pod_permissions=load('RDMA_API_CAPABILITIES_DNS.json')['checks'],
            preflight_files=[str(p.relative_to(root))for p in root.glob('RDMA_*PREFLIGHT.json')],
            container_device_nodes_and_libraries_ready=False,platform_action_required=True),
        decision='Keep qualified world8/micro layout; await actual RDMA and native optimizer redistribution qualification before topology migration',
        excluded_interpretations=['No measured 16-GPU model speedup','No production microbatch speedup applied',
            'Gradient differences are not Navtest performance estimates','No optimizer reset or algorithm change','No new training or inference seeds'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=build(a.assets);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),architectures=[r['architecture']for r in result['rows']],production_migrated=False)))


if __name__=='__main__':main()
