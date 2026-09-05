#!/usr/bin/env python3
"""Lock the user-requested dual-node GB64 layout only from real Lite evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('metrics','real-smoke','probe','input-manifest','shared-init','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    m=json.loads(a.metrics.read_text());smoke=json.loads(a.real_smoke.read_text());probe=json.loads(a.probe.read_text())
    manifest=json.loads(a.input_manifest.read_text())
    assert m['status']=='success' and m['global_batch_size']==64 and m['gpu_count']==16
    assert m['timed_optimizer_steps']>=300
    assert not m.get('oom') and not m.get('deadlock') and m.get('nonfinite_count',0)==0
    assert m['peak_allocated_gib']<72 and m['peak_reserved_gib']<76
    assert smoke['status']=='PASS' and smoke['current_only_export_max_abs_diff']==0
    assert smoke['paired_trainable_initial_state_bitwise_equal']
    assert probe['log_disjoint'] and probe['steps']>=100
    assert manifest['record_count']==103288 and manifest['protocol_version']=='task_future_lite'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert smoke['shared_init_sha256']==sha(a.shared_init)
    steps=math.ceil(manifest['record_count']/64)
    lrs=dict(planning_adapter=2e-4,semantic_fusion=2e-4,action_generator=2e-4,scorer=2e-4,
             future_predictor=1e-4,semantic_qformer=1e-4,vision_qv_lora=3e-5)
    lock=dict(schema_version=2,protocol_version='task_future_lite',selected_layout='16x4',
        gpu_count=16,per_gpu_batch_size=4,global_batch_size=64,num_nodes=2,devices_per_node=8,
        num_workers_per_rank=m['num_workers_per_rank'],scorer_processes_per_rank=4,scorer_partitions_per_scene=1,
        gradient_checkpointing=True,read_only_attention_backend='eager',
        lr_scale_multiplier=1.,logical_peak_learning_rates=lrs,
        ema_actual_start_momentum=.996**4,ema_actual_end_momentum=.9999**4,
        dataset_length=103288,dataset_epochs=27,steps_per_epoch=steps,total_steps=steps*27,
        sampler_padding_per_epoch=steps*64-103288,shared_between_base_and_vqa=True,
        train_only_pilot_locked=True,full_physical_sidecar_smoke_passed=True,
        selection_scope='User requested two 16-GPU runs; tested GB64, not a claim of globally optimal throughput or convergence',
        observed_samples_per_second=m['samples_per_second'],estimated_27_epoch_hours=steps*64*27/m['samples_per_second']/3600,
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        benchmark_metrics_sha256={'16x4':sha(a.metrics)},
        evidence={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in
                  [('real_smoke',a.real_smoke),('train_only_probe',a.probe),('input_manifest',a.input_manifest),('shared_init',a.shared_init)]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(lock,indent=2))


if __name__=='__main__':main()
