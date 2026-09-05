#!/usr/bin/env python3
"""Select a measured Lite layout with an equal-exposure training-loss guard.

This is a bounded engineering screen, NOT a test of final PDMS non-inferiority.
No Navtest labels, early stopping, model simplification, or automatic LR search.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def tail_metrics(metrics, exposure=4096):
    batch=int(metrics['global_batch_size'])
    rows=metrics['training_curve_global_rank_mean']
    n=math.ceil(exposure/batch)
    if len(rows)<n:
        raise ValueError('Insufficient pilot sample exposure')
    rows=rows[-n:]
    keys=set.intersection(*(set(row) for row in rows))
    return {key:statistics.fmean(row[key] for row in rows)
            for key in sorted(keys) if key.startswith(('loss/','gradient/'))}


def compare_candidate(control, candidate):
    reasons=[]
    if candidate.get('status')!='success' or candidate.get('oom') or candidate.get('deadlock') or candidate.get('nonfinite_count',1):
        return dict(eligible=False,reasons=['incomplete/OOM/deadlock/nonfinite'])
    if candidate['peak_allocated_gib']>=72 or candidate['peak_reserved_gib']>=76:
        reasons.append('memory reserve gate')
    if candidate.get('read_only_attention_backend')!='split_sdpa' or candidate.get('gradient_checkpointing') is not False:
        reasons.append('unexpected compute configuration')
    if candidate.get('scorer_partitions_per_scene')!=1:
        reasons.append('candidate groups must remain intact')
    for key in ('sample_exposure_count_including_warmup','sample_exposure_multiset_sha256'):
        if not candidate.get(key) or candidate[key]!=control.get(key):
            reasons.append('unequal sample exposure: '+key)
    if candidate.get('sample_exposure_count_including_warmup',0)<20480:
        reasons.append('pilot needs >=20,480 sample presentations')
    c, m=tail_metrics(control),tail_metrics(candidate)
    ratios={key:m[key]/max(c[key],1e-12) for key in ('loss/trajectory_loss','loss/final_score_loss')}
    # Predeclared gross-regression guard; not a statistical guarantee of a
    # generalization gain. All raw curves and official train-only diagnostics
    # are retained, including unfavorable WM/component/selected-score changes.
    for key,ratio in ratios.items():
        if not math.isfinite(ratio) or ratio>1.10:
            reasons.append(f'{key} tail ratio {ratio:.4f} exceeds 1.10 screen')
    if candidate['p90_step_time']>1.35*candidate['median_step_time']:
        reasons.append('unstable step-time tail')
    return dict(eligible=not reasons,reasons=reasons,tail_exposure=4096,
                trajectory_scorer_ratios=ratios,tail_metrics=m,
                samples_per_second=candidate['end_to_end_samples_per_second'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('metrics-root','reference-lock','attention-parity','output','comparison-output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--control',default='16x4')
    a=p.parse_args()
    if a.output.exists() or a.comparison_output.exists():
        raise FileExistsError('Never overwrite a layout lock or pilot report')
    metrics={path.parent.name:json.loads(path.read_text()) for path in a.metrics_root.glob('*/metrics.json')}
    control=metrics[a.control]
    assert control['status']=='success' and control['global_batch_size']==64
    parity=json.loads(a.attention_parity.read_text())
    assert parity['status']=='PASS' and len(parity['blocks'])==24
    decisions={name:compare_candidate(control,m) for name,m in metrics.items()}
    eligible=[name for name,result in decisions.items() if result['eligible']]
    if not eligible:
        raise RuntimeError('No eligible measured layout: '+json.dumps(decisions))
    fastest=max(eligible,key=lambda name:metrics[name]['end_to_end_samples_per_second'])
    # At essentially equal wall time retain the smaller global batch.
    close=[name for name in eligible if metrics[name]['end_to_end_samples_per_second']>=.95*metrics[fastest]['end_to_end_samples_per_second']]
    selected=min(close,key=lambda name:(metrics[name]['global_batch_size'],-metrics[name]['end_to_end_samples_per_second']))
    comparison=dict(scope='train-only equal sample exposure, initial warmup; not final-score non-inferiority',
                    control=a.control,selected=selected,decisions=decisions,
                    fixed_model_loss_and_data=True,final_pdms_noninferiority='NOT_ESTABLISHED',
                    full_formal_endpoint_epochs=27)
    a.comparison_output.parent.mkdir(parents=True,exist_ok=True)
    a.comparison_output.write_text(json.dumps(comparison,indent=2)+'\n')
    reference=json.loads(a.reference_lock.read_text())
    for name,item in reference['evidence'].items():
        assert sha(item['path'])==item['sha256'], name
    m=metrics[selected]; batch=int(m['global_batch_size']); scale=math.sqrt(batch/64)
    assert batch in (64,128)
    params_path=a.metrics_root.parent/'benchmark'/selected/'run_metadata/training_parameter_audit.json'
    params=json.loads(params_path.read_text())
    assert params['actual_global_batch']==batch and params['world_model_enabled'] and not params['agent_checkpoint_loaded']
    lrs=params['logical_peak_learning_rates']
    steps=math.ceil(103288/batch)
    lock={**reference,'selected_layout':selected,'gpu_count':m['gpu_count'],
        'per_gpu_batch_size':m['per_gpu_batch_size'],'global_batch_size':batch,
        'num_nodes':m['gpu_count']//8,'num_workers_per_rank':m['num_workers_per_rank'],
        'scorer_processes_per_rank':m['scorer_processes_per_rank'],'scorer_partitions_per_scene':1,
        'gradient_checkpointing':False,'read_only_attention_backend':'split_sdpa',
        'lr_scale_multiplier':scale,'logical_peak_learning_rates':lrs,
        'ema_actual_start_momentum':.996**(batch/16),'ema_actual_end_momentum':.9999**(batch/16),
        'steps_per_epoch':steps,'total_steps':steps*27,'sampler_padding_per_epoch':steps*batch-103288,
        'observed_samples_per_second':m['end_to_end_samples_per_second'],
        'estimated_27_epoch_hours':steps*batch*27/m['end_to_end_samples_per_second']/3600,
        'selection_scope':'User-authorized Base-only acceleration; measured equal-exposure guard, no guarantee of final PDMS equivalence',
        'active_formal_variants':['base'],'vqa_paused_by_user':True,
        'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'benchmark_metrics_sha256':{name:sha(a.metrics_root/name/'metrics.json') for name in metrics},
        'training_parameter_audit':dict(path=str(params_path.resolve()),sha256=sha(params_path)),
        'attention_parity':dict(path=str(a.attention_parity.resolve()),sha256=sha(a.attention_parity)),
        'batch_comparison':dict(path=str(a.comparison_output.resolve()),sha256=sha(a.comparison_output)),
        'throughput_scope':'Measured full optimizer-cycle including loader wait and callback overhead',
        'callback_step_only_samples_per_second':m['samples_per_second']}
    # These refer to the superseded eager benchmark and must not be inherited.
    lock.pop('event_file_sha256',None);lock.pop('end_to_end_window',None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,global_batch=batch,samples_per_second=lock['observed_samples_per_second'],estimated_hours=lock['estimated_27_epoch_hours']),indent=2))


if __name__=='__main__':
    main()
