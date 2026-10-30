"""Read-only label fit, full-Test pairing and saved TRAIN-output error audit.

No model forward, optimization, simulator call or new candidate generation.
Consumes the sealed journal analysis produced by analyze_il_loss_pdms.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from tools.qwen_world_postcot.analyze_il_loss_pdms import correlation
from tools.qwen_world_postcot.summarize import load, compare, METRICS
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank


OLD = Path('/mnt/project/qwen-world-postcot-20261025')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def curve_comparison(root):
    arms = {name: read(root/name/'analysis.json') for name in ('original', 'optimized')}
    tables = {name: {r['update']: r for r in a['points_by_window']['500']} for name, a in arms.items()}
    common = sorted(set(tables['original']) & set(tables['optimized']))
    matched = [dict(update=u, original_L1=tables['original'][u]['x'],
                    optimized_L1=tables['optimized'][u]['x'],
                    relative_L1_increase_percent=100*(tables['optimized'][u]['x']/tables['original'][u]['x']-1),
                    original_PDMS=tables['original'][u]['PDMS'], optimized_PDMS=tables['optimized'][u]['PDMS'])
               for u in common]
    stats = {}
    for name, table in tables.items():
        stats[name] = {}
        for minimum in (10000, 20000, 30000):
            rows = [r for u, r in table.items() if u >= minimum]
            stats[name][str(minimum)] = correlation([r['x'] for r in rows], [r['PDMS'] for r in rows])
    return dict(matched=matched, correlations=stats,
                definition='Trailing500 online TRAIN normalized clean-x L1, before coefficient10; not fixed-checkpoint ODE error',
                all8_rank_global_loss_replicas_checked_by_upstream_analysis=True,
                independent_training_seeds=1, levels_not_independent_checkpoint_replicates=True)


def paired_test(root):
    plans = {name: {r['update']: r for r in read(root/(name+'_scores.json'))} for name in ('original','optimized')}
    result = {}
    for u in (30000,35000,40000):
        arm = {}
        for name in plans:
            directory = Path(plans[name][u]['summary']).parent/'evaluation'
            arm[name] = load(directory)
        original_id, original = arm['original']; optimized_id, optimized = arm['optimized']
        paired = compare(optimized_id, optimized, original_id, original)
        deltas = np.array([optimized[t]['feedback']['metrics']['score']-original[t]['feedback']['metrics']['score'] for t in original])
        means = {name: {key: float(np.mean([r['feedback']['metrics'][key] for r in records.values()])) for key in METRICS}
                 for name, (_,records) in arm.items()}
        progress = {name: float(np.mean([r['feedback']['raw_progress_meters'] for r in records.values()]))
                    for name, (_,records) in arm.items()}
        result[str(u)] = dict(paired=paired, metrics_percent={name:{k:100*v for k,v in m.items()}for name,m in means.items()},
            mean_executed_raw_progress_m=progress,
            delta_PDMS_points=100*paired['mean_delta'], CI95_PDMS_points=[100*v for v in paired['CI95']],
            G_plus_points=100*float(np.maximum(deltas,0).mean()),G_minus_points=100*float(np.minimum(deltas,0).mean()))
    return result


def training_probe(root):
    records = pickle.load((OLD/'data/TRAIN/records.pkl').open('rb'))
    tokens = read(OLD/'optimized_labels_IL_20261028/targets/tokens.json')
    optimized = np.load(OLD/'optimized_labels_IL_20261028/targets/targets.npy',mmap_mode='r')
    original = np.stack([records[t]['requested_GT'] for t in tokens]).astype(np.float64)
    valid = np.stack([np.abs(np.cumsum(records[t]['raw_future_dt_seconds'])-np.arange(1,9)*.5)<.06 for t in tokens])
    difference = optimized.astype(np.float64)-original
    difference[...,2] = (difference[...,2]+np.pi)%(2*np.pi)-np.pi
    label_stats = dict(scenes=len(tokens),valid_points=int(valid.sum()),
        per_horizon_mean_optimized_minus_GT=[difference[valid[:,h],h].mean(0).tolist()for h in range(8)],
        mean_absolute_xy_change_m=np.mean(np.abs(difference[valid,:2]),axis=0).tolist(),
        mean_absolute_heading_change_degrees=float(np.abs(difference[...,2][valid]).mean()*180/np.pi),
        fraction_final_x_increased=float((difference[valid[:,7],7,0]>1e-4).mean()),
        source='Full TRAIN labels; geometry changes, not model prediction errors or new scoring')
    # A pre-fixed outcome-independent TRAIN slice; both arms use the same scenes.
    selected = sorted(tokens,key=lambda t:hashlib.sha256(('world_label_fit_probe_v1:'+t).encode()).hexdigest())[:512]
    index = {t:i for i,t in enumerate(tokens)}; results = {}
    for name, path in [('original', OLD/'candidate_refresh/update030000/bank'),
                       ('optimized', OLD/'optimized_labels_IL_20261028/candidate_refresh/update030000/bank')]:
        bank = CandidateBank(path,max_cached_scenes=0)
        source_hash = bank.manifest['identity']['source_model_sha256']
        errors=[]; full_envelope=[]; scores=[]; safe=[]; missing=[]; source_counts=Counter()
        for t in selected:
            i=index[t]; target=original[i] if name=='original' else np.asarray(optimized[i],dtype=np.float64)
            current=[r for r in bank.records(t)if r.candidate_source=='current_student_snapshot' and r.actor_checkpoint_sha256==source_hash]
            source_counts[len(current)]+=1
            if not current:missing.append(t)
            for r in current:
                error=np.asarray(r.trajectory_fp32,dtype=np.float64)-target
                error[...,2]=(error[...,2]+np.pi)%(2*np.pi)-np.pi
                e=error[valid[i]]
                if not len(e):continue
                errors.append(e);scores.append(r.raw_total_score)
                safe.append(all(r.raw_factor_labels[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')))
                # Coordinate enclosure is only a necessary range check. It is
                # not membership of the correlated/timing perturbation family.
                full_envelope.append(bool(np.all(np.abs(e)<=np.array([.3,.2,np.pi/360]))))
        values=np.concatenate(errors)
        scales=np.array([1.,1.,180/np.pi])
        results[name]=dict(source_updates=30000,source_model_sha256=source_hash,bank_hash=bank.bank_hash,
            TRAIN_scenes_requested=512,scene_fresh_candidate_count_histogram=dict(source_counts),
            missing_fresh_scenes=len(missing),candidates=len(errors),valid_points=len(values),
            mean_abs_error_xy_m_heading_degrees=(np.abs(values).mean(0)*scales).tolist(),
            point_abs_error_p50_xy_m_heading_degrees=(np.quantile(np.abs(values),.5,axis=0)*scales).tolist(),
            point_abs_error_p95_xy_m_heading_degrees=(np.quantile(np.abs(values),.95,axis=0)*scales).tolist(),
            candidate_all_valid_points_within_coordinate_envelope_fraction=float(np.mean(full_envelope)),
            sampled_TRAIN_candidate_PDMS=100*float(np.mean(scores)),sampled_TRAIN_official_NC_DAC_TTC_all_pass=float(np.mean(safe)),
            interpretation='Saved independent ODE outputs vs each arm own label, NOT online noisy-input TRAIN L1; two roots per scene, not two training seeds')
        bank.close()
    return dict(labels=label_stats,saved_30k_TRAIN_ODE=results,
        coordinate_envelope=dict(x_m=.3,y_m=.2,heading_degrees=.5,timing_seconds=.1,
            warning='Historical original_config only; heterogeneous V12 evidence. This simple coordinate check does not validate timing, dynamics, family membership or safety.'))


def plots(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms={n:read(root/n/'analysis.json') for n in ('original','optimized')}
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for name,j in arms.items():
        rows=[r for r in j['points_by_window']['500']if r['update']<=40000]
        x=[r['update']/1000 for r in rows];loss=[r['x'] for r in rows];score=[r['PDMS']for r in rows]
        ax[0].plot(x,loss,'o-',label=name);ax[1].plot(x,score,'o-',label=name);ax[2].plot(loss,score,'o-',label=name)
        for r in rows:
            if r['update'] in (13000,35000,40000):ax[2].annotate(str(r['update']//1000)+'k',(r['x'],r['PDMS']),fontsize=8)
    ax[0].set(xlabel='Optimizer updates (k)',ylabel='Trailing500 TRAIN normalized L1')
    ax[1].set(xlabel='Optimizer updates (k)',ylabel='Full Navtest PDMS')
    ax[2].set(xlabel='Trailing500 TRAIN normalized L1',ylabel='Full Navtest PDMS')
    for a in ax:a.grid(alpha=.25);a.legend()
    fig.tight_layout();fig.savefig(output/'loss_and_PDMS.png',dpi=180);fig.savefig(output/'loss_and_PDMS.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--analysis-root',required=True);a=parser.parse_args()
    root=Path(a.analysis_root);out=root/'detail';out.mkdir(exist_ok=False);started=time.monotonic()
    result=dict(curves=curve_comparison(root),paired_Navtest=paired_test(root),TRAIN_probe=training_probe(root),
                cost=dict(new_model_forwards=0,new_scorer_calls=0,new_optimizer_updates=0))
    plots(root,out);result['cost']['CPU_analysis_wall_seconds']=time.monotonic()-started
    write(out/'analysis.json',result);write(out/'COMPLETE.json',dict(status='COMPLETE',read_only=True))
    print(json.dumps(dict(output=str(out),matched=result['curves']['matched'],probe=result['TRAIN_probe']['saved_30k_TRAIN_ODE']),indent=2))


if __name__=='__main__':main()
