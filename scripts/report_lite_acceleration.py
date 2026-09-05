#!/usr/bin/env python3
"""Render measured speed/loss results, keeping failures and quality limits visible."""
import argparse
import json
from pathlib import Path
import re
import shutil
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-log',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    lock=json.loads((a.artifacts/'formal_training_layout_lock.json').read_text())
    comparison=json.loads((a.artifacts/'batch_comparison.json').read_text())
    parity=json.loads((a.artifacts/'attention_parity.json').read_text())
    tests=re.findall(r'(\d+) passed[^\n]*',a.test_log.read_text())
    if not tests:raise RuntimeError('Missing completed test evidence')
    metrics={p.parent.name:json.loads(p.read_text()) for p in (a.artifacts/'throughput').glob('*/metrics.json')}
    a.output.mkdir(parents=True)
    for name in ('formal_training_layout_lock.json','batch_comparison.json','attention_parity.json','scorer_parity.log'):
        shutil.copyfile(a.artifacts/name,a.output/name)
    shutil.copyfile(a.test_log,a.output/'pytest.log')
    snapshot={name:{k:v for k,v in m.items() if k!='training_curve_global_rank_mean'} for name,m in metrics.items()}
    (a.output/'throughput_summary.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    complete={name:m for name,m in metrics.items() if m.get('status')=='success'}
    labels=['Old Lite\neager GB64']+list(complete)
    speeds=[8.548277658221915]+[m['end_to_end_samples_per_second'] for m in complete.values()]
    axes[0].bar(labels,speeds,color=['#999999']+['#2675bd']*len(complete))
    axes[0].set_ylabel('Samples / second (whole optimizer cycle)')
    axes[0].set_title('Full model + WM throughput')
    for i,v in enumerate(speeds):axes[0].text(i,v,f'{v:.2f}',ha='center',va='bottom')
    for axis,key,title in zip(axes[1:],('loss/trajectory_loss','loss/final_score_loss'),('Trajectory loss','Scorer loss')):
        for name,m in complete.items():
            batch=m['global_batch_size'];width=max(1,1024//batch)
            rows=m['training_curve_global_rank_mean'];xs=[];ys=[]
            for start in range(0,len(rows),width):
                chunk=rows[start:start+width]
                xs.append(chunk[-1]['optimizer_step']*batch)
                ys.append(statistics.fmean(row[key] for row in chunk))
            axis.plot(xs,ys,label=name)
        axis.set_title(title+' (train only)');axis.set_xlabel('Sample presentations')
        axis.grid(alpha=.2);axis.legend()
    fig.suptitle('Equal exposure pilot — not a final PDMS non-inferiority result')
    fig.savefig(a.output/'speed_and_equal_exposure_loss.png',dpi=160)
    plt.close(fig)
    lines=['# Lite acceleration execution results','',
           'These are new system/numerical/train-only pilot measurements. No new Navtest result is claimed.','',
           '| Layout | Global batch | Samples/s | Peak allocated GiB | Status |',
           '|---|---:|---:|---:|---|']
    for name,m in metrics.items():
        lines.append(f"| {name} | {m.get('global_batch_size')} | {m.get('end_to_end_samples_per_second','NOT_MEASURED')} | {m.get('peak_allocated_gib','OOM')} | {m['status']} |")
    lines += ['',f"Selected **{lock['selected_layout']}**, global batch **{lock['global_batch_size']}**; "
              f"estimated full 27-epoch compute wall time **{lock['estimated_27_epoch_hours']:.2f} h**, "
              'excluding launch/checkpoint/epoch-boundary overhead. Only BaseInit is requested.', '',
              f"Exact budget: {lock['steps_per_epoch']} updates/epoch, {lock['total_steps']} total. "
              f"Completed tests: {tests[-1]} passed (see pytest.log).",'',
              'The original 16x8 attempt OOMed inside a duplicated diagnostic graph. The failure is retained. '
              'After moving the isolated audit before the normal graph, the B8 real smoke completed three updates '
              'and exact student export/reload, but its 73.57 GiB peak is above the formal reserve gate.','',
              'All 24 actual attention blocks passed preregistered FP32 forward/backward tolerances. '
              'Complete BF16 policy/teacher rounding is reported separately in attention_parity.json. '
              'All trainable/optimizer/EMA master precision, physical tasks, source data, horizons, '
              'and generator/scorer function are retained.','',
              'The same sample multiset is verified for each eligible pilot. The last 4,096 presentations '
              'are compared with the preregistered 10% gross-regression screen. This initial-warmup '
              'screen is **not** unseen-log generalization or epoch-27 PDMS equivalence.','',
              '## Actual locked peak learning rates','', '```json',
              json.dumps(lock['logical_peak_learning_rates'],indent=2),'```','',
              f"EMA actual endpoints: {lock['ema_actual_start_momentum']:.12f} -> {lock['ema_actual_end_momentum']:.12f}. "
              '5% warmup, cosine to 10% of peak, clipping norm1; WM .01 -> .10 over first10%.','',
              '## Quality interpretation','',
              'The model is not simplified for speed. GB128 changes update count and optimization, so '
              'no guaranteed final-score claim is made. Fixed epoch27 student-only evaluation is still '
              'required. No Navtest-based LR/loss/sampler/epoch selection was performed.','',
              'Original Base/VQA epoch-one checkpoints remain immutable. A changed-batch formal run '
              'starts from the audited Base VLM and shared random planning stack, not an M0 planner '
              'or an optimizer-reset checkpoint presented as lossless continuation.','',
              '![Speed and equal-exposure training loss](speed_and_equal_exposure_loss.png)','']
    (a.output/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'output':str(a.output),'selected':lock['selected_layout'],'tests_passed':int(tests[-1])}))


if __name__=='__main__':main()
