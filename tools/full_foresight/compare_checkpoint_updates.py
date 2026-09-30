"""Same-run, complete Navtest checkpoint progression with paired log intervals."""
import argparse
import json
from pathlib import Path

from tools.foresight.score_pdms import atomic_json, digest
from tools.foresight.summarize_experiments import (
    collapse_sampling_runs, paired_difference, sampling_contract, write_csv,
)
from tools.local_interaction_mask_v2.compare_pdms import read


def compare(first_directory, baseline_directory):
    directories=[Path(first_directory),Path(baseline_directory)]
    summaries=[json.loads((p/'summary.json').read_text()) for p in directories]
    exports=[s['export_identity'] for s in summaries]
    checkpoints=[e['checkpoint'] for e in exports]
    for summary,export in zip(summaries,exports):
        if (summary['schema']!='foresight_official_pdms_v1' or not summary['valid'] or
                summary['failed'] or not summary['full_navtest'] or summary['diagnostic'] or
                summary['scenes']!=12146 or summary['logs']!=136 or
                export['protocol']['precision']!='FP32'):
            raise ValueError('Complete valid FP32 Navtest results required')
    if (sampling_contract(exports[0])!=sampling_contract(exports[1]) or
            exports[0]['protocol']['sampling_seed']!=exports[1]['protocol']['sampling_seed'] or
            summaries[0]['evaluator_identity']!=summaries[1]['evaluator_identity'] or
            any(checkpoints[0][k]!=checkpoints[1][k] for k in
                ('run_identity','training_source_sha','arm','training_seed','model_class','scope')) or
            checkpoints[0]['completed']<=checkpoints[1]['completed']):
        raise ValueError('Identical run, evaluator and inference protocol with increasing update required')
    populations=[read(p/'scenes.csv') for p in directories]
    if (any(len(r)!=12146 or len({v['log'] for v in r.values()})!=136 or
            any(v['status']!='ok' for v in r.values()) for r in populations) or
            set(populations[0])!=set(populations[1]) or
            any(populations[0][t]['metric_cache_sha256']!=populations[1][t]['metric_cache_sha256']
                for t in populations[0])):
        raise ValueError('Complete identical original-cache populations required')
    seed=exports[0]['protocol']['sampling_seed']
    collapsed=[collapse_sampling_runs({seed:r},[seed])[0] for r in populations]
    result,scenes,logs=paired_difference(*collapsed)
    result.update(schema='ddp_full_foresight_checkpoint_progression_v1',arm=checkpoints[0]['arm'],
                  first_checkpoint=checkpoints[0],baseline_checkpoint=checkpoints[1],
                  sampling_seed=seed,source_csv_sha256={str(p/'scenes.csv'):digest(p/'scenes.csv') for p in directories},
                  evaluation_sources=[e['evaluation_source'] for e in exports],
                  selection_warning='User-requested checkpoint measurements; not Navtest-based model selection')
    return result,scenes,logs


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--first',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();result,scenes,logs=compare(a.first,a.baseline)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'summary.json',result)
    write_csv(out/'scenes.csv',scenes);write_csv(out/'logs.csv',logs)


if __name__=='__main__':main()
