"""Separate random/targeted and actually updated scenes in TRAIN diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.flow_safe_improvement.evaluate_policy import load_scores
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, SFT_SHA256, fingerprint, file_sha
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('scores', 'baseline-scores', 'reference-scores', 'population', 'cohorts', 'targets', 'evaluation-override', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); authorized(require_dispatch())
    pop=json.loads(Path(a.population).read_text()); cohorts=json.loads(Path(a.cohorts).read_text())
    target=json.loads((Path(a.targets)/'manifest.json').read_text()); seeds,override=evaluation_seeds(pop,a.evaluation_override)
    current,policy,ph=load_scores(a.scores,pop,'evaluation',seeds)
    original,base,bh=load_scores(a.baseline_scores,pop,'evaluation',seeds)
    fixed,reference,rh=load_scores(a.reference_scores,pop,'reference')
    if (set(policy) != set(cohorts['diagnostic_tokens']) or set(policy) != set(base) or set(policy) != set(reference)
            or current['sample_manifest']['cohort'] != 'diagnostic_tokens'
            or current['sample_manifest']['execution_profile'] != original['sample_manifest']['execution_profile']
            or original['sample_manifest']['behavior_sha256'] != SFT_SHA256
            or fixed['sample_manifest']['behavior_sha256'] != SFT_SHA256
            or len({m['protocol_sha256'] for m in (current,original,fixed)}) != 1
            or target['population_identity'] != pop['identity_sha256']
            or not set(target['tokens']) <= set(pop['train_tokens'])):
        raise ValueError('TRAIN diagnostic identity changed')
    random=set(pop['diagnostic_random_tokens']); learned=set(target['tokens']);tokens=list(policy)
    groups={'random_all':[t for t in tokens if t in random],
            'random_in_learning_bank':[t for t in tokens if t in random and t in learned],
            'random_not_in_learning_bank':[t for t in tokens if t in random and t not in learned],
            'targeted_all':[t for t in tokens if t not in random],
            'targeted_in_learning_bank':[t for t in tokens if t not in random and t in learned]}
    tables={}
    for name, selected in groups.items():
        if not selected:
            continue
        logs=[pop['token_logs'][t] for t in selected];means={};changes={}
        for metric in policy[tokens[0]]:
            new=np.array([policy[t][metric].mean() for t in selected]);old=np.array([base[t][metric].mean() for t in selected])
            means[metric]=dict(policy=float(new.mean()),SFT=float(old.mean()))
            changes[metric]=log_bootstrap_difference(new-old,logs)
        safety={}
        for metric in (*SAFETY_METRICS,'joint'):
            keys=SAFETY_METRICS if metric=='joint' else (metric,)
            passed=lambda row: bool(np.logical_and.reduce([row[k]==1 for k in keys]).all())
            safe=[t for t in selected if passed(reference[t])]
            safety[metric]=dict(fixed_reference_safe_scenes=len(safe),
                policy_violations_in_fixed_safe=sum(not passed(policy[t]) for t in safe),
                SFT_violations_in_fixed_safe=sum(not passed(base[t]) for t in safe),
                new_failures=sum(passed(base[t]) and not passed(policy[t]) for t in selected),
                recovered=sum(not passed(base[t]) and passed(policy[t]) for t in selected))
        tables[name]=dict(scenes=len(selected),logs=len(set(logs)),actually_in_learning_bank=sum(t in learned for t in selected),
                          means=means,paired_changes=changes,safety=safety)
    report=dict(status='COMPLETE_STRATIFIED_TRAIN_DIAGNOSTIC',tables=tables,inference_seeds=seeds,
        evaluation_override=override,policy_identity=current['sha256'],baseline_identity=original['sha256'],
        reference_identity=fixed['sha256'],target_bank_identity=target['sha256'],
        payload_hashes=dict(policy=ph,baseline=bh,reference=rh),source_sha256=file_sha(__file__),
        directed_sample_is_not_population_unbiased=True,diagnostic_TRAIN_is_not_all_4096_updated_scenes=True,
        no_fit_or_hyperparameter_selection=True)
    report['sha256']=fingerprint(report);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',report)
    print(json.dumps({k:dict(scenes=v['scenes'],PDMS_change=v['paired_changes']['score']) for k,v in tables.items()}),flush=True)


if __name__ == '__main__':
    main()
