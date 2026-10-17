"""Describe original F-SFT -> A SFT20k -> A RL20k on saved observations.

This is arithmetic attribution across three checkpoints, not a causal isolation
of time weighting. It preserves the own-SFT RL comparison and does not infer an
unregistered original-F Navtest baseline. No model, scoring or dispatch calls.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.analysis.fm_transfer.analyze_evaluation import match_transfer
from scripts.analysis.fm_transfer.analyze_safety_transitions import source_reward
from scripts.analysis.simwam_recipe.analyze_evaluation import read, METRICS
from starVLA.rl.fm_transfer.contracts import ROOT
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.flow_grpo.transactions import atomic_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    require(np.isfinite([left, right]).all() and abs(left-right) <= 1e-12,
            'Independent stage summaries disagree')


def pass_history(before, supervised, reinforced):
    """1 means fully passing; the three positions follow checkpoint order."""
    arrays = [np.asarray(v, dtype=bool) for v in (before, supervised, reinforced)]
    require(all(v.ndim == 1 and v.shape == arrays[0].shape for v in arrays),
            'Pass histories must share an exact scene population')
    return {f'{a}{b}{c}': int(((arrays[0] == bool(a)) & (arrays[1] == bool(b)) &
                              (arrays[2] == bool(c))).sum())
            for a in (0, 1) for b in (0, 1) for c in (0, 1)}


def compose(initial, supervised, reinforced, logs):
    arrays = [np.asarray(v, dtype=np.float64) for v in (initial, supervised, reinforced)]
    require(all(v.shape == (len(logs), len(METRICS)) and np.isfinite(v).all()
                and (v >= 0).all() and (v <= 1).all() for v in arrays) and len(logs) > 0,
            'Finite complete matched scene metrics in 0--1 are required')
    first, second, last = arrays
    sft, rl, total = second-first, last-second, last-first
    require(np.allclose(sft+rl, total, atol=1e-15, rtol=0), 'Stage arithmetic differs')
    result = dict(stage_order=['original_F_SFT', 'A_SFT20k', 'A_RL20k'],
                  history_encoding='1=fully passes; positions follow stage_order', stages={})
    for name, delta, a, b in [('SFT_minus_initial', sft, first, second),
                              ('RL_minus_own_SFT', rl, second, last),
                              ('RL_minus_initial_composed', total, first, last)]:
        result['stages'][name] = dict(
            mean_delta={key: float(delta[:, i].mean()) for i, key in enumerate(METRICS)},
            PDMS_log_cluster_bootstrap=log_bootstrap_difference(delta[:, 0], logs),
            G_plus_raw=float(delta[delta[:, 0] > 0, 0].sum()),
            G_minus_raw=float(-delta[delta[:, 0] < 0, 0].sum()),
            source_reward_delta_mean=float((source_reward(b)-source_reward(a)).mean()),
            nonzero_to_zero=int(((a[:, 0] > 0) & (b[:, 0] == 0)).sum()),
            zero_to_nonzero=int(((a[:, 0] == 0) & (b[:, 0] > 0)).sum()),
            practical_improved_ge005=int((delta[:, 0] >= .005).sum()),
            practical_degraded_le_minus005=int((delta[:, 0] <= -.005).sum()))
    histories = {key: pass_history(*(v[:, i] == 1 for v in arrays))
                 for i, key in enumerate(METRICS[1:4], 1)}
    histories['joint_NC_DAC_TTC'] = pass_history(*((v[:, 1:4] == 1).all(1) for v in arrays))
    result['safety_history_counts'] = histories
    result['score_history_counts'] = dict(
        SFT_improved_ge005=int((sft[:, 0] >= .005).sum()),
        SFT_improved_then_RL_degraded_ge005=int(((sft[:, 0] >= .005) & (rl[:, 0] <= -.005)).sum()),
        SFT_improved_but_final_not_above_initial=int(((sft[:, 0] >= .005) & (total[:, 0] <= 0)).sum()),
        RL_improved_ge005_after_nonimproving_SFT=int(((rl[:, 0] >= .005) & (sft[:, 0] <= 0)).sum()))
    return result


def analyze(root, cohort):
    root = Path(root)
    proof_path = root/'qualification/A_DDP_TIME/initial_baseline_reuse.json'
    proof = json.loads(proof_path.read_text())
    paths = [Path(proof['reuse'][cohort]['path']),
             root/'evaluations'/f'A_DDP_TIME_SFT_update020000_{cohort}',
             root/'evaluations'/f'A_DDP_TIME_RL_update020000_{cohort}']
    loaded = [read(path) for path in paths]
    identities = [entry[0] for entry in loaded]
    compatibility = [match_transfer(identities[1], identities[0]),
                     match_transfer(identities[2], identities[1])]
    require(all(x[1] == loaded[0][1] and x[2] == loaded[0][2] for x in loaded),
            'Three stages must have exactly the same scene/log order')
    expected_count = {'DEV': (1696, 16), 'TRAIN_DIAGNOSTIC': (512, 343)}[cohort]
    require((len(loaded[0][1]), len(set(loaded[0][2]))) == expected_count,
            'Incomplete registered population')
    evidence = {str(proof_path): file_sha(proof_path)}
    for stage, identity, path in zip(('SFT', 'RL'), identities[1:], paths[1:]):
        export_path = root/'exports/A_DDP_TIME'/f'{stage}_update020000/export.json'
        export = json.loads(export_path.read_text())
        require(export['updates'] == 20000 and export['stage'] == stage and not export['qualification']
                and export['new_SFT'] == identity['new_SFT']
                and export['weight_sha256'] == identity['weight_sha256']
                and file_sha(export_path) == identity['export_receipt_sha256'],
                'Only registered own-SFT and fixed RL endpoints are eligible')
        evidence[str(export_path)] = file_sha(export_path)
    old_summaries = [root/'analysis'/f'A_DDP_TIME_SFT_update020000_{cohort}_paired_initial.json',
                     root/'analysis'/f'A_DDP_TIME_RL_update020000_{cohort}_paired_own_SFT.json']
    paired = [json.loads(path.read_text()) for path in old_summaries]
    for i, item in enumerate(paired):
        require(item['status'] == 'COMPLETE_POPULATION' and item['identity'] == identities[i+1]
                and item['reference'] == str(paths[i]), 'Wrong existing matched summary')
    output = dict(schema='fm_transfer_A_saved_stage_composition_v1',
        status='COMPLETE_SAVED_THREE_STAGE_COMPARISON_NOT_NAVTEST', cohort=cohort,
        scenes=expected_count[0], logs=expected_count[1], seed=2101,
        analysis_sha256=file_sha(__file__), compatibility=compatibility,
        stage_evaluations=[str(p) for p in paths], score_units='raw0to1; differences multiply100 for percentage points',
        threshold_raw_score=.005, new_inference=0, new_scoring=0, optimizer_updates=0,
        limitations=['Arithmetic stages do not isolate causal effects of time sampling/weighting or added SFT exposure.',
                    'RL-minus-initial is composed end-to-end change, not the own-SFT RL increment.',
                    'Gain/loss masses and bootstrap intervals are recomputed from paired scenes, not added across stages.',
                    'DEV may have been seen by pretraining; TRAIN_DIAGNOSTIC is intentionally selected.',
                    'Fixed single training/inference seed; intervals include log resampling only.',
                    'Original-F Navtest is not part of this analysis.'], modes={})
    for mode in identities[1]['modes']:
        output['modes'][mode] = compose(*(entry[3][mode] for entry in loaded), loaded[0][2])
        for key, previous in zip(('SFT_minus_initial', 'RL_minus_own_SFT'), paired):
            actual = output['modes'][mode]['stages'][key]
            expected = previous['modes'][mode]['paired_comparison']
            close(actual['PDMS_log_cluster_bootstrap']['mean'], expected['PDMS_log_cluster_bootstrap']['mean'])
            for a, b in zip(actual['PDMS_log_cluster_bootstrap']['CI95'], expected['PDMS_log_cluster_bootstrap']['CI95']):
                close(a, b)
            close(actual['G_plus_raw'], expected['positive_gain_sum'])
            close(actual['G_minus_raw'], expected['negative_degradation_sum'])
            require(actual['nonzero_to_zero'] == expected['nonzero_to_zero']
                    and actual['zero_to_nonzero'] == expected['zero_to_nonzero'], 'Zero-score transitions disagree')
    for path in [*(p/'manifest.json' for p in paths), *old_summaries]:
        evidence[str(path)] = file_sha(path)
    output['evidence_sha256'] = evidence
    output['paired_population_digest'] = fingerprint(dict(tokens=loaded[0][1], logs=loaded[0][2]))
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cohort', choices=('DEV', 'TRAIN_DIAGNOSTIC'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous stage analysis')
    start = time.monotonic()
    result = analyze(args.root, args.cohort)
    result['analysis_wall_seconds'] = time.monotonic()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(dict(status=result['status'], cohort=args.cohort, scenes=result['scenes'],
                          new_inference=0, output=str(args.output))))


if __name__ == '__main__':
    main()
