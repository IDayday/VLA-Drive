"""Assemble completed, immutable study evidence without training or selecting models."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from iqe.contracts import require
from iqe.io import read_json, file_hash, atomic_json
from iqe.scoring.reference import COMPONENTS
from .config import load
from .data import canonical_bank
from .binding import verify_binding


SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def metric_row(name, r):
    ci = r['gain_ci']
    return [name, r['scenes'], r['logs'], f"{100*r['selected_01']:.4f}",
            f"{100*r['selected_gain_01']:+.4f}",
            f"[{100*ci['lower_95']:.4f}, {100*ci['upper_95']:.4f}]",
            f"{100*r['regret_01']:.4f}", r['high_to_zero'],
            '/'.join(str(r['safety'][k]['new_failures']) for k in SAFETY)]


def run(config, output):
    """Refuse a final report until real refits, evaluation and deployment checks exist."""
    root = config.root
    binding = verify_binding(config)
    plan = read_json(root/'REFIT_PLAN.json')
    external = read_json(root/'evaluation/stage_val/RESULTS.json')
    dev = read_json(root/'evaluation/dev_report/RESULTS.json')
    margin = read_json(root/'MARGIN_CALIBRATION.json')
    benchmark = read_json(root/'latency/BENCHMARK.json')
    online = read_json(root/'qualification/trained_online/REPORT.json')
    probe = read_json(root/'current_probe/TRIGGER.json')
    if probe['status'] == 'TRIGGERED':
        probe_result = read_json(root/'current_probe/RESULTS.json')
        require(probe_result['status'] == 'COMPLETE', 'current-state diagnostic incomplete')
    else:
        probe_result = probe
    require(benchmark['status'] == 'COMPLETE' and online['status'] == 'PASS', 'real deployment evidence missing')
    for kind in ('M_scene', 'M_set'):
        for f in [root/f'qualification/trained_online/FRESH_PROCESS_{kind}.json',
                  root/f'qualification/resume_{kind}/REPORT.json']:
            require(read_json(f)['status'] == 'PASS', 'new-process/resume qualification failed')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).parents[2]
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    inventory = {}

    def copy(src, dst=None):
        src = root/src
        target = out/(dst or src.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        inventory[str(src)] = dict(checksum=file_hash(src), archived_as=str(target.relative_to(out)))

    def md(name, body):
        (out/name).write_text(body.rstrip() + '\n')

    def csvfile(name, rows):
        with open(out/name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    for name in ('SELECTION_STUDY_BINDING.json', 'INNER_SPLIT_MANIFEST.json', 'REFIT_PLAN.json',
                 'M_SCENE_DIAGNOSTIC_RESULTS.json', 'M_SET_DIAGNOSTIC_RESULTS.json',
                 'MARGIN_CALIBRATION.json', 'FINAL_EVALUATION_LOCK.json',
                 'PROCESS_LABEL_FEASIBILITY.md', 'PROCESS_LABEL_FEASIBILITY.json'):
        copy(name)
    for name in ('ERROR_DECOMPOSITION.md', 'PER_EXPERT_SWITCH_COST.csv', 'SUBPOOL_DIAGNOSTICS.csv',
                 'ERROR_COHORTS.csv', 'RICH_COVERAGE.json'):
        copy('existing_audit/'+name)
    copy('existing_audit/TRAIN_HELDOUT_ERROR_GAP.md', 'EXISTING_TRAIN_HELDOUT_ERROR_GAP.md')
    for role in ('stage_val', 'dev_report'):
        copy(f'evaluation/{role}/RESULTS.json', f'evidence/{role}_RESULTS.json')
    copy('qualification/trained_online/REPORT.json', 'evidence/TRAINED_ONLINE.json')
    copy('latency/BENCHMARK.json', 'evidence/BENCHMARK.json')
    copy('latency/PLAN.json', 'evidence/LATENCY_PLAN.json')
    copy('current_probe/TRIGGER.json', 'evidence/CURRENT_PROBE_TRIGGER.json')
    copy('runtime/RESOURCE_INVENTORY_CLOSEOUT.json', 'evidence/RESOURCE_INVENTORY_CLOSEOUT.json')
    copy('runtime/finish_chain/STATUS.json', 'evidence/FINISH_CHAIN_STATUS.json')
    for p in sorted((root/'runtime/finish_chain').glob('*_EXIT.json')):
        copy(p.relative_to(root), 'evidence/finish_chain_'+p.name)
    if probe['status'] == 'TRIGGERED':
        for name in ('PLAN.json', 'RESULTS.json', 'FIT_NORMALIZATION.json', 'INITIALIZATION.json',
                     'INPUT_AUDIT_SUMMARY.json'):
            copy('current_probe/'+name, 'evidence/CURRENT_PROBE_'+name)
    evidence = []
    for phase in ('diagnostic', 'refit', 'overfit'):
        for kind in ('M_scene', 'M_set'):
            e = read_json(root/phase/kind/'TRAINING_EVIDENCE.json')
            require(e['status'] == 'COMPLETE', 'training incomplete')
            evidence.append(e)
            for name in ('TRAINING_EVIDENCE.json', 'INITIALIZATION.json', 'FIT_REPORT.json'):
                copy(f'{phase}/{kind}/{name}', f'evidence/{phase}_{kind}_{name}')
    for kind in ('M_scene', 'M_set'):
        copy(f'qualification/resume_{kind}/REPORT.json', f'evidence/RESUME_{kind}.json')
        copy(f'qualification/trained_online/FRESH_PROCESS_{kind}.json', f'evidence/FRESH_PROCESS_{kind}.json')

    # Every new policy is compared on exactly the same real bank and fixed role.
    flat = []
    fields = ('scenes', 'logs', 'base_01', 'oracle_01', 'selected_01', 'selected_gain_01',
              'selected_over_best_fixed_01', 'regret_01', 'oracle_capture', 'positive_switch_gain_01',
              'negative_switch_cost_01', 'switch_rate', 'base_fallback_rate', 'high_to_zero',
              'selected_zero', 'large_gap_regret_0_1', 'large_gap_regret_0_5', 'epsilon_optimal_rate')
    for report in (external, dev):
        for r in report['results']:
            row = dict(role=report['role'], model=r['model'], rule=r['rule'], **{k:r[k] for k in fields})
            row.update(gain_CI_low=r['gain_ci']['lower_95'], gain_CI_high=r['gain_ci']['upper_95'])
            row.update({f'new_{name}':r['safety'][name]['new_failures'] for name in SAFETY})
            require(abs(r['selected_gain_01']-(r['oracle_gain_01']-r['regret_01'])) < 1e-7, 'G-R identity')
            flat.append(row)
    csvfile('EXTERNAL_COMPARISON.csv', flat)
    headers = ['Policy', 'Scenes', 'Logs', 'PDMS /100', 'Gain pp', '95% log CI pp', 'Regret pp', 'High→0', 'New NC/DAC/TTC']
    lines = ['# Fixed-pool selection and tail risk', '',
             'All reported policies use the same Query70k Base and fixed road/interaction/safe-progress pool. '
             'CSV scores are [0,1]; tables use percent PDMS and percentage-point differences. '
             'Entire fixed manifests are retained. No newly executed Navtest or simulation calls.', '']
    for report in (external, dev):
        lines += [f"## {report['role']}", '', table(headers, [metric_row(r['model'],r) for r in report['results']]), '',
                  'Paired comparisons, same scenes/logs, 2000 bootstrap repetitions:', '']
        lines += [table(['Difference', 'Mean pp', '95% CI pp'],
                        [[c['first']+' − '+c['second'], f"{100*c['mean_difference_01']:+.4f}",
                          f"[{100*c['log_paired_CI']['lower_95']:.4f}, {100*c['log_paired_CI']['upper_95']:.4f}]"]
                         for c in report['comparisons']]), '']
    lines += ['Development has already been observed in previous studies and is exploratory, not blind testing. '
              'One seed is not a multi-seed reliability result. R00 used full D_S and historical checkpoint selection; '
              'it is not an unseen control on the new internal validation partition.', '',
              'High→0: Base ≥0.8 and selected ≤1e-8. New component failures compare valid full-score Base to '
              'selected below full score, using the locked tolerance. All rate denominators are the full fixed role. '
              'DDC has zero weight in this PDMS protocol. Positive and negative switch costs are separate in the CSV. '
              'Negative Oracle capture is preserved. Raw argmax is always reported even if calibration returns Base.']
    md('SELECTION_REGRET_AND_TAIL_RISK.md', '\n'.join(lines))

    # Signed costs by actual chosen branch, and per-log rows, without fitting any new policy.
    contribution, log_rows = [], []
    for role in ('stage_val', 'dev_report'):
        meta, bank, _ = canonical_bank(config, role)
        with np.load(root/f'evaluation/{role}/SELECTIONS.npz', allow_pickle=False) as f:
            require(f['scene_ids'].tolist() == meta['scene_ids'], 'report scene join')
            selections = {k:f[k].copy() for k in f.files if k not in ('scene_ids','source_logs','expert_ids')}
        truth = bank['scores'].astype(float); ix = np.arange(len(truth)); logs = np.asarray(meta['source_logs'])
        for policy, selected in selections.items():
            delta = truth[ix,selected]-truth[:,0]; regret = truth.max(1)-truth[ix,selected]
            for j, eid in enumerate(meta['expert_ids']):
                mask = selected == j
                contribution.append(dict(role=role, policy=policy, expert_id=eid, denominator=len(truth),
                    selections=int(mask.sum()), positive_gain_sum=float(np.maximum(delta,0)[mask].sum()),
                    negative_cost_sum=float(np.maximum(-delta,0)[mask].sum()),
                    high_to_zero=int(((truth[:,0]>=.8)&(truth[ix,selected]<=1e-8)&mask).sum()),
                    regret_gt_0_1=int(((regret>.1)&mask).sum()), regret_gt_0_5=int(((regret>.5)&mask).sum())))
            for log in sorted(set(logs)):
                m = logs == log
                log_rows.append(dict(role=role, policy=policy, source_log=log, scenes=int(m.sum()),
                                     gain_01=float(delta[m].mean()), regret_01=float(regret[m].mean()),
                                     positive_01=float(np.maximum(delta[m],0).mean()), negative_01=float(np.maximum(-delta[m],0).mean())))
    csvfile('NEW_MODEL_PER_EXPERT_COST.csv', contribution)
    csvfile('NEW_MODEL_PER_LOG.csv', log_rows)

    rows = []
    for e in evidence:
        rows.append([e['phase'], e['kind'], e['optimizer_steps'], e['unique_fit_scenes'], e['unique_fit_logs'],
                     e['consumed_scenes'], f"{e['Z_train']:.8f}", f"{e['GPU_seconds']:.2f}", e['new_parameters']])
    md('REFIT_TRAINING_EVIDENCE.md', '# Scratch training and fixed refits\n\n'+table(
        ['Phase','Model','Updates','Unique scenes','Logs','Presentations','Fixed Z','GPU seconds','New parameters'],rows)+
        '\n\nEach refit INITIALIZATION hash equals its diagnostic initial random hash, not its trained best hash. '
        'Native optimizer/scheduler/RNG/sampler checkpoints are preserved. Final refits use the predeclared last endpoint; '
        'external stage_val and dev do not choose a checkpoint. Heads read frozen caches and execute zero VLM calls. '
        'Model, optimizer and losses are FP32; deployment inherits the existing Base numerical runtime. '
        'GPU-seconds above are single-GPU training elapsed time including validation/checkpoint; they are not utilization integrals. '
        'Context construction/preload and final fit reporting are additional wall time in runtime launch receipts.\n\n'+
        f"M_scene: inner {plan['models']['M_scene']['inner_endpoint']} → refit {plan['models']['M_scene']['updates']}; "
        f"M_set: inner {plan['models']['M_set']['inner_endpoint']} → refit {plan['models']['M_set']['updates']}. "
        'Formula: ceil(T_inner × 16581 / 13139), without an external score-dependent extension.')

    diagnostic_rows = []
    for kind in ('M_scene','M_set'):
        diagnostic_rows += [metric_row(kind+' fit',read_json(root/f'diagnostic/{kind}/FIT_REPORT.json')),
                            metric_row(kind+' internal val',plan['models'][kind]['inner_metrics']),
                            metric_row(kind+' overfit64',read_json(root/f'overfit/{kind}/FIT_REPORT.json'))]
    md('TRAIN_HELDOUT_ERROR_GAP.md', '# Training versus unseen-log validation\n\n'+table(headers,diagnostic_rows)+
        '\n\nFit and internal-val numbers use the same internally selected model. The independent 64-scene models '
        'are implementation/fitting diagnostics only and never initialize either main model. New-model internal validation '
        'has 42 held-out logs; R00 saw those scenes and is only in-sample background there. Final refits see full D_S, '
        'so internal validation is no longer unseen for them. Final external checks are separate.')

    if probe['status'] == 'TRIGGERED':
        p = probe_result; rows = []
        input_audit = read_json(root/'current_probe/INPUT_AUDIT_SUMMARY.json')
        require(input_audit['all_future_reference_removal_equal'], 'current-state input independence failed')
        for role, r in p['metrics'].items():
            rows.append(metric_row('P_current '+role,r))
            rows.extend(metric_row(k+' matched '+role,v['metrics']) for k,v in p['matched_subset_comparators'][role].items())
        md('CURRENT_STATE_PROBE.md', '# Privileged current-state diagnostic\n\n'+table(headers,rows)+
            '\n\nTriggered only by internal validation; both main models had nonpositive gain. '
            '2048 fit / 512 validation scenes were selected by fixed hash; 2000 updates, one random initialization. '
            'The final endpoint is fixed and cannot enter a deployment bundle. Matched M_scene/M_set comparators are '
            'their internal diagnostic endpoints, not full-D_S refits. Data quantity, parameterization and information differ.\n\n'
            'Inputs: t0 timestamp-attested agents, provided current velocity, static map geometry and current ego, '
            'plus the fixed candidate trajectories. No reference future, GT, future velocity differences or simulation outcomes. '
            'The official occupancy t0 can contain future-only one-frame objects; the extractor explicitly rejects mismatched '
            f"object timestamps. Across all {input_audit['scenes']} actual records, "
            f"{input_audit['fields']['future_only_objects_rejected']['sum']} future-only objects were rejected; "
            'deleting future/reference containers leaves all extracted inputs identical. '
            'Range is 50 m and current three-camera frusta are enforced; occlusion is not verified, maps/annotations remain privileged. '
            'Current state improving selection would not alone prove W/A discarded the same visible information. '
            'Failure would not prove planning selection is intrinsically impossible.\n\n'
            'Actual source, normalization-fit counts, log counts and per-record hashes are in current_probe/PLAN, '
            'FIT_NORMALIZATION and immutable record receipts; no new official scoring or dev reads were used.')
    else:
        md('CURRENT_STATE_PROBE.md', '# Current-state diagnostic\n\nNOT_TRIGGERED. See immutable trigger evidence.')

    call_rows=[]
    for e in online['evidence']:
        c=e['calls'];call_rows.append([e['kind'],len(e['original_batch']),c['visual'],c['language'],c['context_projections'],
                                     json.dumps(c['decoder_samples'],sort_keys=True),e['invalid_fallbacks'],max(e['errors'].values())])
    md('INFERENCE_CALL_AUDIT.md', '# Real-image call and retention audit\n\n'+table(
        ['Model','Batch','Visual','Language','Context projection','Decoded samples by ID','Fallbacks','Max reported error'],call_rows)+
        '\n\nM_scene chooses before any action branch and groups only selected samples; M_set executes each K4 branch '
        'once. Synthetic negative tests additionally force mixed selections and invalid selected outputs to verify grouping, '
        'order restoration and an explicitly counted Base fallback. Real-image tests use actual Query70k and three trained experts, '
        'not fixture weights. Trained bundle fresh-process outputs are separately checked. Base/expert parameter and buffer hashes '
        'remain unchanged; shuffled/deleted labels do not alter context or trajectories.\n\n'
        'Online/cache numeric comparison replays attested original feature-extraction batches. It does not claim BF16 language '
        'kernels are bitwise invariant under arbitrary batch regrouping. This qualification and batch1 deployment timing are '
        'distinct protocols. No GT or metric-cache dependency is passed to deployment inference.')
    latency_rows=[]
    for r in benchmark['modes']:
        latency_rows.append([r['mode'],r['samples'],f"{1000*r['mean_seconds']:.2f}",f"{1000*r['p50_seconds']:.2f}",
                            f"{1000*r['p95_seconds']:.2f}",f"{r['decision_head_mean_ms']:.2f}",
                            f"{r['peak_allocated_bytes']/2**30:.3f}",r['decision_parameters'],r['invalid_fallbacks']])
    md('LATENCY_AND_PARAMETERS.md','# Real-image inference cost\n\n'+table(
        ['Mode','Samples','Mean ms','p50 ms','p95 ms','Head mean ms','Peak allocated GiB','Decision params','Fallbacks'],latency_rows)+
        '\n\nOne A800 80GB; batch1; 10 warmups and the same 100 real stage_val images per mode. '
        'Includes observation verification/read, image decode/preprocessing, visual and language forward, action branches, '
        'selection and CPU trajectory copy. CUDA synchronized wall times and separate CUDA events for the head are retained. '
        'No feature-cache latency is called end-to-end. Model parameter counts and every call are in evidence/BENCHMARK.json. '
        'All modes retain original Base precision and frozen generator weights. Allocated and reserved peaks are both recorded. '
        'Modes run sequentially on the same GPU. Small timing differences are not a controlled claim of a speedup: '
        'storage/page-cache state, GPU clocks and shared-host load can vary. The shared VLM dominates the end-to-end path. '
        'Margin fallback to Base does not constitute learned selection success.\n\n'+
        'Parameter inventory (actual loaded tensors):\n\n'+table(
            ['Module','Parameters'],
            [[k,v] for k,v in benchmark['modes'][0]['Base_parameter_breakdown'].items()]+
            [[eid,v] for eid,v in benchmark['modes'][0]['expert_parameters'].items()])+
        '\n\nThe Base breakdown reports its principal visual/language/planner modules; remaining embeddings '
        'and other Base parameters are included in entire_Base_adapter. Private expert counts include their '
        'condition readers and complete action branches. All generator parameters remain frozen.')

    current = {r['model']:r for r in dev['results']}
    statuses = {}
    for k in ('M_scene','M_set'):
        r=current[k];tail=any(r['safety'][s]['new_failures'] for s in SAFETY[:2]) or r['high_to_zero']>0
        statuses[k]=('NO_SELECTION_GAIN' if r['selected_gain_01']<=0 else
                     'SELECTION_GAIN_WITH_TAIL_RISK' if tail else
                     'DIRECTIONAL_POSITIVE' if r['gain_ci']['lower_95']>0 else 'INCONCLUSIVE')
    verdict=dict(IMPLEMENTATION='COMPLETE',REAL_TRAINING='PASS',FIXED_BASE_AND_K4='PASS',
        RANDOM_INITIALIZATION='PASS',EXACT_RESUME='PASS',REAL_IMAGE_INTEGRATION='PASS',
        FROZEN_GENERATOR_RETENTION='PASS',FRESH_PROCESS_BUNDLE='PASS',NEW_MODEL_RESULTS=statuses,
        INTERNAL_LOG_GENERALIZATION='both selected diagnostic models below Base',
        CURRENT_STATE_PROBE=probe_result['status'],CURRENT_STATE_PROBE_DEPLOYABLE=False,
        NEW_NAVTEST=False,ACTIVE_UPDATED=False,EXPERTS_TRAINED=0,EXTERNAL_SCORER_WEIGHTS_LOADED=False,
        R00_RETRAINED=False,PUSHED=False,single_seed=True,dev_is_exploratory=True,
        CANDIDATE_CONDITIONING_HELPFUL='relative to M_scene, not a gain over Base or R00',
        ROUTER_SUFFICIENT=False,REPRESENTATION_DIAGNOSTIC_SUPPORTED=False,
        JOINT_VLM_TRAINING_AUTHORIZED_OR_STARTED=False,
        PRODUCTION_RELEASE=False,
        source_audit_commit=binding['audit_commit'],report_source_commit=code_commit,binding_hash=binding['hash'],
        pool_hash=binding['pool_hash'],full_population_denominators={r['role']:r['requested_scenes'] for r in (external,dev)})
    atomic_json(out/'VERDICT.json',verdict)
    for item in inventory.values():
        require(file_hash(out/item['archived_as']) == item['checksum'], 'report archive checksum mismatch')
    atomic_json(out/'SOURCE_MANIFEST.json',dict(runtime_root=str(root),report_source_commit=code_commit,
                                               input_artifacts=inventory,weights_images_and_metric_caches_committed=False))
    return verdict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    print(json.dumps(run(load(args.config),args.output)))


if __name__=='__main__':main()
