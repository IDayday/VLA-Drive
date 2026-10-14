"""Rebuild cross-series tables from the archived evidence, without GPUs or data assets."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / '20261008'
REPO = ROOT.parents[2]
DATA = ROOT / 'data'
METRICS = ['PDMS', 'NC', 'DAC', 'TTC', 'EP', 'Comfort', 'Direction']


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def dump(name, value):
    (DATA / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write(name, values):
    fields = list(dict.fromkeys(k for row in values for k in row))
    with (DATA / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main():
    historical = rows(OLD / 'data/checkpoint_history.csv')
    g = rows(DATA / 'NAVSIM_NAVTEST.csv')
    status = {(r['dataset'], r['group']): r for r in rows(DATA / 'TRAINING_STATUS.csv')}
    ledger = []
    for r in historical:
        ledger.append({**r, 'method_id': f"{r['series']}/{r['version']}/{r['arm']}",
                       'variant': 'executed', 'protocol': 'canonical10_FP32_seed42',
                       'source_archive': '20261008/data/checkpoint_history.csv'})
    for r in g:
        ledger.append({'series': 'G', 'version': r['group'][:2], 'arm': 'structured_world',
                       'method_id': 'G/' + r['group'][:2], 'split': 'navtest',
                       'update': r['updates'], 'variant': r['variant'],
                       **{k: r.get({'Comfort': 'comfort', 'Direction': 'DDC'}.get(k, k), '') for k in METRICS},
                       **{k: r[k] for k in ['scenes', 'logs', 'failed', 'zero_scenes', 'training_seed', 'sampling_seed']},
                       'evaluation_identity': r['evaluation_identity'],
                       'protocol': 'canonical10_FP32_seed42', 'source_archive': '20261014/data/NAVSIM_NAVTEST.csv'})
    # Retain previous G development observations in their own split, never mix into Navtest best.
    p = REPO / 'reports/structured_world/round1_20261008/snapshot_20261010_v1/PLANNING_RESULTS.csv'
    for r in rows(p):
        if r['dataset'] != 'navsim' or not r['group'].startswith('G'):
            continue
        ledger.append({'series': 'G', 'version': r['group'][:2], 'arm': 'structured_world',
                       'method_id': 'G/' + r['group'][:2], 'split': 'dev', 'update': r['updates'],
                       'variant': r['variant'], **{k: r.get(k if k != 'Comfort' else 'comfort', '') for k in METRICS},
                       'scenes': r['samples'], 'logs': 16, 'failed': 0,
                       'zero_scenes': round(float(r['zero_fraction']) * int(r['samples'])),
                       'training_seed': 42, 'sampling_seed': 42, 'evaluation_identity': r['evaluation_identity'],
                       'protocol': 'canonical10_FP32_seed42',
                       'source_archive': 'reports/structured_world/round1_20261008/snapshot_20261010_v1/PLANNING_RESULTS.csv'})
    final = read(OLD / 'analysis/objective_v2/final_report.json')
    for r in final['summary']:
        ledger.append({'series': 'objective_v2', 'version': r['variant'], 'arm': r['role'],
                       'method_id': 'objective_v2/' + r['variant'] + '/' + r['role'],
                       'split': 'dev', 'update': r['update'], 'variant': 'executed',
                       **{k: r.get(k, '') for k in METRICS},
                       'scenes': r['scenes'], 'logs': r['logs'], 'failed': r['failed'],
                       'zero_scenes': r['zero_count'], 'training_seed': 42, 'sampling_seed': 42,
                       'protocol': 'canonical10_FP32_seed42',
                       'source_archive': '20261008/analysis/objective_v2/final_report.json'})
    write('ALL_CHECKPOINTS.csv', ledger)

    best = []
    groups = defaultdict(list)
    for r in ledger:
        if r['split'] == 'navtest' and r['variant'] in ['executed', 'q_final']:
            groups[r['method_id']].append(r)
    for name, candidates in groups.items():
        b = max(candidates, key=lambda r: float(r['PDMS']))
        latest = max(candidates, key=lambda r: int(r['update']))
        best.append({'method_id': name, 'best_update': b['update'],
                     **{k: b[k] for k in METRICS}, 'zero_scenes': b['zero_scenes'],
                     'latest_evaluated_update': latest['update'], 'latest_PDMS': latest['PDMS'],
                     'observed_checkpoints': len(candidates),
                     'selection': 'post_hoc_best_of_completed_full_navtest',
                     'source_archive': b['source_archive']})
    write('BEST_NAVTEST.csv', best)
    updates = sorted({int(r['update']) for vv in groups.values() for r in vv})
    write('MATCHED_NAVTEST.csv', [dict(update=u, **{
        name: next((r['PDMS'] for r in vv if int(r['update']) == u), '')
        for name, vv in groups.items()}) for u in updates])

    pilots = {}
    evid = ROOT / 'evidence/objective_pilots'
    for architecture, filename in [
        ('S0', 'dev_S0_matched_evidence.json'), ('S3', 'dev_S3_matched_evidence.json'),
        ('S0', 'dev_S0_fm_gain_both_evidence.json'), ('S3', 'dev_S3_fm_gain_evidence.json'),
        ('S0', 'dev_S0_fm_endpoint_evidence.json'), ('S3', 'dev_S3_fm_endpoint_evidence.json')]:
        e = read(evid / filename)
        for method, m in e['means_points'].items():
            key = (architecture, method)
            if key in pilots:
                assert pilots[key]['PDMS'] == m['score']
                continue
            comparison = e['comparisons'].get(method, {})
            pilots[key] = {'architecture': architecture, 'method': method,
                           'adaptation_updates': 0 if method == 'initial' else 512,
                           'parent_update': 50000 if architecture == 'S0' else 100000,
                           'split': 'dev', 'scenes': e['scenes'], 'training_seed': 42, 'sampling_seed': 42,
                           'PDMS': m['score'], 'NC': m['no_at_fault_collisions'],
                           'DAC': m['drivable_area_compliance'], 'TTC': m['time_to_collision_within_bound'],
                           'EP': m['ego_progress'], 'Comfort': m['comfort'], 'Direction': m['driving_direction_compliance'],
                           'delta_PDMS_vs_control': comparison.get('delta_points', {}).get('score', ''),
                           'paired_log_95CI_json': json.dumps(comparison.get('log_cluster_95ci_points', {}).get('score')),
                           'fitting_json': json.dumps(e.get('target_errors', {}).get(method, {}), sort_keys=True),
                           'source': 'evidence/objective_pilots/' + filename,
                           'scope': 'mature_checkpoint_fresh_optimizer; not generic-start long training'}
    write('OBJECTIVE_MATURE_PILOTS.csv', list(pilots.values()))

    # Cross-series analysis pairs exact same anonymized token and log IDs.
    wanted_old = {f'navtest:S:original:{s}:{u}:42' for s in ['S0', 'S3'] for u in [50000, 60000, 70000, 100000]}
    wanted_old |= {'navtest:C:original:C1:100000:42',
                   'navtest:A_V:planning_interface:V_QUERY:100000:42',
                   'navtest:A_V:planning_interface:A_NO_MAE:100000:42'}
    banks = defaultdict(dict)
    for path, wanted in [(OLD / 'data/scenes/navtest.csv.gz', wanted_old),
                         (DATA / 'G_NAVTEST_SCENES.csv.gz', None)]:
        with gzip.open(path, 'rt') as stream:
            for r in csv.DictReader(stream):
                if wanted is not None and r['evaluation_id'] not in wanted:
                    continue
                if wanted is None and not (r['group'] == 'G2' and r['variant'] == 'q_final'):
                    continue
                banks[r['evaluation_id']][r['scene_id']] = r
    requests = []
    for step in [50000, 60000, 70000]:
        for s in ['S0', 'S3']:
            requests.append((f'navtest:G:G2:{step}:q_final:42', f'navtest:S:original:{s}:{step}:42',
                             'matched optimizer updates; different architecture, preparation and compute'))
    for s in ['S0', 'S3']:
        requests.append(('navtest:G:G2:45000:q_final:42', f'navtest:S:original:{s}:100000:42',
                         'post-hoc best versus best; different exposure and architecture'))
    for b in sorted(wanted_old - {f'navtest:S:original:{s}:{u}:42' for s in ['S0', 'S3'] for u in [50000, 60000, 70000, 100000]}):
        requests.append(('navtest:G:G2:45000:q_final:42', b, 'post-hoc best versus best; overall recipe only'))
    log_order = read(DATA / 'G_BOOTSTRAP_LOG_ORDER.json')
    log_index = {v: i for i, v in enumerate(log_order)}
    draws = np.random.default_rng(42).integers(0, len(log_order), size=(20000, len(log_order)))
    comparisons = []
    for candidate, baseline, scope in requests:
        a, b = banks[candidate], banks[baseline]
        assert set(a) == set(b) and len(a) == 12146
        tokens = sorted(a)
        assert all(a[t]['log_id'] == b[t]['log_id'] for t in tokens)
        idx = np.array([log_index[a[t]['log_id']] for t in tokens])
        counts = np.bincount(idx, minlength=136)
        metrics = {}
        for metric in METRICS:
            alias = {'Comfort': 'comfort', 'Direction': 'DDC'}.get(metric, metric)
            delta = np.array([float(a[t][alias]) - float(b[t][metric]) for t in tokens])
            sums = np.bincount(idx, weights=delta, minlength=136)
            bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
            metrics[metric] = {'delta_points': float(delta.mean()), 'paired_log_95CI': np.quantile(bootstrap, [.025, .975]).tolist()}
        az = np.array([float(a[t]['PDMS']) == 0 for t in tokens])
        bz = np.array([float(b[t]['PDMS']) == 0 for t in tokens])
        comparisons.append({'candidate': candidate, 'baseline': baseline, 'scope': scope,
                            'scenes': 12146, 'logs': 136, 'metrics': metrics,
                            'fixed_baseline_zero': int((bz & ~az).sum()),
                            'introduced_zero': int((~bz & az).sum()), 'both_zero': int((az & bz).sum())})
    dump('G2_HISTORICAL_COMPARISONS.json', {'replicates': 20000, 'seed': 42, 'log_order': 'G_BOOTSTRAP_LOG_ORDER.json',
          'weighting': 'resampled log sums / resampled scene counts', 'comparisons': comparisons,
          'limits': 'One training seed. Exploratory unadjusted intervals. Architecture, supervision, initialization preparation and compute differ; not isolated auxiliary-task attribution.'})

    # Compact graph of actual methods and their evidence, including untrained candidates.
    attempts = []
    def attempt(name, family, change, status, evidence, **kw):
        attempts.append(dict(method_id=name, family=family, change=change, status=status, evidence=evidence, **kw))
    csettings = {'C0': (128,96,1,144), 'C1': (256,192,2,144), 'C2': (256,192,1,576),
                 'C3': (384,288,2,324), 'C4': (512,384,4,144), 'C5': (512,384,2,576)}
    cs = read(ROOT / 'evidence/C_S_COMPLETED_RUNS.json')['runs']
    for name, (w,h,pool,wq) in csettings.items():
        attempt(name, 'C', f'DINO target {w}x{h}, pool{pool}, W={wq}; current + one sampled future + GT-MAE',
                'formal_100k_complete' if name in cs else 'no_formal_planning_result',
                'evidence/c/formal_registration_v1.json',
                engineering_only='C3 real 4-step resume check; C5 120-update profiles; C2 target/config preparation' if name not in cs else '',
                training_source_sha=cs[name]['status']['source_sha'] if name in cs else '')
    schanges = {'S0': 'C1 current; eight future DINO frames; token LayerNorm; MAE reads W',
                'S1': 'S0 + GT-ego action conditioning in training-only future head',
                'S2': 'S0 future target replaced by V-JEPA video tokens; no action condition',
                'S3': 'S2 + GT-ego action memory in future head',
                'S4': 'S3 + original action head directly reads final W'}
    for name, change in schanges.items():
        attempt(name, 'S', change, 'formal_100k_complete', 'evidence/C_S_COMPLETED_RUNS.json',
                training_source_sha=cs[name]['status']['source_sha'])
    for name,change,src in [('A_ACTION','S0 MAE readout moves from W to action representation','f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882'),
                            ('A_NO_MAE','S0 removes MAE supervision, retains current/future DINO','f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882'),
                            ('V_QUERY','S3 GT-ego action enters both memory and future queries','7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c')]:
        attempt(name,'A/V',change,'formal_100k_complete','../20261008/configs/av',training_source_sha=src)
    for alias, original in [('A_W','S0'),('V_NONE','S2'),('V_MEMORY','S3')]:
        attempt(alias,'A/V alias','Identical verified checkpoint reuse','alias_not_independent_run',
                '../20261008/analysis/baseline_reuse_proof.json',alias_of=original)
    probe_rows = []
    for name in ['F0','F1','V0','V1','F_ACTION_ONLY','V_ACTION_ONLY','P_QUERY_v2','P_ACTION_ONLY_QUERY_v2']:
        folder = ROOT / 'evidence/probes' / name
        summary, identity = read(folder/'SUMMARY.json'), read(folder/'identity.json')
        probe_rows.append({'probe': name, **summary, 'identity_json': json.dumps(identity, sort_keys=True),
                           'scope': 'frozen C1 readout; auxiliary MSE, not planner PDMS'})
        attempt(name,'frozen_readout_probe','Fixed C1 W: teacher type / action memory-query / action-only auxiliary controls',
                'completed_2000_readout_updates; no_policy_training','evidence/probes/'+name,
                normalized_mse=summary['normal_normalized_mse'])
    write('FROZEN_READOUT_PROBES.csv', probe_rows)
    attempt('C1_W_INTERVENTION','dependency_diagnostic','Permute intermediate W at layers9/18 on fixed128dev',
            'completed_read_only; not_new_policy','evidence/probes/C1_W_CANONICAL_PDMS_DIAGNOSTIC.json')
    attempt('C1_FM_STEPS','inference','Fixed C1@100k original Euler 1/2/3/5/8/10/15/20/30 steps',
            'nine_complete_navtest_banks; no_training','evidence/fm_steps/RESULTS.json')
    for name in ['R','S0','S3']:
        attempt(name+'_r1','optimized_labels','r1 optimized ego labels; original FM unchanged',
                'formal_100k_complete','../20261008/configs/optimized')
    for (architecture, method), row in pilots.items():
        attempt(architecture+'_pilot_'+method,'objective mature pilot',method,
                'completed_512_adaptation' if method != 'initial' else 'read_only_parent',row['source'],
                parent_update=row['parent_update'],dev_PDMS=row['PDMS'])
    paused = read(OLD/'analysis/objective_v2/original_training_pause.json')
    for name, r in paused['runs'].items():
        attempt(name,'aggressive_objective','absolute x4 + braking gain + bridge endpoint velocity/acceleration',
                'paused_rejected_joint_recipe','../20261008/analysis/objective_v2/original_training_pause.json',
                last_update=r['paused_completed'],training_source_sha='b3cefdd994146526cf70f259010fce5077fc5692')
    for name in ['S0','S3']:
        for role in ['control','candidate']:
            attempt(name+'_v2_'+role,'objective_v2','original FM' if role=='control' else 'milder FM weights, scene-normalized braking, 1k curriculum, weak native rollout',
                    'paused_at_2000; no_5k_or_100k_claim','../20261008/analysis/objective_v2/final_report.json',
                    training_source_sha='6ec45b0e8705c7919fb555c20324f91580add976')
    changes={'G0':'same full geometry/Bt/FGTR structure; no future semantic/relation labels',
             'G1':'full future occupancy; uniform K1024 structured queries',
             'G2':'conditional enter/release; uniform K1024 structured queries',
             'G3':'same targets as G2; 50% q0 body/neighborhood + 25% boundary/event + 25% global for ALL structured supervision'}
    for (dataset, group), s in status.items():
        attempt(dataset+'/'+group,'G',changes[group[:2]],s['status'],'data/TRAINING_STATUS.csv',
                last_update=s['last_logged_update'],endpoint=s['registered_endpoint'],
                training_source_sha='0c54f1c49e7ed2a76b22826b654d6c72625f5b6d' if dataset=='navsim' else '41da7c198cf0ac4f799aee2a27a0a8126de30230')
    attempt('nuscenes/G0_NO_FUTURE_LABEL','G',changes['G0'],'prepared_not_started_then_paused_by_user',
            'evidence/NUSCENES_PAUSE.json',last_update=0)
    for name, change, result, path in [
        ('io_preserving_v1','Accepted IO execution; official runs retain original algorithm','adopted', 'reports/structured_world/round1_20261008/NAVTEST_25K_AND_TRAINING_EFFICIENCY_20261011.md'),
        ('engineering_fast','metadata, processor, transfers/logs, MLN/Fuser, queries reuse','faster_windows; formal_guard_still_closed','reports/structured_world/execution_20261011/PERFORMANCE_EVIDENCE_20261011_1405_ZH.md'),
        ('engineering_timevec','time batch vectorization','Bt_tolerance_failed; not_adopted','reports/structured_world/execution_20261011/IMPLEMENTATION_AND_ACCEPTANCE_ZH.md'),
        ('precision_fast','local geometry BF16','E3_screen_only; not_adopted','reports/structured_world/execution_20261011/IMPLEMENTATION_AND_ACCEPTANCE_ZH.md'),
        ('visual_cache_probe','frozen Qwen visual + all DeepStack','batch_shape_difference; no_full_cache','reports/structured_world/execution_20261012/NO_REGRESSION_HOST_FAST_AND_16GPU_ZH.md'),
        ('engineering_host_fast_and_16GPU','host-only preservation and 8-to-16 execution candidates','qualification/CPU_evidence; no_accepted_full_GPU_comparison','reports/structured_world/execution_20261012/NO_REGRESSION_HOST_FAST_AND_16GPU_ZH.md')]:
        attempt(name,'execution',change,result,path)
    dump('METHOD_LEDGER.json', attempts)
    write('METHOD_LEDGER.csv', attempts)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,2,figsize=(12.5,8.2),constrained_layout=True)
    panels=[('C', ['C/original/'+c for c in ['C0','C1','C4']]),
            ('S and action/video interfaces', ['S/original/'+s for s in ['S0','S3']] + ['A/V/planning_interface/'+s for s in ['A_NO_MAE','A_ACTION','V_QUERY']]),
            ('Optimized labels; original FM', ['S/r1_old_objective/'+s for s in ['R','S0','S3']]),
            ('Structured world / FGTR', ['G/'+s for s in ['G0','G1','G2','G3']])]
    for ax,(title,names) in zip(axes.flat,panels):
        for name in names:
            assert name in groups and groups[name], f'Empty plot series: {name}'
            vv=sorted(groups[name],key=lambda r:int(r['update']))
            ax.plot([int(r['update'])/1000 for r in vv],[float(r['PDMS']) for r in vv],marker='o',markersize=3,label=name.split('/')[-1])
        ax.set(title=title,xlabel='Optimizer updates (thousands)',ylabel='Navtest PDMS (0–100)')
        ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Completed observations only · one training seed · architecture and compute differ',fontsize=11)
    fig.savefig(DATA/'C_TO_G_LEARNING_CURVES.png',dpi=160)
    plt.close(fig)
    lines = ['# 各系列中间权重结果', '',
             '由归档 CSV 自动生成；PDMS 为0–100。空白代表没有该更新的完整评价。开发集与Navtest分开，q0仅为同模型诊断。', '',
             '[返回总览](README.md) · [完整分项CSV](data/ALL_CHECKPOINTS.csv) · [算法与实现](IMPLEMENTATION_GUIDE_ZH.md)', '']
    for split in ['navtest', 'dev']:
        sections = [
            ('C', ['C/original/'+c for c in ['C0','C1','C4']]),
            ('S', ['S/original/'+s for s in ['S0','S1','S2','S3','S4']]),
            ('A/V', ['A/V/planning_interface/'+s for s in ['A_ACTION','A_NO_MAE','V_QUERY']]),
            ('r1原FM与旧新目标', ['S/r1_old_objective/'+s for s in ['R','S0','S3']]+['S/r1_new_objective/S3','S/V12_new_objective/S3']),
            ('G执行轨迹', ['G/'+s for s in ['G0','G1','G2','G3']]),
            ('v2冷启动', ['objective_v2/'+s+'/'+r for s in ['S0','S3'] for r in ['control','candidate']])]
        for title, names in sections:
            selected=[r for r in ledger if r['split']==split and r['method_id'] in names and r['variant']!='q0']
            if not selected:
                continue
            lines += [f'## {split} · {title}', '', '| updates | '+' | '.join(names)+' |', '|---:|'+'---:|'*len(names)]
            for step in sorted({int(r['update']) for r in selected}):
                cells=[]
                for name in names:
                    rr=[r for r in selected if r['method_id']==name and int(r['update'])==step]
                    assert len(rr)<=1
                    cells.append(f"{float(rr[0]['PDMS']):.4f}" if rr else '—')
                lines.append('| '+str(step)+' | '+' | '.join(cells)+' |')
            lines.append('')
    (ROOT/'RESULTS_BY_CHECKPOINT.md').write_text('\n'.join(lines).rstrip()+'\n')
    print(json.dumps({'checkpoint_rows':len(ledger),'navtest_best_methods':len(best),'pilot_rows':len(pilots),
                      'method_ledger_entries':len(attempts),'cross_series_pairs':len(comparisons)}))


if __name__ == '__main__':
    main()
