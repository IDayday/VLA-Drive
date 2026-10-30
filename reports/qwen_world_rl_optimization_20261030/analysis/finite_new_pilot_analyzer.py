"""Finite, event-driven CPU review of two registered pilots at 1k/5k/10k.

Waits on directory inotify metadata events, never training journals. A pair is
analyzed only after both complete formal NAVTEST summaries exist. No GPU/model
load, dispatch, stop, score imputation or automatic training decision.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from finite_matched12k_analyzer import MetadataEvents, atomic, fingerprint, read, sha

ROOT = Path('/mnt/project/qwen-world-rl-optimization-20261030')
ANALYSIS = ROOT/'analysis'
OWNER_ROOT = ROOT/'owners/NEW_PILOT_MATCHED_ANALYSIS'
RESULT = ANALYSIS/'NEW_PILOT_MATCHED_REVIEW.json'
ARMS = ('PPO_MEMORYLR20', 'PPO_SAFETY_EP')
BUDGETS = (1000, 5000, 10000)
PAIRS = [(f'{a}_u{u:06d}_NAVTEST', f'PPO_FULL_u{u:06d}_NAVTEST') for a in ARMS for u in BUDGETS]
LABELS = tuple(dict.fromkeys(label for pair in PAIRS for label in pair))
NOW = lambda:datetime.now(timezone.utc).isoformat()


class PilotMetadataEvents(MetadataEvents):
    def discover(self, specs):
        for p in (ROOT, ROOT/'owners/CAMPAIGN', ROOT/'evaluation_specs', ROOT/'jobs'):
            self.add(p)
        for arm in ARMS:
            self.add(ROOT/'owners'/arm)
            family = ROOT/'jobs'/arm
            self.add(family)
            if family.exists():
                for attempt in family.glob('formal_to*_attempt*'):
                    self.add(attempt)
        for label in LABELS:
            family = ROOT/'jobs'/('EVAL_'+label)
            self.add(family)
            if family.exists():
                for attempt in family.glob('rank*_attempt*'):
                    self.add(attempt)
        for spec in specs.values():
            if spec:
                output = Path(spec['output'])
                self.add(output.parent)
                self.add(output)

    def wait(self, remaining):
        # Use the inherited inotify reader with this finite label list, without
        # mutating its old module globals (its wait filter is probe-specific).
        import select
        ready, _, _ = select.select([self.fd], [], [], max(0, remaining))
        if not ready:
            return []
        changed = []
        while True:
            try:
                raw = os.read(self.fd, 262144)
            except BlockingIOError:
                break
            pos = 0
            while pos < len(raw):
                wd, mask, cookie, length = self.HEADER.unpack_from(raw, pos)
                pos += self.HEADER.size
                name = os.fsdecode(raw[pos:pos+length].split(b'\0', 1)[0]); pos += length
                if mask & 0x4000:
                    changed.append('INOTIFY_OVERFLOW'); continue
                directory = self.paths.get(wd)
                if directory is None:
                    continue
                if mask & 0x40000000:
                    if (name in ARMS or name in LABELS or name in ['EVAL_'+x for x in LABELS]
                            or name.startswith(('rank', 'formal_to'))):
                        changed.append(str(directory/name))
                elif (name in ('STATE.json', 'OWNER.json', 'SUMMARY.json', 'RESULT.json', 'RUNNING.json')
                      or name in [x+'.json' for x in LABELS]):
                    changed.append(str(directory/name))
        return sorted(set(changed))


def scientific_child_failure(result):
    """Require positive evidence of a model child failure, never absence/proxy."""
    if not result:
        return False
    code = result.get('returncode', result.get('exit_code'))
    if (code in (None, 0, 75) or result.get('child_not_started') is True
            or result.get('scientific_result_unknown') is True
            or result.get('forced_expiry') is True
            or result.get('expiry_guard_requested') is True
            or result.get('status') in ('REFUSED_RESOURCE', 'HELD_RESOURCE_LEASE',
                                       'FORCED_EXIT_REVIEW_REQUIRED', 'TRANSPORT_EXIT_WITHOUT_RESULT')):
        return False
    return result.get('status') == 'FAILED' and result.get('child_pid') is not None


def failure_evidence(arm, node, specs):
    """Only current training and current-rank eval attempts; preserve old proof."""
    candidates = []
    if node.get('job'):
        candidates.append(Path(node['job']))
    for u in BUDGETS:
        spec = specs.get(f'{arm}_u{u:06d}_NAVTEST')
        if not spec:
            continue
        latest = {}
        for job in spec.get('jobs', []):
            p = Path(job['receipt'] if isinstance(job, dict) else job)
            match = re.fullmatch(r'rank(\d+)_attempt(\d+)', p.name)
            key, attempt = (int(match[1]), int(match[2])) if match else (str(p), 0)
            if key not in latest or attempt > latest[key][0]:
                latest[key] = (attempt, p)
        candidates.extend(p for attempt, p in latest.values())
    for p in candidates:
        value = read(p/'RESULT.json')
        if scientific_child_failure(value):
            return dict(kind='confirmed_child_failure_requires_review', receipt=str(p),
                        result_path=str(p/'RESULT.json'), result_sha256=sha(p/'RESULT.json'), result=value,
                        interpretation='实际模型child失败；不据此判定算法无效，当前臂剩余配对跳过，另一臂继续。')
    return None


def current_metadata():
    campaign = read(ROOT/'owners/CAMPAIGN/STATE.json', {})
    nodes = {}
    for arm in ARMS:
        nodes[arm] = read(ROOT/'owners'/arm/'STATE.json') or campaign.get('runs', {}).get(arm, {})
    specs = {label:read(ROOT/'evaluation_specs'/(label+'.json')) for label in LABELS}
    ready, invalid = set(), {}
    for label, spec in specs.items():
        if not spec:
            continue
        path = Path(spec['output'])/'SUMMARY.json'; summary = read(path)
        if not summary or summary.get('status') != 'COMPLETE':
            continue
        updates = int(label.split('_u')[-1].split('_')[0])
        if (summary.get('scenes') != 12146 or summary.get('invalid_scenes', 0) != 0
                or summary.get('RL_updates') != updates):
            invalid[label] = dict(kind='invalid_complete_formal_summary', path=str(path),
                                  summary_sha256=sha(path), summary=summary)
        else:
            ready.add(label)
    return nodes, specs, ready, invalid


def load_value(label):
    """Adapt old1k compact cache; use the standard once-only loader otherwise."""
    import numpy as np
    from analyze_portfolio_10k import CACHE, METRICS, inputs, load_once, file_sha
    if label != 'PPO_FULL_u001000_NAVTEST' or (CACHE/(label+'.json')).exists():
        return load_once(label)
    old_path = ANALYSIS/'MATCHED_1K_PPO_metrics.npz'
    proof_path = ANALYSIS/'MATCHED_1K_PPO_CONTRASTS.json'
    out, summary_path, summary, identity = inputs(label)
    from analyze_NFT_u001000_vs_IL70 import sealed
    proof = sealed(proof_path) if proof_path.exists() else None
    if not proof or proof.get('status') != 'COMPLETE' or not old_path.exists():
        raise ValueError('Required existing1k compact cache/proof missing')
    provenance = proof['provenance']['inputs']['PPO_FULL']
    if (provenance['summary']['seal'] != summary['sha256']
            or provenance['identity']['seal'] != identity['sha256']):
        raise ValueError('Old1k metric cache ancestry differs')
    with np.load(old_path) as a:
        if (a['tokens'].tolist() != identity['tokens'] or a['metric_keys'].tolist() != list(METRICS)
                or abs(float(a['PPO_FULL'][:, 0].mean())*100-summary['PDMS']) > 1e-10):
            raise ValueError('Old1k metric population differs')
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE/(label+'.npz')
        np.savez_compressed(path, metrics=a['PPO_FULL'], tokens=a['tokens'], logs=a['logs'],
                            cache_hashes=a['cache_hashes'], metric_keys=a['metric_keys'])
    atomic(CACHE/(label+'.json'), dict(
        schema='exact_canonical_evaluation_metric_cache_v1', label=label,
        identity_sha256=identity['sha256'], summary_sha256=summary['sha256'], npz_sha256=file_sha(path),
        scene_raw_hash_inventory_sha256=provenance['scene_raw_hash_inventory_sha256'],
        units='official metric values0..1; exact FP64 JSON values', shape=[12146, len(METRICS)],
        metric_keys=list(METRICS), adapted_from=dict(path=str(old_path), sha256=file_sha(old_path),
            proof=str(proof_path), proof_sha256=file_sha(proof_path), scene_reparse=False)))
    return load_once(label)


def pair_review(target, baseline):
    from analyze_portfolio_10k import contrast
    from analyze_portfolio_20k import component_intervals
    result = component_intervals(target, baseline, contrast(target, baseline))
    result['status'] = 'COMPLETE'
    result['completed_UTC'] = NOW()
    ci = result['CI95_PDMS']
    negative = [k for k, v in result['safety_metrics'].items()
                if k in ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')
                and v['CI95_delta_percentage_points'][1] < 0]
    if negative:
        recommendation = '安全指标存在配对下降证据；优先检查新增failure及效率交换，扩预算前由root审查，不因总分上涨自动推广。'
    elif ci[1] < 0:
        recommendation = '同预算总分负差且区间排除零；建议root评估机会成本并考虑停止/修订，不由分析器执行。'
    elif ci[0] > 0:
        recommendation = '同预算总分正差且区间排除零；支持在既定pilot内继续观察持续性。安全无显著下降不等于证明安全。'
    else:
        recommendation = '总分配对区间跨零；当前不能确认优于FULL，按预注册下一预算观察，不根据点排名推广。'
    result['review'] = dict(recommendation_zh=recommendation, safety_decrease_CI_excludes_zero=negative,
        no_automatic_training_action=True,
        scope_zh='固定IL70、训练seed42与评测seed2101；只能检验此单因素pilot，不能推出knowledge sufficient或已具备场景风险推理。')
    return result


def refresh(status):
    atomic(ROOT/'reporting/refresh_request.json', dict(
        id=f'NEW_PILOT_MATCHED_{status}_{NOW()}', scientific_result=False,
        reason='新pilot同预算完整Navtest配对分析增量更新：'+status,
        evidence=str(RESULT), evidence_kind='completed_supplemental_analysis', analysis_status=status))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--max-wait-seconds', type=float, default=9*3600)
    args = parser.parse_args()
    OWNER_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (OWNER_ROOT/'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time(); deadline = start+args.max_wait_seconds
    registration = {a:ROOT/f'EXPERIMENT_{a}_REGISTRATION.json' for a in ARMS}
    for arm, path in registration.items():
        value = read(path)
        if not value or value.get('arm') != arm or value.get('pilot_budget_updates') != 10000 or value.get('native_scheduler_max_updates') != 50000:
            raise ValueError('Registered pilot identity/budget mismatch: '+arm)
    owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
        source_file=str(Path(__file__).resolve()), source_sha256=sha(__file__), started_UTC=NOW(),
        deadline_UTC=datetime.fromtimestamp(deadline, timezone.utc).isoformat(), CPU_only=True,
        targets=[dict(target=a, baseline=b) for a, b in PAIRS],
        registrations={a:dict(path=str(p), sha256=sha(p)) for a, p in registration.items()},
        event_backend='Linux directory inotify; STATE/spec/SUMMARY/exact worker RESULT atomic metadata events',
        journal_polling=False, GPU_process_actions=0)
    atomic(OWNER_ROOT/'OWNER.json', owner)
    previous = read(RESULT)
    if previous:
        atomic(OWNER_ROOT/'STATE.json', dict(status='EXISTING_RESULT_REQUIRES_REVIEW', utc=NOW(), evidence=str(RESULT)))
        return 2
    result = dict(schema='world_new_pilot_matched_review_v1', status='WAITING_SUMMARIES', utc=NOW(),
        contrasts={}, evaluations={}, arm_failures={}, pending_pairs=[a+'_minus_'+b for a, b in PAIRS],
        completed_pairs=0, target_pairs=6, provenance=owner,
        matching_evidence=dict(scenes=12146, physical_logs=44, bootstrap=2000, bootstrap_seed=42,
            evaluation_seed=2101, training_seed=42, component_CIs_same_pooled_physical_log_draws=True,
            matched_budget_only=True, formal_NAVTEST_only=True, inference='single-candidate ODE10 shift5'),
        interpretation_zh='每组仅在双方完整正式结果齐备后写入；单seed物理log簇bootstrap不是跨训练重复不确定性，未校正多重比较。NAVTEST已用于研究选择。',
        knowledge_scope_zh='两个pilot均固定IL70知识基础；MemoryLR20只变适应强度，SafetyEP只分解RL信用，不能推广为知识已充分或场景风险输入已充分。',
        scope=dict(CPU_only=True, new_forwards=0, new_scores=0, new_seeds=0,
            existing_compact_caches_reused=True, each_uncached_complete_population_read_at_most_once=True,
            partial_scores_imputed=False, automatic_training_actions=False, GPU_process_actions=0))
    notify = PilotMetadataEvents(); values = {}; previous_state = None

    def publish():
        result['utc'] = NOW(); result.pop('sha256', None); result['sha256'] = fingerprint(result)
        atomic(RESULT, result)
        atomic(OWNER_ROOT/'STATE.json', dict(status=result['status'], utc=NOW(), evidence=str(RESULT),
            completed_pairs=result['completed_pairs'], pending_pairs=result['pending_pairs'],
            arm_failures=result['arm_failures'], GPU_process_actions=0))

    try:
        publish()
        notify.discover({})
        while True:
            nodes, specs, ready, invalid = current_metadata()
            watch_count = len(notify.paths)
            notify.discover(specs)
            # Close the registration race if a summary arrived just before its
            # output directory acquired a watch. This is event setup, not polling.
            if len(notify.paths) != watch_count:
                nodes, specs, ready, invalid = current_metadata()
            # Complete pairs first, even if a later stage on the same arm fails.
            for target, baseline in PAIRS:
                key = target+'_minus_'+baseline
                arm = target.split('_u')[0]
                if (arm in result['arm_failures'] or key in result['contrasts']
                        or target not in ready or baseline not in ready):
                    continue
                # Shared control integrity failures require global review. A
                # pilot's own invalid evidence never blocks the other pilot.
                if baseline not in values:
                    values[baseline] = load_value(baseline)
                    result['evaluations'][baseline] = dict(PDMS=values[baseline]['summary']['PDMS'],
                        RL_updates=values[baseline]['identity']['RL_updates'], provenance=values[baseline]['provenance'])
                try:
                    if target not in values:
                        values[target] = load_value(target)
                        result['evaluations'][target] = dict(PDMS=values[target]['summary']['PDMS'],
                            RL_updates=values[target]['identity']['RL_updates'], provenance=values[target]['provenance'])
                    comparison = pair_review(values[target], values[baseline])
                except Exception:
                    result['arm_failures'][arm] = dict(kind='pilot_pair_evidence_requires_review',
                        pair=key, error=traceback.format_exc(), scientific_effect_failure_not_inferred=True)
                    publish(); refresh('PILOT_EVIDENCE_REQUIRES_REVIEW'); continue
                result['contrasts'][key] = comparison
                result['completed_pairs'] = len(result['contrasts'])
                result['pending_pairs'].remove(key)
                result['status'] = 'PARTIAL'
                publish(); refresh('PAIR_COMPLETE')
                print(json.dumps(dict(utc=NOW(), event='pair_complete', pair=key)), flush=True)
            for arm in ARMS:
                if arm in result['arm_failures']:
                    continue
                evidence = next((x for label, x in invalid.items() if label.startswith(arm+'_u')), None)
                evidence = evidence or failure_evidence(arm, nodes[arm], specs)
                if evidence:
                    result['arm_failures'][arm] = evidence
            pending = [a+'_minus_'+b for a, b in PAIRS if a+'_minus_'+b not in result['contrasts']
                       and not any(a.startswith(arm+'_u') for arm in result['arm_failures'])]
            result['pending_pairs'] = pending
            result['skipped_pairs'] = [a+'_minus_'+b for a, b in PAIRS if a+'_minus_'+b not in result['contrasts'] and a+'_minus_'+b not in pending]
            snapshot = (tuple(pending), tuple(result['arm_failures']), result['completed_pairs'])
            if not pending:
                result['status'] = 'COMPLETE' if result['completed_pairs'] == 6 else 'COMPLETE_WITH_ARM_FAILURES'
                publish(); refresh(result['status']); return 0
            if time.time() >= deadline:
                result['status'] = 'WAIT_TIMEOUT'
                result['timeout_zh'] = '9小时有限等待结束；保留已完整配对，未完成不补点，训练不受影响。'
                publish(); refresh('WAIT_TIMEOUT'); return 0
            if snapshot != previous_state:
                if result['completed_pairs'] == 0:
                    result['status'] = 'WAITING_SUMMARIES'
                publish(); previous_state = snapshot
            changed = notify.wait(deadline-time.time())
            if changed:
                print(json.dumps(dict(utc=NOW(), metadata_events=changed[:20])), flush=True)
    except Exception:
        result['status'] = 'ANALYSIS_FAILED_REQUIRES_REVIEW'
        result['analysis_error'] = traceback.format_exc()
        publish(); refresh(result['status']); print(result['analysis_error'], file=sys.stderr, flush=True)
        return 2
    finally:
        notify.close()


if __name__ == '__main__':
    raise SystemExit(main())
