"""Finite latest-checkpoint request using the qualified native Navtest workers.

The inherited loop schedules actual compute. Reporting runs once, directly after
all workers finish; there is no background result-watching/polling process.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.analysis.qwen_sourceflow.finite_navtest_scan import Owner, read, seal


LABELS = {'IL098000': ('il_interim', 98000), 'RL77_FINAL050000': ('rl', 50000),
          'DINO_FRESH024000': ('temporal', 24000)}


def validate_request(plan):
    body = dict(plan)
    if body.pop('sha256') != seal(body): raise ValueError('Changed evaluation registration')
    expected = dict(schema='native_latest_three_NAVTEST_v1', seed=2101, scenes_each=12146,
                    logs_each=136, metric='canonical_NAVSIM_v1.1',
                    inference='single_candidate_source_shift5_ODE_K10',
                    temporary_eval_host=None, total_new_single_candidate_inferences=36438)
    if any(plan.get(k) != v for k, v in expected.items()): raise ValueError('Changed protocol/budget/resource scope')
    if set(plan['models']) != set(LABELS): raise ValueError('Exactly three pinned checkpoints')
    for label, (kind, updates) in LABELS.items():
        m = plan['models'][label]
        if (m['kind'], m['updates']) != (kind, updates): raise ValueError('Checkpoint role/update mismatch')
        if kind == 'rl' and not m.get('parent_selection'): raise ValueError('RL must bind its exact IL77 parent')


def scoped_report_paths(porcelain, prefix):
    """Preserve porcelain leading status spaces; never strip the whole output."""
    paths = []
    for entry in porcelain.split('\0'):
        if not entry: continue
        if len(entry) < 4 or entry[2] != ' ' or 'R' in entry[:2] or 'C' in entry[:2]:
            raise ValueError('Unexpected rename/status; require explicit review')
        path = entry[3:]
        if not path.startswith(prefix + '/'):
            raise ValueError('Unrelated change: ' + path)
        paths.append(path)
    return paths


class LatestOwner(Owner):
    def __init__(self, plan):
        super().__init__(plan)
        self.order = ['RL77_FINAL050000', 'IL098000', 'DINO_FRESH024000']

    def enter_lease(self):
        root, atomic = self.root, self.atomic
        class FixedOnly:
            def close(self):
                atomic(root / 'lease_closed.json', dict(at=time.time(), fixed_hosts_only=True,
                       temporary_hosts=[], no_training_signaled=True))
        atomic(root / 'lease.json', dict(at=time.time(), fixed_hosts_only=True, temporary_hosts=[]))
        return FixedOnly()

    def probe(self):
        routes, inventory = super().probe()
        return [r for r in routes if r['host'] in self.plan['fixed_hosts']], inventory


def completed_summary(path):
    row = read(path); body = dict(row)
    if body.pop('sha256') != seal(body): raise ValueError('Changed summary')
    if row['status'] != 'COMPLETE' or row['scenes'] != 12146 or row['logs'] != 136:
        raise ValueError('Incomplete Navtest population')
    return row


def finalize(owner, report):
    start = time.time(); root, plan = owner.root, owner.plan
    done = read(root / 'COMPLETE.json')
    if done['status'] != 'COMPLETE' or not done['all_owned_workers_exited']:
        raise ValueError('Some workers failed; keep incomplete results explicitly incomplete')
    summaries = {label: completed_summary(root / 'analysis' / (label + '_summary.json')) for label in LABELS}
    references = plan['comparison_references']
    jobs = [
        ('RL50k_vs_own_IL77', 'analyze_paired', 'RL77_FINAL050000', references['RL77_parent']),
        ('RL50k_vs_RL24_6k', 'compare_rl_checkpoints', 'RL77_FINAL050000', references['RL77_previous']),
        ('IL98k_vs_IL80k', 'analyze_il_milestones', 'IL098000', references['IL_previous']),
        ('DINO24k_vs_DINO14k', 'analyze_il_milestones', 'DINO_FRESH024000', references['DINO_previous']),
    ]
    analyses = {}
    checkout = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name, module, label, reference in jobs:
        target = root / 'analysis' / (name + '.json')
        with (root / 'analysis' / (name + '.log')).open('x') as log:
            subprocess.run([owner.resource.MODEL_PYTHON, '-m', 'scripts.analysis.qwen_sourceflow.' + module,
                '--evaluation', plan['models'][label]['output'], '--reference', reference,
                '--output', str(target)], cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        analyses[name] = completed_summary(target)
    cost = {key: sum(s[key] for s in summaries.values()) for key in [
        'logical_single_candidates', 'physical_inference_including_retries', 'worker_wall_seconds',
        'model_loading_seconds', 'forward_worker_seconds', 'scoring_wait_worker_seconds']}
    cost.update(elapsed_compute_owner_seconds=done['at'] - read(root / 'owner.json')['at'],
        analysis_wall_seconds=time.time() - start,
        semantics='Shared worker occupancy, not exclusive GPU compute; worker sums do not equal job wall.')
    result = dict(schema='latest_three_native_navtest_results_v1', status='COMPLETE',
        registration_sha256=plan['sha256'], summaries=summaries, analyses=analyses, costs=cost,
        uncertainty='One training/inference seed; scene-weighted paired log bootstrap only.',
        Test_observed=True, no_training_changes=True)
    result['sha256'] = seal(result); report.mkdir(parents=True, exist_ok=True)
    owner.atomic(report / 'results.json', result)
    for name, row in analyses.items(): owner.atomic(report / (name + '.json'), row)
    lines = ['# Latest Qwen-Flow Navtest — 2026-10-27', '',
        'All three full12146-scene/136-log evaluations completed. Canonical NAVSIM v1.1; '
        'single-candidate source shift5 ODE K10; inference seed2101. No best-of-N or training change.', '',
        '| Checkpoint | Updates | Full Navtest PDMS |', '|---|---:|---:|']
    lines += [f'| {label} | {plan["models"][label]["updates"]:,} | {s["PDMS"]:.6f} |' for label, s in summaries.items()]
    lines += ['', '| Matched paired comparison | Net PDMS points | 95% log-bootstrap interval | New zero / recovered zero |',
              '|---|---:|---|---:|']
    for name, a in analyses.items():
        b = a['PDMS_log_cluster_bootstrap']; lo, hi = [100*x for x in b['CI95']]
        lines.append(f'| {name} | {100*b["mean"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {a["nonzero_to_zero"]} / {a["zero_to_nonzero"]} |')
    lines += ['', 'NC/DAC/TTC new violations/recoveries, G+/G− and components are in the paired JSON. '
              'Intervals exclude seed and Test checkpoint-selection uncertainty. The completed50k RL endpoint '
              'is compared with its exact IL77 parent and its own24.6k intermediate; old IL53-parent models '
              'are not treated as matched controls. DINO is fresh three-frame IL, distinct from the paused warm-start model.', '',
              f'Logical single candidates: {cost["logical_single_candidates"]:,}; physical including retries: '
              f'{cost["physical_inference_including_retries"]:,}. Compute owner wall: '
              f'{cost["elapsed_compute_owner_seconds"]/3600:.3f} h; summed shared worker occupancy: '
              f'{cost["worker_wall_seconds"]/3600:.3f} h. Export/native parity and historical calibration costs '
              'are separate job receipts; they are not free data.', '',
              'Frozen registration: [registration.json](registration.json). Full data: [results.json](results.json).', '']
    (report / 'README.md').write_text('\n'.join(lines))
    return result


def publish(owner, report):
    checkout = Path(__file__).resolve().parents[3]; launch = read(owner.root / 'launch.json')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=checkout, text=True)
    if git('rev-parse', 'HEAD').strip() != launch['source_sha'] or git('branch', '--show-current').strip() != launch['branch']:
        raise ValueError('Source/branch changed; report requires explicit publication review')
    prefix = str(report.relative_to(checkout))
    scoped_report_paths(git('status', '--porcelain=v1', '-z', '--untracked-files=all'), prefix)
    subprocess.run(['git', 'add', '--', prefix], cwd=checkout, check=True)
    subprocess.run(['git', 'commit', '-m', 'Record complete latest IL98k RL50k and DINO24k Navtest'], cwd=checkout, check=True)
    commit = git('rev-parse', 'HEAD').strip()
    subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/' + launch['branch']], cwd=checkout, check=True)
    remote = git('ls-remote', 'origin', 'refs/heads/' + launch['branch']).split()[0]
    if remote != commit: raise ValueError('Remote SHA mismatch after publication')
    owner.atomic(owner.root / 'PUBLICATION.json', dict(status='PUBLISHED', at=time.time(), commit=commit, remote_SHA=remote))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--registration', required=True)
    p.add_argument('--report-dir', required=True); a = p.parse_args()
    plan = read(a.registration); validate_request(plan); root = Path(plan['root'])
    lock = (root / 'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'owner.json').exists(): raise FileExistsError('Do not duplicate a registered owner')
    owner = LatestOwner(plan)
    owner.atomic(root / 'owner.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]), at=time.time()))
    (root / 'parity').mkdir(exist_ok=True)
    owner.run()
    try:
        finalize(owner, Path(a.report_dir)); publish(owner, Path(a.report_dir))
    except BaseException:
        import traceback
        owner.atomic(root / 'report_failure.json', dict(at=time.time(), error=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__': main()
