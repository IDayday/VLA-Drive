"""Exact user-selected RL77 30k/40k/45k request; native inference is unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.analysis.qwen_sourceflow.finite_navtest_scan import read, seal
from scripts.analysis.qwen_sourceflow.latest_navtest_request import LatestOwner, completed_summary, scoped_report_paths

LABELS = {'RL77_030000': 30000, 'RL77_040000': 40000, 'RL77_045000': 45000}
HOSTS = ['training-vlawm-tsinghua-zt', 'training-vlawm-tsinghua-zt3', 'training-vlawm-tsinghua-zt4']


def validate_request(plan):
    body = dict(plan)
    if body.pop('sha256') != seal(body): raise ValueError('Changed registration')
    expected = dict(schema='native_selected_RL77_three_NAVTEST_v1', seed=2101,
        scenes_each=12146, logs_each=136, metric='canonical_NAVSIM_v1.1',
        inference='single_candidate_source_shift5_ODE_K10', temporary_eval_host=None,
        total_new_single_candidate_inferences=36438, fixed_hosts=HOSTS,
        max_evaluation_workers_per_model=8)
    if any(plan.get(k) != v for k, v in expected.items()): raise ValueError('Changed protocol or routing scope')
    if set(plan['models']) != set(LABELS): raise ValueError('Only the three user-selected checkpoints')
    parents = set()
    for label, update in LABELS.items():
        model = plan['models'][label]
        if model['kind'] != 'rl' or model['updates'] != update: raise ValueError('Wrong checkpoint')
        parents.add((model['parent_checkpoint'], model['parent_selection']))
    if len(parents) != 1: raise ValueError('Different IL parents')


class SelectedOwner(LatestOwner):
    def __init__(self, plan):
        super().__init__(plan)
        self.order = list(LABELS)


def finalize(owner, report):
    root, plan = owner.root, owner.plan
    done = read(root / 'COMPLETE.json')
    if done['status'] != 'COMPLETE' or not done['all_owned_workers_exited']:
        raise ValueError('All native workers must finish before results')
    rows = {k: completed_summary(root / 'analysis' / (k + '_summary.json')) for k in LABELS}
    checkout = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    comparisons = {}
    for label in LABELS:
        for suffix, module, reference in [
            ('vs_own_IL77', 'analyze_paired', plan['comparison_references']['RL77_parent']),
            ('vs_fixed_RL50k', 'compare_rl_checkpoints', plan['comparison_references']['RL77_final'])]:
            name = label + '_' + suffix
            target = root / 'analysis' / (name + '.json')
            with (root / 'analysis' / (name + '.log')).open('x') as log:
                subprocess.run([owner.resource.MODEL_PYTHON, '-m', 'scripts.analysis.qwen_sourceflow.' + module,
                    '--evaluation', plan['models'][label]['output'], '--reference', reference,
                    '--output', str(target)], cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            comparisons[name] = completed_summary(target)
    costs = {key: sum(r[key] for r in rows.values()) for key in [
        'logical_single_candidates', 'physical_inference_including_retries', 'worker_wall_seconds',
        'model_loading_seconds', 'forward_worker_seconds', 'scoring_wait_worker_seconds']}
    costs['compute_owner_wall_seconds'] = done['at'] - read(root / 'owner.json')['at']
    costs['semantics'] = 'Shared worker occupancy, not exclusive GPU computation; worker sum is not job wall.'
    result = dict(status='COMPLETE', registration_sha256=plan['sha256'], summaries=rows,
        comparisons=comparisons, costs=costs, Test_observed=True,
        uncertainty='Single training/inference seed; paired log-bootstrap excludes checkpoint-selection uncertainty.')
    result['sha256'] = seal(result)
    report.mkdir(parents=True, exist_ok=True)
    owner.atomic(report / 'results.json', result)
    owner.atomic(report / 'registration.json', plan)
    lines = ['# IL77-parent RL30k/40k/45k full Navtest', '',
        'User-selected Test observations; full12146 scenes/136 manifest log fragments per model. '
        'Canonical NAVSIM v1.1, single-candidate source shift5 ODE K10, seed2101. '
        'No training, sampler, scoring, checkpoint-selection or additional-seed changes.', '',
        '| Checkpoint | PDMS |', '|---|---:|']
    lines += [f'| {label} | {row["PDMS"]:.6f} |' for label, row in rows.items()]
    lines += ['', 'Matched safety transitions, G+/G− and log-cluster intervals against the exact IL77 '
        'parent and fixed RL50k endpoint are in results.json. These additional Test observations '
        'are exploratory; they do not change the fixed50k endpoint or IL parent.', '']
    (report / 'README.md').write_text('\n'.join(lines))
    launch = read(root / 'launch.json')
    def git(*args): return subprocess.check_output(['git', *args], cwd=checkout, text=True)
    if git('rev-parse', 'HEAD').strip() != launch['source_sha']: raise ValueError('Changed reporting source')
    prefix = str(report.relative_to(checkout))
    scoped_report_paths(git('status', '--porcelain=v1', '-z', '--untracked-files=all'), prefix)
    subprocess.run(['git', 'add', '--', prefix], cwd=checkout, check=True)
    subprocess.run(['git', 'commit', '-m', 'Record user-selected RL77 30k 40k 45k full Navtest'], cwd=checkout, check=True)
    commit = git('rev-parse', 'HEAD').strip()
    branch = launch['branch']
    subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/' + branch], cwd=checkout, check=True)
    remote = git('ls-remote', 'origin', 'refs/heads/' + branch).split()[0]
    if remote != commit: raise ValueError('Remote SHA mismatch')
    owner.atomic(root / 'PUBLICATION.json', dict(status='PUBLISHED', at=time.time(), commit=commit, remote_SHA=remote))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', required=True); parser.add_argument('--report-dir', required=True)
    args = parser.parse_args(); plan = read(args.registration); validate_request(plan)
    root = Path(plan['root']); lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'owner.json').exists(): raise FileExistsError('Explicit new attempt required')
    owner = SelectedOwner(plan)
    owner.atomic(root / 'owner.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]), at=time.time()))
    (root / 'parity').mkdir(exist_ok=True)
    owner.run()
    try: finalize(owner, Path(args.report_dir))
    except BaseException:
        owner.atomic(root / 'report_failure.json', dict(at=time.time(), error=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__': main()
