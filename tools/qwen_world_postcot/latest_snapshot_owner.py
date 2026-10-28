"""Finite user-pinned World observations, qualified native workers, one publisher."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import traceback

from tools.qwen_world_postcot.user_interim_owner import Request, read, write, fingerprint, sha

LOCK = threading.RLock()
RESERVED = {}


class SnapshotRequest(Request):
    def __init__(self, root, pin):
        from starVLA.rl.qwen_sourceflow import resources
        self.resources = resources; self.root = root; self.pin = pin
        self.source = Path(__file__).resolve().parents[2]
        self.source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.source, text=True).strip()
        self.export = root/'export'; self.output = root/'evaluation'; self.jobs = {}
        self.reg = root/'registration.json'; self.proof = root/'native_export_parity.json'; self.world = 8

    def done(self, name):
        result = super().done(name)
        if result is not None:
            with LOCK:
                for uuid, value in list(RESERVED.items()):
                    if value == (str(self.root), name): RESERVED.pop(uuid)
        return result

    def route(self, routes, inventory, *, parity=False):
        with LOCK:
            route = super().route([r for r in routes if r['uuid'] not in RESERVED], inventory, parity=parity)
            if route: RESERVED[route['uuid']] = (str(self.root), None)
            return route

    def submit(self, name, module, args, route=None):
        args = list(args)
        if module.endswith('.export') and self.pin['kind'] == 'RL':
            args = ['--checkpoint', self.pin['parent_checkpoint'], '--rl-checkpoint', self.pin['checkpoint'],
                    '--output', str(self.export)]
        args = ['--user-snapshot-registration' if v == '--user-interim-registration' else v for v in args]
        if route:
            with LOCK: RESERVED[route['uuid']] = (str(self.root), name)
        try: return super().submit(name, module, args, route)
        except BaseException:
            if route:
                with LOCK: RESERVED.pop(route['uuid'], None)
            raise

    def register(self):
        from starVLA.rl.qwen_world_postcot.snapshot_navtest import policy_binding
        meta = read(self.export/'export.json'); seal = read(self.export/'COMPLETE')
        if seal['export_sha256'] != sha(self.export/'export.json'):
            raise ValueError('Changed export receipt')
        binding = policy_binding(meta['IL_parent'], meta)
        if any(binding[k] != self.pin[k] for k in ('checkpoint', 'updates')):
            raise ValueError('Export is not the pinned user-requested policy')
        if binding['checkpoint_files_sha256'] != self.pin['files_sha256']:
            raise ValueError('Native pinned inventory changed')
        data = Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST')
        manifest = read(data/'manifest.json'); population = Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
        if manifest['tokens'] != read(population)['navtest_tokens'] or len(manifest['tokens']) != 12146:
            raise ValueError('Changed full Test population')
        protocol = read('/mnt/project/qwen-world-postcot-20261025/evaluations/DEV_IL_update035000/identity.json')['protocol']
        row = dict(schema='world_user_snapshot_NAVTEST_v1', **binding,
            IL_parent_checkpoint=meta['IL_parent']['path'], IL_parent_model_sha256=meta['IL_parent']['model_sha256'],
            model_sha256=meta['model_sha256'], data_sha256=sha(data/'manifest.json'),
            token_sha256=fingerprint(manifest['tokens']), population_sha256=sha(population),
            protocol_sha256=protocol['sha256'], output=str(self.output.resolve()), seed=2101, count=12146,
            prediction_type='x0', inference='single_candidate_cleanx_K10_shift5',
            used_for_parent_selection=False, used_for_actor_update=False,
            user_instruction=self.pin.get('user_instruction', '统计目前的训练进度，新的ckpts可以用于评测'),
            Test_observed=True)
        row['sha256'] = fingerprint(row); write(self.reg, row)

    def publish(self, summary):
        # Keep the git checkout clean until ALL worker invocations have exited.
        write(self.root/'FINAL_MODEL_RESULT.json', dict(summary=summary,
              registration=read(self.reg), native_parity=read(self.proof)))


def finalize(root, requests, report):
    if read(root/'PINNED.json').get('schema') == 'world_checkpoint_sweep_NAVTEST_v1':
        return finalize_sweep(root, requests, report)
    from tools.qwen_world_postcot.summarize import summarize
    reference = '/mnt/project/qwen-sourceflow-native-20261018/selected_WORLD40_RL77_NAVTEST_20261027T0940Z/WORLD_IL040000/evaluation'
    results = {}; report.mkdir(parents=True, exist_ok=True)
    for label, request in requests.items():
        done = read(request.root/'COMPLETE.json')
        if done['status'] != 'COMPLETE' or not done['all_owned_workers_exited']:
            raise ValueError('Incomplete native workers cannot become a main result')
        result = summarize(request.output, reference)
        results[label] = result; write(report/(label+'.json'), result)
        write(report/(label+'_native_parity.json'), read(request.proof))
    lines = ['# World latest IL/RL Navtest snapshots', '',
        'Canonical NAVSIM v1.1; full12146 scenes; single clean-x K10 shift5 ODE; seed2101. '
        'All three RL policies share IL40k as parent. IL52k is a separate IL continuation, not a replacement RL control.', '',
        '| Snapshot | Updates | Full Navtest PDMS |', '|---|---:|---:|']
    for label, request in requests.items(): lines.append(f'| {label} | {request.pin["updates"]} | {results[label]["PDMS"]:.6f} |')
    lines += ['', 'Paired IL40k comparison, native submetrics, new/recovered failures, G+/G−, '
              'log-cluster bootstrap and recorded worker costs are in the JSON files. '
              'This is one training/inference seed and different intermediate update budgets; '
              'it does not establish a matched endpoint ranking of methods. '
              'No candidate oracle, teacher or scorer selects the deployed trajectory. '
              'Test has been observed historically and these observations are not claimed blind.', '']
    (report/'README.md').write_text('\n'.join(lines))
    return results


def finalize_sweep(root, requests, report):
    """Derived comparisons only; all native prediction/scoring paths are reused."""
    from tools.qwen_world_postcot.summarize import summarize
    pinned = read(root/'PINNED.json')
    reference = pinned['reference_evaluation']
    results = {}; report.mkdir(parents=True, exist_ok=True)
    loss_study=pinned.get('loss_designs_study')
    lines = ['# Qwen-World IL checkpoints and later RL observations', '',
        'Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. '
        'IL observations do not change the fixed IL40k RL parent. Test is historically observed.', '',
        '| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same RL (points) |',
        '|---|---:|---:|---:|---:|']
    if loss_study:
        lines=['# Qwen-World independent IL loss designs', '',
            'Full12146 canonical v1.1 Test scenes, seed2101, single clean-x K10 shift5 ODE. '
            'All forks use the same63k native model/Adam/RNG/data boundary and2000 added updates. '
            'The shared CONTROL is original mean L1, not a new weak IL baseline. Test has been observed.', '',
            '| Fixed endpoint | Updates | PDMS | Change vs matched CONTROL65k (points) | Previous RL comparison |',
            '|---|---:|---:|---:|---|']
    for label, request in requests.items():
        done = read(request.root/'COMPLETE.json')
        if done['status'] != 'COMPLETE' or not done['all_owned_workers_exited']:
            raise ValueError('Incomplete native workers cannot become a Test result')
        result = summarize(request.output, reference)
        previous = request.pin.get('previous_evaluation')
        earlier_delta = None
        if previous:
            earlier = summarize(request.output, previous)
            result['paired_previous_same_RL'] = dict(reference_evaluation=previous, **earlier['paired'])
            earlier_delta = 100 * earlier['paired']['mean_delta']
        # Additional derived fields require a new seal, not the original summary seal.
        result.pop('sha256')
        result['sha256'] = fingerprint(result)
        results[label] = result
        write(report/(label+'.json'), result)
        write(report/(label+'_native_parity.json'), read(request.proof))
        earlier_text = '—' if earlier_delta is None else f'{earlier_delta:+.6f}'
        lines.append(f'| {label} | {request.pin["updates"]} | {result["PDMS"]:.6f} | '
                     f'{100 * result["paired"]["mean_delta"]:+.6f} | {earlier_text} |')
    if loss_study:
        study_root=Path(loss_study['root'])
        for original,destination in [('REGISTRATION.json','registration.json'),('calibration/CALIBRATION.json','TRAIN_calibration.json')]:
            write(report/destination,read(study_root/original))
        lines+=['', 'Only one loss factor changes per arm: bounded TRAIN residual-based horizon weights; '
            'GT-increment matching; or physical-tolerance Huber. Original clean-x codec, sigma distribution, '
            'four auxiliary losses, data ordering and frozen parent auxiliary bank are retained. '
            'Calibration matches only output-layer gradient magnitude, not full-model gradients. '
            'The shared control was separately qualified for exact loss/gradient and real eight-card execution.', '',
            'The JSON paired fields contain physical-log bootstrap, native submetrics, G+/G− and failure transfers. '
            'A2000-update window is an initial controlled test, not proof that an entire loss family is ineffective. '
            'Qualification, failed attempts, model loading, checkpoint saving and evaluation costs remain in private owned-worker receipts.']
    else:lines += ['', 'The previous-same-RL comparisons use full paired scenes/logs under the same protocol. '
              'A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval '
              'separately. This is not multiple training-seed evidence. Different RL update budgets do not '
              'establish method superiority.', '',
              'All metric components, zero-score and safety transfers, and paired95% intervals are in the '
              'per-model JSON. Raw worker costs and scene records are retained at '+str(root)+'. '
              'No teacher/search/scorer chooses a deployed output; every observation is one candidate.', '']
    (report/'README.md').write_text('\n'.join(lines))
    if pinned.get('trajectory_loss_analysis'):
        from tools.qwen_world_postcot.finish_il_midlate_analysis import finish
        finish(root, report)
    if pinned.get('coordinate_loss_study'):
        from tools.qwen_world_postcot.finish_coordinate_study import finish
        finish(root, report)
    return results


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', required=True); p.add_argument('--report-dir', required=True)
    args = p.parse_args(); root = Path(args.root); lock = (root/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'owner.json').exists(): raise FileExistsError('Do not duplicate this request')
    pinned = read(root/'PINNED.json'); source = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    write(root/'owner.json', dict(pid=os.getpid(), source=str(source), source_sha=head, at=time.time(),
          start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])))
    requests = {}
    for label, pin in pinned['models'].items():
        directory = root/label; directory.mkdir(exist_ok=False)
        pin = dict(pin, evaluation_hosts=['training-vla-zt','training-vlawm-tsinghua-zt',
                   'training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt5'])
        requests[label] = SnapshotRequest(directory, pin)
    failures = {}
    concurrency = int(pinned.get('max_concurrent_models', 4))
    if concurrency < 1 or concurrency > 8:
        raise ValueError('Unregistered model concurrency')
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(request.run): label for label, request in requests.items()}
        for future in as_completed(futures):
            label = futures[future]
            try: future.result()
            except BaseException: failures[label] = traceback.format_exc()
            write(root/'status.json', dict(at=time.time(), failed=failures,
                completed=[k for k,r in requests.items() if (r.root/'COMPLETE.json').exists()]))
    if failures:
        write(root/'failure.json', dict(failures=failures, automatic_retry=False)); return
    report = Path(args.report_dir); finalize(root, requests, report)
    def git(*argv): return subprocess.check_output(['git',*argv],cwd=source,text=True)
    if git('rev-parse','HEAD').strip() != head: raise ValueError('Changed source before publication')
    prefix = str(report.relative_to(source))
    for row in git('status','--porcelain=v1','-z','--untracked-files=all').split('\0'):
        if row and not row[3:].startswith(prefix+'/'): raise ValueError('Unrelated publication change')
    subprocess.run(['git','add','--',prefix],cwd=source,check=True)
    subprocess.run(['git','commit','-m','Record complete pinned World Test observations'],cwd=source,check=True)
    branch = git('branch','--show-current').strip(); commit = git('rev-parse','HEAD').strip()
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=source,check=True)
    if git('ls-remote','origin','refs/heads/'+branch).split()[0] != commit: raise ValueError('Remote SHA mismatch')
    write(root/'PUBLICATION.json', dict(status='PUBLISHED',commit=commit,remote_SHA=commit,at=time.time()))
    write(root/'COMPLETE.json', dict(status='COMPLETE',models=list(requests),at=time.time(),all_owned_workers_exited=True))


if __name__ == '__main__': main()
