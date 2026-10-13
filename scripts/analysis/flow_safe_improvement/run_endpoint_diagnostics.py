"""Finish the registered online endpoints with one TRAIN seed and auxiliary v2.

This finite wrapper composes existing inference, scoring and analysis commands.
It waits for the three-round controller to finish; it never trains or selects a
checkpoint, and does not generate additional inference seeds.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.analysis.flow_safe_improvement.run_learning_window import finish
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized, launch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--variant', required=True, choices=['ODE_REWARD_FM', 'ODE_SAFE_FM', 'EXPLORE_SAFE_FM', 'ODE_NFT'])
    p.add_argument('--wait-seconds', type=float, default=14400.)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(); root = Path(a.run_root); host = authorized(require_dispatch())
    name = a.variant+'_round3_TRAIN_D1'
    base = root/'evaluations/original_export_single_seed'; out = base/(name+'_operator')
    out.mkdir(parents=True, exist_ok=a.resume)
    lease = (out/'operator.lock').open('a'); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = root/'online'/(a.variant+'_seed42'); endpoint = run/'round3'
    population = root/'population/manifest.json'; cohorts = root/'cohorts/manifest.json'
    override = root/'resources/single_seed_override/REGISTRATION.json'
    modules = ['run_endpoint_diagnostics', 'sample_original_export', 'score_samples', 'evaluate_original_policy',
               'evaluate_policy', 'diagnose_policy_projection', 'diagnose_context_alignment',
               'stratify_training_evaluation', 'score_auxiliary_v2']
    identity = dict(host=host, variant=a.variant, population=file_sha(population), cohorts=file_sha(cohorts),
                    override=file_sha(override), sources={m: file_sha(Path(__file__).with_name(m+'.py')) for m in modules})
    identity['sha256'] = fingerprint(identity)
    if (out/'identity.json').exists() and read(out/'identity.json') != identity:
        raise ValueError('Endpoint diagnostic recovery identity changed')
    atomic_json(out/'identity.json', identity)
    if (out/'COMPLETE.json').exists():
        return
    record_path = base/(name+'_commands.json')
    records = read(record_path) if record_path.exists() else []
    for record in records:
        proc_stat = Path('/proc')/str(record['pid'])/'stat'
        if record.get('finished_unix') is None and proc_stat.exists():
            fields = proc_stat.read_text().rsplit(') ', 1)[1].split()
            if fields[19] == record.get('process_start_ticks') and fields[0] != 'Z':
                raise RuntimeError('An owned child is still active; do not duplicate it during operator recovery')
    state = dict(status='WAITING_FOR_REGISTERED_THIRD_ROUND', host=host, pid=os.getpid(), variant=a.variant)

    def save(**kwargs):
        state.update(kwargs); atomic_json(out/'state.json', state); atomic_json(record_path, records)

    def start(module_name, args, label, distributed=False):
        process, record = launch(module_name, args, out/(label+'.log'), records, host, distributed=distributed)
        record['process_start_ticks'] = (Path('/proc')/str(process.pid)/'stat').read_text().rsplit(') ', 1)[1].split()[19]
        save(status=label)
        return process, record

    def complete(process, record):
        try:
            finish(process, record)
        finally:
            save()

    deadline = time.monotonic()+a.wait_seconds
    save()
    while True:
        authorized(host)
        path = run/'single_seed_controller_state.json'
        current = read(path) if path.exists() else {}
        if current.get('status') == 'COMPLETE_THREE_FRESH_POLICY_ITERATIONS':
            break
        if 'FAILED' in current.get('status', '') or current.get('status') == 'PAUSE_VARIANT':
            raise RuntimeError('Preserve failed/paused variant; no automatic alternative endpoint')
        if time.monotonic() > deadline:
            raise TimeoutError('Third-round endpoint is not complete')
        time.sleep(10)
    export = endpoint/'learning/train/export_update000512'
    dev = endpoint/'learning/DEV_seed2101_original_epoch2'
    if not (export/'COMPLETE').exists() or not (dev/'COMPLETE.json').exists():
        raise ValueError('Registered final checkpoint and DEV evaluation must be sealed')
    weight_sha = file_sha(export/'pytorch_model.pt')
    bound = dict(weight_sha256=weight_sha, targets=read(endpoint/'targets/manifest.json')['sha256'],
                 DEV_report_sha256=read(dev/'summary/report.json')['sha256'])
    if (out/'endpoint.json').exists() and read(out/'endpoint.json') != bound:
        raise ValueError('Previously bound endpoint changed')
    atomic_json(out/'endpoint.json', bound)

    samples, scores, summary = [base/(name+'_'+part) for part in ('samples', 'v1', 'summary')]
    common = ['--population', population, '--evaluation-override', override]
    baseline_scores = root/'evaluations/original_export/SFT_TRAIN_D5_v1'
    baseline_samples = root/'evaluations/original_export/SFT_TRAIN_D5_samples'
    reference = root/'evaluations/SFT_TRAIN_Dreference4_v1'
    try:
        if not (summary/'report.json').exists():
            # Actual raw-observation peak was 7.40 GiB; this 16-GiB free-memory
            # guard reserves additional headroom and never evicts another owner.
            admissions = []
            for _ in range(2):
                authorized(host)
                lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                                 '--format=csv,noheader,nounits'], text=True).splitlines()
                if len(lines) != 8 or min(float(x.split(',')[-1]) for x in lines) < 16384:
                    raise RuntimeError('Insufficient measured headroom; preserve outputs for explicit recovery')
                admissions.append(dict(rows=lines, required_free_MiB=16384, unix=time.time()))
                atomic_json(out/'GPU_admission.json', admissions)
                time.sleep(2)
            args = ['--export', export, '--export-sha256', weight_sha, *common, '--cohort', 'diagnostic_tokens',
                    '--cohorts', cohorts, '--output', samples]
            if (samples/'manifest.json').exists():
                args += ['--resume']
            sampler, sr = start('sample_original_export', args, 'sample', True)
            while not (samples/'manifest.json').exists():
                if sampler.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Sampler did not publish its identity')
                time.sleep(2)
            args = ['--samples', samples, '--population', population, '--cache-root', root/'metric_cache_v1',
                    '--workers', 4, '--output', scores]
            if (scores/'manifest.json').exists():
                args += ['--resume']
            scorer, cr = start('score_samples', args, 'score')
            complete(sampler, sr); complete(scorer, cr)
            proc, rec = start('evaluate_original_policy', ['--scores', scores, '--baseline-scores', baseline_scores,
                '--reference-scores', reference, *common, '--output', summary], 'summary')
            complete(proc, rec)
        geometry = ['--samples', samples, '--baseline-samples', baseline_samples, *common]
        analysis_root = root/'diagnostics'
        projection = analysis_root/(a.variant+'_round3_single_seed_TRAIN_projection')
        if not (projection/'report.json').exists():
            args = [*geometry, '--cohorts', cohorts, '--output', projection]
            if a.variant != 'ODE_NFT':
                args += ['--targets', endpoint/'targets']
            proc, rec = start('diagnose_policy_projection', args, 'projection'); complete(proc, rec)
        context = analysis_root/(a.variant+'_round3_single_seed_TRAIN_context')
        if a.variant != 'ODE_NFT' and not (context/'report.json').exists():
            proc, rec = start('diagnose_context_alignment', [*geometry, '--targets', endpoint/'targets',
                             '--output', context], 'context'); complete(proc, rec)
        stratified = analysis_root/(a.variant+'_round3_single_seed_TRAIN_stratified')
        if not (stratified/'report.json').exists():
            proc, rec = start('stratify_training_evaluation', ['--scores', scores, '--baseline-scores', baseline_scores,
                '--reference-scores', reference, *common, '--cohorts', cohorts, '--targets', endpoint/'targets',
                '--output', stratified], 'stratified'); complete(proc, rec)
        auxiliary = root/'evaluations/single_seed_auxiliary'/(a.variant+'_round3_DEV1_v2')
        if not (auxiliary/'report.json').exists():
            proc, rec = start('score_auxiliary_v2', ['--samples', dev/'samples', *common,
                '--cache-root', '/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/metric_cache_navtrain_v2',
                '--workers', 4, '--output', auxiliary], 'auxiliary_v2'); complete(proc, rec)
        save(status='COMPLETE_SINGLE_SEED_ENDPOINT_DIAGNOSTICS')
        atomic_json(out/'COMPLETE.json', dict(identity_sha256=identity['sha256'], endpoint=bound,
                    training_summary=str(summary/'report.json'), auxiliary=str(auxiliary/'report.json'),
                    no_extra_training_or_inference_seed=True))
    except BaseException as error:
        save(status='FAILED_PRESERVE_OUTPUTS_NO_AUTOMATIC_RETRY', error=repr(error))
        raise


if __name__ == '__main__':
    main()
