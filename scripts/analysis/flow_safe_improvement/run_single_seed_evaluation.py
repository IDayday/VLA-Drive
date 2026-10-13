"""One matched original-export DEV seed, reusing only identified sealed outputs."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from scripts.analysis.flow_safe_improvement.run_learning_window import module, finish
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import launch, authorized
from scripts.analysis.flow_safe_improvement.sample_original_export import PROFILE
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('export', 'population', 'evaluation-override', 'baseline-scores', 'reference-scores', 'dev-cache', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    host = require_dispatch(); authorized(host)
    pop = json.loads(Path(a.population).read_text())
    seeds, override = evaluation_seeds(pop, a.evaluation_override)
    if seeds != [2101]:
        raise ValueError('This controller only runs the user-requested single seed')
    registration = json.loads(Path(a.evaluation_override).read_text())
    export = Path(a.export); weight_sha = file_sha(export/'pytorch_model.pt')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    identity = dict(export_weight_sha256=weight_sha, config_sha256=file_sha(export/'config.yaml'),
                    population_identity=pop['identity_sha256'], evaluation_override=override,
                    baseline=str(a.baseline_scores), reference=str(a.reference_scores),
                    source_files={name: file_sha(Path(__file__).with_name(name+'.py')) for name in
                                  ('run_single_seed_evaluation', 'sample_original_export', 'score_samples', 'evaluate_policy', 'evaluate_original_policy')})
    identity['sha256'] = fingerprint(identity)
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text()) != identity:
        raise ValueError('Single-seed evaluation identity changed')
    atomic_json(out/'identity.json', identity)
    if (out/'summary/report.json').exists():
        report = json.loads((out/'summary/report.json').read_text())
        if report['inference_seeds'] != seeds or report['status'] != 'COMPLETE_MATCHED_EVALUATION':
            raise ValueError('Existing summary is not the locked single-seed evaluation')
        return
    records = json.loads((out/'phase_records.json').read_text()) if (out/'phase_records.json').exists() else []
    matching = [x for x in registration['existing_original_outputs'] if x['behavior_sha256'] == weight_sha and x['cohort'] == 'dev_tokens']
    reuse = max(matching, key=lambda x: x['sealed_scenes'], default=None)
    scores = out/'v1'; samples = out/'samples'
    if reuse and reuse.get('complete_scores'):
        scores = Path(reuse['complete_scores'])
        manifest = json.loads((scores/'manifest.json').read_text())
        if manifest['sample_manifest']['sha256'] != reuse['sample_manifest_sha256']:
            raise ValueError('Read-only reuse identity changed')
        atomic_json(out/'reused_complete_scores.json', dict(source=str(scores), selected_seeds=seeds,
                                                           no_new_actor_or_scoring_calls=True, source_identity=manifest['sha256']))
    else:
        args = ['--export', export, '--export-sha256', weight_sha, '--population', a.population,
                '--evaluation-override', a.evaluation_override, '--output', samples]
        if reuse:
            args += ['--reuse-samples', reuse['samples']]
        if (samples/'manifest.json').exists():
            args += ['--resume']
        sampler, sample_record = launch('sample_original_export', args, out/'sample.log', records, host, distributed=True)
        atomic_json(out/'phase_records.json', records)
        deadline = time.monotonic()+7200
        while not (samples/'manifest.json').exists():
            if sampler.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Original sampler failed before manifest publication')
            time.sleep(2)
        args = ['--samples', samples, '--population', a.population, '--cache-root', a.dev_cache,
                '--workers', 8, '--output', scores]
        if (scores/'manifest.json').exists():
            args += ['--resume']
        scorer, score_record = launch('score_samples', args, out/'score.log', records, host)
        atomic_json(out/'phase_records.json', records)
        finish(sampler, sample_record); finish(scorer, score_record)
    args = ['--scores', scores, '--baseline-scores', a.baseline_scores, '--reference-scores', a.reference_scores,
            '--population', a.population, '--evaluation-override', a.evaluation_override, '--output', out/'summary']
    proc, record = launch('evaluate_original_policy', args, out/'summary.log', records, host)
    finish(proc, record)
    atomic_json(out/'phase_records.json', records)
    atomic_json(out/'COMPLETE.json', dict(identity_sha256=identity['sha256'], summary=str(out/'summary/report.json'),
                                         score_source=str(scores), no_new_training_seed=True))


if __name__ == '__main__':
    main()
