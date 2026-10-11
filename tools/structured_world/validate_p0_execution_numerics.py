"""Optional prospective P0 EXACT forward observation using unchanged be5 operators.

register freezes an external producer plus a separate immutable model root.
probe invokes the original legacy validator, keeps its assertion/exit and strict
JSON unchanged, and writes P0_FORWARD_COMPARISON.json even on completed strict
failure. No raw images, full tensors or parameter gradient banks are persisted.
This separate verification does not amend the earlier native acceptance binding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RELATIVE = 'tools/structured_world/validate_p0_execution_numerics.py'
MODEL_SHA = 'be5e72248a46afd43e1bf279c21134c391c5edad'
COLLECTOR_SHA = 'fc01086f1da6a8420db73de3b27451a330961d25894df5c0949d473652bc059b'
REGISTRATION_SHA = '8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007'
LEGACY_SHA = '31dc8caf31df0daa1e96c074569d1d9cff39ca80ca522f94ae9331534e8d2ffd'
PRESET = {'name': 'p0_exact_v1', 'mode': 'engineering_fast', 'atol': 0., 'rtol': 0.,
          'relative_L2_max': 0., 'physical_q_final_max_position_error_m': 0.,
          'all_forward_modules_finite_bitwise': True}
ROLES = {'A': 'io_preserving_v1', 'AA': 'io_preserving_v1', 'B': 'engineering_fast'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def write_new(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def inventory(root):
    root = Path(root)
    paths = [p for name in ('starVLA/model', 'starVLA/dataloader', 'tools/ddpolicy_vehicle', 'tools/foresight', 'third_party')
             for p in sorted((root/name).rglob('*.py'))]
    paths += [root/'tools/structured_world'/n for n in ('train_vla.py', 'recovery_boundary.py', 'training_assets.py',
              'stage_timing.py', 'prefetch.py', 'resume_origin.py', 'execution_log.py', 'native_capture.py',
              'validate_engineering_execution.py', 'validate_execution_numerics.py', 'compare_execution_checkpoints.py',
              'validate_execution_equivalence.py')]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def frozen_root(root, expected=None):
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    require(expected is None or sha == expected, 'Immutable root commit mismatch')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip(), 'Clean committed immutable root required')
    return sha


def shared_collector(model_root):
    path = Path(model_root)/'tools/structured_world/validate_execution_numerics.py'
    require(digest(path) == COLLECTOR_SHA, 'Original timevec Collector source changed')
    spec = importlib.util.spec_from_file_location('_p0_original_collector', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def policy_and_preset(shared, protected_original_root):
    policy, _ = shared.registration(protected_original_root=protected_original_root)
    p0 = policy['tiers']['E2']['forward_by_mode']['engineering_fast']
    require(p0['required_exact'] == ['q0', 'Bt', 'q_final', 'loss'] and p0['module_tolerances'] is None,
            'Unknown original P0 EXACT registration; no threshold override')
    return policy


def register(model_root, acceptance_bindings, bindings_sha, protected_original_root, output):
    model_root = Path(model_root).resolve()
    shared = shared_collector(model_root)
    policy = policy_and_preset(shared, protected_original_root)
    producer_sha, model_sha = frozen_root(ROOT), frozen_root(model_root, MODEL_SHA)
    require(hashlib.sha256(subprocess.check_output(['git', 'show', producer_sha+':'+RELATIVE], cwd=ROOT)).hexdigest() == digest(__file__),
            'External producer must be included in the frozen commit')
    require(digest(acceptance_bindings) == bindings_sha, 'Historical acceptance binding changed')
    b = load(acceptance_bindings)
    require(b['schema'] == 'execution_acceptance_bindings_v1' and b['registration_sha256'] == REGISTRATION_SHA
            and b['candidate_mode'] == 'engineering_fast' and b['tier'] == 'E2', 'Original P0 binding required')
    require(set(b['cases']) == {'G1/initial', 'G1/mid', 'G3/initial', 'G3/mid'} and
            all(b['sources'][r]['git_sha'] == model_sha for r in ROLES), 'Four original origins and be5 roles required')
    files = inventory(model_root)
    changes = []
    for row in policy['protected_files']:
        current = digest(model_root/row['path'])
        if current != row['sha256']:
            require(row['path'] == 'tools/structured_world/train_vla.py', 'Unexpected protected model-root source change')
            changes.append({'path': row['path'], 'historical_sha256': row['sha256'], 'model_sha256': current,
                            'reason': shared.CAPTURE_REASON, 'algorithm_changed': False,
                            'reason_is_prior_operator_reviewed_declaration': True})
    value = {'schema': 'execution_p0_forward_producer_source_binding_v1', 'state': 'frozen_before_candidate',
             'frozen_UTC': datetime.now(timezone.utc).isoformat(), 'registration_sha256': REGISTRATION_SHA,
             'candidate_mode': 'engineering_fast', 'roles': ROLES, 'preset': PRESET,
             'model': {'root': str(model_root), 'git_sha': model_sha, 'files_sha256': files},
             'producer': {'root': str(ROOT), 'git_sha': producer_sha, 'files_sha256': {RELATIVE: digest(__file__)}},
             'original_acceptance_bindings': {'path': str(Path(acceptance_bindings).resolve()), 'sha256': bindings_sha},
             'cases': {name: {'origin': case['origin'], 'science': case['science']} for name, case in b['cases'].items()},
             'protected_original_root': str(Path(protected_original_root).resolve()),
             'protected_source_changes': changes, 'historical_protected_files': policy['protected_files'],
             'verification_relationship': 'Separate prospective optional forward verification; not a new forward receipt under the historical native binding',
             'native_binding_or_prior_reports_modified': False, 'formal_authorized': False}
    write_new(output, value)
    return {'path': str(Path(output).resolve()), 'sha256': digest(output), 'frozen_UTC': value['frozen_UTC']}


def check_binding(path, sha, model_root, shared):
    require(digest(path) == sha, 'External producer binding SHA mismatch')
    b = load(path)
    require(b['schema'] == 'execution_p0_forward_producer_source_binding_v1' and b['state'] == 'frozen_before_candidate'
            and b['registration_sha256'] == REGISTRATION_SHA and b['roles'] == ROLES and b['preset'] == PRESET
            and b['candidate_mode'] == 'engineering_fast', 'Wrong P0 source binding or EXACT preset')
    frozen = datetime.fromisoformat(b['frozen_UTC'].replace('Z', '+00:00'))
    require(frozen.tzinfo is not None and frozen <= datetime.now(timezone.utc), 'Binding must precede this optional probe')
    require(Path(model_root).resolve() == Path(b['model']['root']).resolve() and Path(b['producer']['root']).resolve() == ROOT,
            'Explicit model/producer roots differ from binding')
    frozen_root(model_root, MODEL_SHA); frozen_root(ROOT, b['producer']['git_sha'])
    require(b['model']['git_sha'] == MODEL_SHA and inventory(model_root) == b['model']['files_sha256'], 'Frozen be5 model inventory changed')
    require(b['producer']['files_sha256'] == {RELATIVE: digest(__file__)}, 'External wrapper changed')
    ref = b['original_acceptance_bindings']
    require(digest(ref['path']) == ref['sha256'], 'Original native binding changed')
    original = load(ref['path'])
    require(b['cases'] == {name: {'origin': case['origin'], 'science': case['science']} for name, case in original['cases'].items()},
            'Original scientific/origin assignment changed')
    policy = policy_and_preset(shared, b['protected_original_root'])
    expected_changes = []
    for old in policy['protected_files']:
        actual = digest(Path(model_root)/old['path'])
        if actual != old['sha256']:
            require(old['path'] == 'tools/structured_world/train_vla.py', 'Protected source changed')
            expected_changes.append({'path': old['path'], 'historical_sha256': old['sha256'], 'model_sha256': actual,
                'reason': shared.CAPTURE_REASON, 'algorithm_changed': False, 'reason_is_prior_operator_reviewed_declaration': True})
    require(b['protected_source_changes'] == expected_changes and b['historical_protected_files'] == policy['protected_files'],
            'Historical protected source declaration changed')
    return b


def collector_class(shared):
    class ExactCollector(shared.Collector):
        def after_backward(self, model, rng):
            super().after_backward(model, rng)
            for comparison in self.rows.values():
                for row in comparison['modules'].values():
                    row['advisory_passed'] = row['advisory_passed'] and row['bitwise_equal']
    return ExactCollector


def combine(shared, records, strict_ref, expected_ranks=8):
    result = shared.combine(records, PRESET, strict_ref, expected_ranks=expected_ranks)
    # Original Collector already observes all three pairs, including A_A and AA_B.
    for name in result['logical_forward_module_inventory']:
        result['modules'][name]['A_A'] = shared.aggregate_stat([r['comparisons']['A_A']['modules'][name] for r in records], PRESET)
    for r in records:
        for name, row in r['comparisons']['A_A']['modules'].items():
            result['modules'][f"rank{r['rank']}/{name}"]['A_A'] = row
    pairs = result['forward_exact_by_pair']
    module_exact = {name: all(row['bitwise_equal'] and row['nonfinite_elements'] == 0 for row in stats.values())
                    for name, stats in result['modules'].items()}
    result['forward_exact'].update({name: module_exact[name] for name in result['logical_forward_module_inventory']})
    result.update(schema='execution_p0_exact_forward_stats_v1', preset=PRESET,
                  advisory_forward_passed=all(module_exact.values()) and all(all(v.values()) for v in pairs.values()),
                  candidate_mode='engineering_fast', full_loss_capture='Actual returned loss, weighted task losses and raw auxiliary losses',
                  full_parameter_gradient_statistics_captured=False)
    result['limits'].append('Representative activation gradients remain diagnostic; full parameter AA_B gradient norms are not fabricated')
    return result


def selected_origin(binding, run, tag):
    require(Path(tag).name == tag, 'Explicit immutable safe tag required')
    identity, complete = load(Path(run)/'identity.json'), load(Path(run)/'checkpoints'/tag/'COMPLETE.json')
    require(identity['identity'] == complete['identity'], 'Origin identity differs')
    matches = [name for name, case in binding['cases'].items() if case['origin']['identity'] == complete['identity']
               and case['origin']['completed'] == complete['completed']
               and case['origin']['COMPLETE_sha256'] == digest(Path(run)/'checkpoints'/tag/'COMPLETE.json')]
    require(len(matches) == 1, 'Actual origin is not exactly one of four prospective assignments')
    science = binding['cases'][matches[0]]['science']
    actual_config = hashlib.sha256(json.dumps(identity['config'], sort_keys=True).encode()).hexdigest()
    require(actual_config == science['config_sha256'], 'Actual origin config changed')
    for native, bound in (('current_and_GT_identity', 'current_and_GT_identity'), ('DINO_identity', 'DINO_identity'),
                         ('training_seed', 'training_seed'), ('effective_global_batch', 'global_batch'),
                         ('FM_repeat', 'FM_repeat'), ('schedule_horizon', 'schedule_horizon'), ('schedule_warmup', 'warmup')):
        require(json.dumps(identity[native], sort_keys=True) == json.dumps(science[bound], sort_keys=True), 'Actual origin science changed: '+native)
    require(identity['world_size'] == 8 and identity['microbatch'] == 4 and identity['dataset'] == 'navsim',
            'Original full8 NAVSIM microbatch4 required')
    return matches[0]


def run_observed(args, shared, legacy, framework, binding, recheck):
    collector = collector_class(shared)(PRESET, before_candidate=recheck)
    original_class, original_rng, original_dist = legacy.VLAStructuredFGTR, legacy.capture_rng, legacy.dist
    original_queries, original_counts = framework.sample_queries, legacy.optimizer_batch_counts
    holder = {}
    class ObservedModel(original_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw); holder['model'] = self
        def forward(self, *a, **kw):
            collector.begin(); result = super().forward(*a, **kw); collector.forward(result); return result
    def observed_rng(generators):
        value = original_rng(generators); collector.after_backward(holder['model'], value); return value
    class ObservedDist:
        def __getattr__(self, name): return getattr(original_dist, name)
        def all_gather_object(self, output, row):
            original_dist.all_gather_object(output, row)
            records = [None]*len(output)
            original_dist.all_gather_object(records, collector.report(row['rank'], row['tokens']))
            if row['rank'] == 0:
                holder['report'] = combine(shared, records, {'path': str((args.output/'strict/COMPLETE.json').resolve()), 'pending': True})
    legacy.VLAStructuredFGTR, legacy.capture_rng, legacy.dist = ObservedModel, observed_rng, ObservedDist()
    legacy.optimizer_batch_counts = shared.observe_counts(original_counts, collector)
    framework.sample_queries = shared.observe_queries(original_queries, collector)
    old_argv = sys.argv
    forwarded = ['--run', str(args.run), '--tag', args.tag, '--cache', str(args.cache), '--dino-root', str(args.dino_root),
                 '--output', str(args.output/'strict'), '--microbatch', str(args.microbatch), '--optimized-mode', 'engineering_fast',
                 '--attention-diagnostic', 'native', '--precision-diagnostic', 'native']
    if args.model_state_file:
        forwarded += ['--model-state-file', str(args.model_state_file)]
    try:
        sys.argv = [str(Path(args.model_source_root)/'tools/structured_world/validate_engineering_execution.py')]+forwarded
        legacy.main()  # Preserve original assertion and exit; no exception suppression.
    finally:
        sys.argv = old_argv
        legacy.VLAStructuredFGTR, legacy.capture_rng, legacy.dist = original_class, original_rng, original_dist
        legacy.optimizer_batch_counts, framework.sample_queries = original_counts, original_queries
        if 'report' in holder:
            path = args.output/'strict/COMPLETE.json'
            require(path.is_file(), 'Original strict completion receipt missing')
            original = load(path); report = holder['report']
            report.update(original_strict={'path': str(path.resolve()), 'sha256': digest(path), 'passed': original['passed'],
                          'tool_sha256': LEGACY_SHA, 'unchanged_original_json': True}, producer_sha256=digest(__file__),
                          producer_source_binding={'path': str(args.source_binding.resolve()), 'sha256': args.source_binding_sha256},
                          model_source={'root': str(Path(args.model_source_root).resolve()), 'git_sha': binding['model']['git_sha']},
                          original_acceptance_bindings=binding['original_acceptance_bindings'],
                          verification_relationship=binding['verification_relationship'], checkpoint={'run': str(args.run.resolve()), 'tag': args.tag})
            write_new(args.output/'P0_FORWARD_COMPARISON.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('register')
    for name in ('model-source-root', 'acceptance-bindings', 'protected-original-root', 'output'):
        r.add_argument('--'+name, type=Path, required=True)
    r.add_argument('--acceptance-bindings-sha256', required=True)
    p = sub.add_parser('probe')
    for name in ('model-source-root', 'source-binding', 'run', 'cache', 'dino-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-binding-sha256', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--model-state-file', type=Path); p.add_argument('--microbatch', type=int, choices=(4,), default=4)
    p.add_argument('--preflight', action='store_true', help='CPU source/origin binding checks only; no model import or GPU job')
    args = parser.parse_args()
    if args.command == 'register':
        print(json.dumps(register(args.model_source_root, args.acceptance_bindings, args.acceptance_bindings_sha256,
                                  args.protected_original_root, args.output))); return
    shared = shared_collector(args.model_source_root)
    recheck = lambda: check_binding(args.source_binding, args.source_binding_sha256, args.model_source_root, shared)
    binding = recheck()
    require(not args.output.exists(), 'NEW output required; no old report overwrite')
    case = selected_origin(binding, args.run, args.tag)
    if args.preflight:
        print(json.dumps({'preflight_passed': True, 'case': case, 'preset': PRESET, 'model_source_sha': MODEL_SHA,
                          'source_binding_sha256': args.source_binding_sha256, 'GPU_jobs_started': 0})); return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    model_root = Path(args.model_source_root).resolve(); sys.path.insert(0, str(model_root))
    # Refuse previously imported shadow model/trainer modules before importing.
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name == 'starVLA' or name.startswith('starVLA.') or name == 'tools' or name.startswith('tools.')):
            require(Path(path).resolve().is_relative_to(model_root), 'Foreign model/trainer module already imported: '+name)
    legacy = importlib.import_module('tools.structured_world.validate_engineering_execution')
    framework = importlib.import_module('starVLA.model.framework.vla_structured_fgtr')
    require(Path(legacy.__file__).resolve() == model_root/'tools/structured_world/validate_engineering_execution.py'
            and digest(legacy.__file__) == LEGACY_SHA and Path(framework.__file__).resolve() == model_root/'starVLA/model/framework/vla_structured_fgtr.py',
            'Actual imported legacy/model source differs from frozen be5')
    run_observed(args, shared, legacy, framework, binding, recheck)


if __name__ == '__main__':
    main()
