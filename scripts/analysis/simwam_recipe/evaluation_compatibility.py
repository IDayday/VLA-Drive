"""Explicit, evidence-bound comparison across an audited accounting-only edit.

The original source-hash equality remains the default. A receipt never changes
an evaluation manifest, prediction seal, scorer, actor or population.
"""
import json
from pathlib import Path

from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint


MATCHED_FIELDS = (
    'schema', 'population_identity', 'tokens_sha256', 'scenes', 'cohort',
    'seed', 'modes', 'profile', 'feature_cache', 'actor', 'scoring',
)
REPO = Path(__file__).resolve().parents[3]


def match_protocol(identity, reference, receipt=None):
    for key in MATCHED_FIELDS:
        if key not in identity or key not in reference or identity[key] != reference[key]:
            raise ValueError('Unmatched reference protocol: ' + key)
    if identity['source_sha256'] == reference['source_sha256']:
        return dict(mode='identical_evaluator_source', source_sha256=identity['source_sha256'])
    if receipt is None:
        raise ValueError('Unmatched reference protocol: source_sha256; explicit qualified receipt required')
    path = Path(receipt)
    record = json.loads(path.read_text())
    body = dict(record)
    seal = body.pop('sha256')
    if fingerprint(body) != seal:
        raise ValueError('Compatibility receipt changed')
    if record['schema'] != 'simwam_evaluation_accounting_compatibility_v1' or record['status'] != 'QUALIFIED_LOGGING_ONLY':
        raise ValueError('Unqualified compatibility receipt')
    if set(record['fixed_protocol']) != {'seed','modes','profile','feature_cache','actor','scoring'}:
        raise ValueError('Incomplete fixed protocol in receipt')
    if {r['role'] for r in record['evidence_files']} != {'qualification','source_diff'} or not record['unchanged_numerical_files']:
        raise ValueError('Incomplete compatibility evidence')
    if (record['reference_source_sha256'] != reference['source_sha256']
            or record['evaluation_source_sha256'] != identity['source_sha256']):
        raise ValueError('Receipt does not cover this exact directed source pair')
    for key, value in record['fixed_protocol'].items():
        if identity.get(key) != value or reference.get(key) != value:
            raise ValueError('Receipt protocol changed: ' + key)
    for item in record['evidence_files']:
        evidence = REPO / item['path']
        if file_sha(evidence) != item['sha256']:
            raise ValueError('Compatibility evidence changed: ' + item['path'])
        if item['role'] == 'qualification':
            proof = json.loads(evidence.read_text())
            if (proof.get('status') != 'PASS'
                    or proof.get('reference_source_sha256') != reference['source_sha256']
                    or proof.get('evaluation_source_sha256') != identity['source_sha256']
                    or not proof.get('rechecked_saved_actions_metrics_and_simulation_exact')
                    or not proof.get('scientific_calls_and_shifted_ode_AST_identical')):
                raise ValueError('Qualification does not prove this source pair')
    for name, sha in record['unchanged_numerical_files'].items():
        if file_sha(REPO / name) != sha:
            raise ValueError('Numerical dependency changed: ' + name)
    if file_sha(REPO / 'scripts/analysis/simwam_recipe/evaluate_policy.py') != record['evaluation_source_sha256']:
        raise ValueError('Evaluator no longer matches the qualified source')
    return dict(mode='explicit_logging_only_source_compatibility',
                receipt=str(path), receipt_sha256=file_sha(path), receipt_identity=seal,
                original_reference_source=reference['source_sha256'],
                original_evaluation_source=identity['source_sha256'],
                prediction_score_and_population_mutations=False)
