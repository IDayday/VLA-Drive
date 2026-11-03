"""Explicit, byte-identical deployment references; never extra measured scores.

Only a registered checkpoint pair can avoid duplicate NAVTEST inference. Each
pair keeps its own training identity and is excluded from independent results.
No checkpoint or training state is copied between runs.
"""
import json
from pathlib import Path
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint


def read(path):
    return json.loads(Path(path).read_text())


def deployment_equivalence(left, right):
    from starVLA.rl.d44_native_rl.deployment import inspect
    li, ls, lf = inspect(left)
    ri, rs, rf = inspect(right)
    for identity in (li, ri):
        if identity.get('qualification') is not False or identity.get('world') != 8:
            raise ValueError('Only sealed formal world8 checkpoints')
    # These are the complete architecture options used by native load_policy;
    # source_files also binds the loader, original parent, pruning and codec.
    for key in ('parent_model_sha256', 'scope', 'source_files', 'score_protocol'):
        if li[key] != ri[key]:
            raise ValueError('Different deployment ' + key)
    for key in ('parent_checkpoint', 'parent_registration', 'text_rank',
                'film_last_layers', 'post_feedback'):
        if li['recipe'][key] != ri['recipe'][key]:
            raise ValueError('Different native loader option ' + key)
    if lf['optimizer.pt'] != rf['optimizer.pt']:
        raise ValueError('Different trainable payload')
    if ls['updates'] != rs['updates']:
        raise ValueError('Only same-boundary references are registered here')
    return dict(parent=li['parent_model_sha256'], scope=li['scope'],
                trainable_payload_sha256=lf['optimizer.pt'], updates=ls['updates'],
                source_files_sha256=fingerprint(li['source_files']))


def prepare_references(campaign, root):
    """Keep unmeasured aliases explicit; reject any already-dispatched alias."""
    profile = Path(root) / 'operations/EVALUATION_EQUIVALENCE.json'
    if not profile.exists():
        return {}
    plan = read(profile)
    if plan.get('schema') != 'd46_explicit_equivalent_evaluation_v1':
        raise ValueError('Unrecognized evaluation equivalence registration')
    cases = campaign['evaluations']
    for pair in plan['pairs']:
        alias, reference = cases.get(pair['alias']), cases.get(pair['reference'])
        if alias is None or reference is None:
            continue
        if any(alias.get('jobs', {}).values()):
            raise ValueError('Cannot replace an actually dispatched evaluation')
        if (alias['request']['checkpoint'] != pair['alias_checkpoint'] or
                reference['request']['checkpoint'] != pair['reference_checkpoint']):
            raise ValueError('Checkpoint reference changed')
        proof = deployment_equivalence(pair['alias_checkpoint'], pair['reference_checkpoint'])
        if fingerprint(proof) != pair['deployment_proof_sha256']:
            raise ValueError('Registered model equivalence changed')
        alias.update(reference_only=True, reference_case=pair['reference'],
                     equivalence_proof=proof, independently_evaluated=False,
                     reference_profile_sha256=file_sha(profile),
                     status='EQUIVALENT_REFERENCE_PENDING')
        if reference['status'] == 'COMPLETE':
            summary = read(reference['summary'])
            if summary['qualification'] or summary['scenes'] != 12146 or summary['invalid_scenes']:
                raise ValueError('Reference is not complete official B1 NAVTEST')
            alias.update(status='EQUIVALENT_REFERENCE_COMPLETE',
                         reference_summary=reference['summary'])
    return plan.get('priorities', {})
