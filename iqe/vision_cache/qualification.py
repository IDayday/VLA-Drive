"""Certificates authorize a measured data execution path, not scientific gain."""
from pathlib import Path
from copy import deepcopy
from ..contracts import require
from ..io import file_hash, read_json, digest


def validate_pilot(path,identity,parent_hash):
    value=read_json(path)
    require(value['schema']=='iqe_fp32_pilot_qualification_v1' and value['status']=='PASS','real pilot qualification missing')
    require(value['visual_identity_hash']==digest(identity) and value['parent_sha256']==parent_hash,'foreign pilot/parent identity')
    for item in value['evidence'].values():require(file_hash(item['path'])==item['sha256'],'pilot evidence changed')
    payload=read_json(value['evidence']['payload']['path']);updates=read_json(value['evidence']['updates']['path'])
    require(payload['status']=='PASS' and payload['all_payloads_bitwise_equal'] and payload['storage_bitwise'] and payload['deepstack_count']==3,'payload equivalence failed')
    require(updates['status']=='PASS' and updates['full_state_rng_cursor_equal'] and
        len(updates['full_gradient_audit'])==8 and all(r['full_gradient_bitwise'] for r in updates['full_gradient_audit']), 'full gradient/update/RNG pilot failed')
    return value


def validate_cache_qualification(path, *, identity=None, fit_hash=None, runtime_identity=None, for_build=False):
    value = read_json(path)
    require(value['schema'] == 'iqe_fp32_cache_qualification_v1' and value['status'] == 'PASS', 'cache runtime unqualified')
    require(value['visual_numeric_contract'] == 'legacy_fp32', 'foreign cache precision')
    if identity is not None: require(value['visual_identity_hash'] == digest(identity), 'cache numerical identity changed')
    if fit_hash is not None: require(value['fit_hash'] == fit_hash, 'qualified population changed')
    evidence = {}
    for key, item in value['evidence'].items():
        require(file_hash(item['path']) == item['sha256'], 'qualification evidence changed: '+key)
        evidence[key] = read_json(item['path'])
    require({'numerical','representative_io','resume','online_inference'} <= evidence.keys(), 'incomplete cache qualification evidence')
    for key in evidence: require(evidence[key]['status'] == 'PASS', 'failed qualification: '+key)
    require(evidence['numerical']['storage_bitwise'] and evidence['numerical']['no_images_or_visual_on_hit'] and
            evidence['numerical']['deepstack_count'] == 3 and evidence['numerical']['full_state_rng_cursor_equal'],
            'FP32 cache equivalence not established')
    require(evidence['representative_io']['shared_storage'] and evidence['representative_io']['steady_updates'] >= 100,
            'representative shared-storage timing missing')
    require(evidence['representative_io']['actual_global_batch'] == 32 and
            evidence['representative_io']['sample_sequence_sha256'] == evidence['numerical']['sample_sequence_sha256'],
            'qualification batch/sample sequence mismatch')
    if not for_build:
        require(runtime_identity is not None, 'formal runtime identity required')
        # Full coverage index differs from the pilot; compute/source/reader options
        # stay identical and full-cache integrity/coverage is checked at attachment.
        expected = deepcopy(value['runtime_identity']); actual = deepcopy(runtime_identity)
        for candidate in (expected, actual):
            candidate = candidate.get('production_cache_runtime', {})
            candidate.pop('cache_index_sha256', None); candidate.pop('cache_root', None)
        require(expected == actual, 'qualified runtime/code/environment changed')
    return value
