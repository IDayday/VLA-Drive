"""Configuration and prepared-data contracts, shared by all entry points."""
import hashlib
import json
import math
from pathlib import Path

ARMS = ['A', 'B', 'C', 'F', 'E', 'A_VLM', 'C_VLM', 'E_VLM', 'A_DiT', 'C_DiT', 'E_DiT']
INPUTS = ['A', 'B', 'C', 'F', 'E']
SCOPE_VERSION = 'history-frozen-v2'
INPUT_VERSION = 'action-only-suffix-v3'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, name, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if integer and type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    if value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be {"nonnegative" if zero else "positive"}')


def validate_training(cfg, formal=True):
    if cfg.get('input_protocol_version') != INPUT_VERSION:
        raise ValueError('Input protocol mismatch: ' + INPUT_VERSION)
    if cfg.get('parameter_scope_version') != SCOPE_VERSION:
        raise ValueError('Parameter scope mismatch: ' + SCOPE_VERSION)
    if not isinstance(cfg.get('arms'), list) or len(cfg['arms']) != 11 or set(cfg['arms']) != set(ARMS):
        raise ValueError('Expected all 11 unique arms')
    for key, value in [('group_size', 8), ('accumulation_scenes', 4), ('steps', 10), ('evaluation_replicates', 3)]:
        if type(cfg.get(key)) is not int or cfg[key] != value:
            raise ValueError(f'Changed {key} contract: expected {value}')
    for key in ['eta', 'clip', 'vlm_lr', 'action_lr', 'gradient_clip']:
        positive(cfg.get(key), key)
    if cfg['clip'] >= 1:
        raise ValueError('clip must be below 1')
    positive(cfg.get('beta'), 'beta', zero=True)
    positive(cfg.get('checkpoint_every_updates'), 'checkpoint_every_updates', integer=True, zero=True)
    if cfg.get('score_workers_per_replica', 1) != 1:
        raise ValueError('This runner scores serially; score_workers_per_replica must be 1')
    tolerance = cfg.get('f_max_token_delta', 0)
    if type(tolerance) is not int or tolerance not in [0, 2]:
        raise ValueError('f_max_token_delta must be 0 (strict) or explicitly confirmed 2')
    for name in ['epochs', 'wall_seconds', 'evaluation_wall_seconds', 'total_wall_seconds']:
        if cfg.get(name) is not None:
            positive(cfg[name], name, integer=name == 'epochs')
    seeds = cfg.get('training_seeds')
    if seeds is not None:
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError('training_seeds must be distinct nonnegative integers')
    if formal and (cfg.get('formal_protocol_locked') is not True or cfg.get('epochs') is None or seeds is None or cfg.get('wall_seconds') is None):
        raise ValueError('Formal epochs, seeds, wall budget and protocol lock required')
    return cfg


def validate_prepared(root, cfg=None, complete=True):
    root = Path(root)
    receipt = read(root/'PREPARATION.json')
    for name in ['PROMPTS.json', 'DONORS.json', 'SPLITS.json']:
        if name not in receipt.get('files', {}) or sha(root/name) != receipt['files'][name]:
            raise ValueError('Prepared file changed or missing from receipt: ' + name)
    splits = read(root/'SPLITS.json')
    if len(splits['train']) != 1264 or len(splits['test']) != 141 or len(splits['monitor']) != 64:
        raise ValueError('Expected train/test/monitor counts 1264/141/64')
    train, test = set(splits['train']), set(splits['test'])
    if len(train) != 1264 or len(test) != 141 or train & test or len(set(splits['monitor'])) != 64 or not set(splits['monitor']) <= train:
        raise ValueError('Duplicate, overlapping, or invalid split tokens')
    prompts = read(root/'PROMPTS.json')['scenes']
    if len(prompts) != 1405 or {r['token'] for r in prompts} != train | test:
        raise ValueError('Prompt coverage does not match splits')
    donors = read(root/'DONORS.json')
    assignments = donors['assignments']
    by_token = {r['token']: r for r in assignments}
    if len(by_token) != len(assignments):
        raise ValueError('Duplicate F assignments')
    policy = receipt.get('f_policy', {'max_token_delta': 0, 'exception_tokens': []})
    allowed = set(policy['exception_tokens'])
    bound = policy['max_token_delta']
    if type(bound) is not int or bound not in [0, 2] or (bound == 0 and allowed) or not allowed <= train:
        raise ValueError('Invalid F exception policy')
    if bound==2 and (policy.get('confirmed') is not True or len(allowed)!=22):
        raise ValueError('Expected the 22 explicitly confirmed training exceptions')
    if cfg is not None and cfg.get('f_max_token_delta', 0) != bound:
        raise ValueError('Training and prepared F policies differ')
    prompt_map = {r['token']: r for r in prompts}
    for row in prompts:
        arms = row['arms']
        if not {'A', 'B', 'C', 'E'} <= set(arms):
            raise ValueError('Missing non-F input')
        for arm, payload in arms.items():
            if arm not in INPUTS or hashlib.sha256(payload['text'].encode()).hexdigest() != payload['sha256']:
                raise ValueError('Invalid prompt hash or arm')
        token = row['token']
        if ('F' in arms) != (token in by_token):
            raise ValueError('F text/assignment coverage differs')
        if token not in by_token:
            continue
        donor = by_token[token]
        dt = donor['donor_token']
        if dt not in train or row['log'].rsplit('_', 2)[0] == prompt_map[dt]['log'].rsplit('_', 2)[0]:
            raise ValueError('Invalid F donor source drive or split')
        delta = arms['F']['token_count'] - arms['C']['token_count']
        if delta != donor.get('token_delta', 0) or abs(delta) > (bound if token in allowed else 0):
            raise ValueError('F token delta outside recorded policy')
    missing = (train | test) - set(by_token)
    if len(assignments) != receipt['matched_F'] or len(missing) != receipt['missing_F']:
        raise ValueError('F receipt counts differ from actual coverage')
    if complete and (receipt['status'] != 'PASS' or missing):
        raise ValueError('Full F coverage not accepted: ' + receipt['status'])
    return receipt


def candidate_budget(cfg):
    validate_training(cfg, formal=False)
    seeds, epochs = cfg.get('training_seeds'), cfg.get('epochs')
    result = {'initial_jobs': 5, 'initial_candidates': 16920,
              'training_replicas': None, 'updates_per_replica': None,
              'training_candidates': None, 'monitor_candidates': None,
              'final_candidates': None, 'total_candidates': None}
    if seeds is not None:
        replicas = len(seeds)*11
        result.update(training_replicas=replicas, final_candidates=replicas*141*3*8)
        if epochs is not None:
            result.update(updates_per_replica=epochs*316,
                          training_candidates=replicas*epochs*1264*8,
                          monitor_candidates=replicas*(epochs+1)*64*8)
            result['total_candidates'] = sum(result[k] for k in ['initial_candidates', 'training_candidates', 'monitor_candidates', 'final_candidates'])
    return result


def device_ids(text):
    items = text.split(',')
    if any(not x.isdigit() for x in items) or len(items) != len(set(map(int, items))):
        raise ValueError('Devices must be distinct nonnegative indices without spaces')
    return [str(int(x)) for x in items]
