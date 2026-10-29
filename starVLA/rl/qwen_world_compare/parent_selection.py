"""Own-architecture complete-window DEV selection; exact ties choose earlier."""
import json
import math
from pathlib import Path

from .contracts import fingerprint, file_sha
from .loading import inspect_IL_checkpoint, inspect_RL_checkpoint

IL_SELECTION_SCHEMA = 'qwen_world_compare_IL_DEV_selection_v1'
RL_SELECTION_SCHEMA = 'qwen_world_compare_RL_DEV_selection_v1'
IL_MILESTONES = tuple(range(5000, 100001, 5000))
RL_MILESTONES = (0, 5000, 10000, 20000, 35000, 50000)


def _publish(data, output):
    data = dict(data, sha256=fingerprint(data))
    if output is not None:
        from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
        path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if json.loads(path.read_text()) != data:
                raise FileExistsError('Frozen parent selection cannot be overwritten')
        else:
            atomic_json(path, data)
    return data


def _candidate(row, architecture_id, stage):
    from tools.qwen_world_compare.evaluate import summarize, read_evaluation_identity
    step = int(row['updates'])
    root = Path(row['evaluation'])
    summary = summarize(root)
    identity = read_evaluation_identity(root)
    if (identity['cohort'] != 'DEV' or identity['qualification'] or identity['seed'] != 2101
            or identity['architecture_id'] != architecture_id or identity['count'] != 1696
            or identity['inference'] != 'single_candidate_cleanx_K10_shift5'
            or summary['invalid_scenes'] != 0 or not math.isfinite(summary['PDMS'])):
        raise ValueError('Selection requires the same complete formal DEV protocol')
    if stage == 'IL':
        info = inspect_IL_checkpoint(row['checkpoint'], verify_files='inventory')
        if (info['identity']['qualification'] or info['seal']['updates'] != step
                or info['identity']['architecture_id'] != architecture_id
                or identity['parent_model_sha256'] != info['model_sha256']
                or identity.get('policy_lineage', {}).get('RL_updates') is not None):
            raise ValueError('DEV score is not the exact registered IL checkpoint')
        checkpoint, model_sha, identity_sha = info['path'], info['model_sha256'], fingerprint(info['identity'])
    elif step == 0:
        if identity.get('policy_lineage', {}).get('RL_updates') is not None:
            raise ValueError('RL update0 must be the selected own IL reference')
        checkpoint, model_sha = None, identity['parent_model_sha256']
        identity_sha = identity['IL_training_identity_sha256']
    else:
        info = inspect_RL_checkpoint(row['checkpoint'], verify_files='inventory')
        if (info['identity'].get('purpose') != 'formal' or info['seal']['updates'] != step
                or info['identity']['architecture_id'] != architecture_id
                or identity.get('policy_lineage', {}).get('RL_updates') != step
                or identity.get('RL_checkpoint_identity_sha256') != info['identity']['sha256']
                or identity['parent_model_sha256'] != info['identity']['IL_parent']['model_sha256']):
            raise ValueError('DEV score is not the exact registered RL checkpoint')
        checkpoint, model_sha, identity_sha = info['path'], identity['parent_model_sha256'], info['identity']['sha256']
    return dict(updates=step, checkpoint=checkpoint, model_sha256=model_sha,
        training_identity_sha256=identity_sha, deployment_model_sha256=summary['model_sha256'],
        PDMS=summary['PDMS'], evaluation=str(root.resolve()),
        evaluation_identity_sha256=identity['sha256'], summary_sha256=summary['sha256'],
        protocol_sha256=summary['protocol_sha256'], token_sha256=summary['token_sha256'],
        data_sha256=identity['data_sha256'], numerics=identity['numerics'])


def _shared(rows):
    if not rows:
        raise ValueError('An empty DEV window cannot select a parent')
    for key in ('protocol_sha256', 'token_sha256', 'data_sha256', 'numerics'):
        if any(row[key] != rows[0][key] for row in rows):
            raise ValueError('DEV selection mixes different evaluation contracts: '+key)


def select_il_parent(candidates, completed_IL_checkpoint, output=None):
    end = inspect_IL_checkpoint(completed_IL_checkpoint, verify_files='inventory')
    if end['identity']['qualification'] or end['seal']['updates'] != 100000:
        raise ValueError('Every architecture must complete100k before selecting its IL parent')
    architecture = end['identity']['architecture_id']
    rows = sorted((_candidate(row, architecture, 'IL') for row in candidates), key=lambda row:row['updates'])
    if tuple(row['updates'] for row in rows) != IL_MILESTONES:
        raise ValueError('IL selection needs all twenty full DEV milestones exactly once')
    _shared(rows)
    if any(row['training_identity_sha256'] != fingerprint(end['identity']) for row in rows):
        raise ValueError('IL selection mixes different training runs/recipes')
    selected = max(rows, key=lambda row:(row['PDMS'], -row['updates']))
    return _publish(dict(schema=IL_SELECTION_SCHEMA, architecture_id=architecture,
        completed_IL_updates=100000, endpoint_checkpoint=end['path'],
        endpoint_files_sha256=end['seal']['files_sha256'],
        training_identity_sha256=fingerprint(end['identity']), evaluated_candidates=rows,
        selected=selected, Navtest_used=False, selection_rule='maximum full DEV canonical_PDMS; exact tie earliest update',
        training_seed=42, evaluation_seed=2101), output)


def select_rl_policy(candidates, completed_RL_checkpoint, il_selection, output=None):
    parent = validate_selection(il_selection, stage='IL')
    end = inspect_RL_checkpoint(completed_RL_checkpoint, verify_files='inventory')
    architecture = end['identity']['architecture_id']
    if (end['seal']['updates'] != 50000 or end['identity'].get('purpose') != 'formal'
            or architecture != parent['architecture_id']
            or end['identity']['IL_parent']['model_sha256'] != parent['selected']['model_sha256']):
        raise ValueError('RL selection waits for50k from its own selected IL parent')
    rows = sorted((_candidate(row, architecture, 'RL') for row in candidates), key=lambda row:row['updates'])
    if tuple(row['updates'] for row in rows) != RL_MILESTONES:
        raise ValueError('RL selection requires every registered full DEV observation')
    _shared(rows)
    if (rows[0]['model_sha256'] != parent['selected']['model_sha256']
            or any(row['training_identity_sha256'] != end['identity']['sha256'] for row in rows[1:])):
        raise ValueError('RL DEV window mixes parent models or training identities')
    selected = max(rows, key=lambda row:(row['PDMS'], -row['updates']))
    return _publish(dict(schema=RL_SELECTION_SCHEMA, architecture_id=architecture,
        completed_RL_updates=50000, endpoint_checkpoint=end['path'],
        endpoint_files_sha256=end['seal']['files_sha256'], IL_selection=parent,
        IL_parent_model_sha256=parent['selected']['model_sha256'], evaluated_candidates=rows,
        selected=selected, Navtest_used=False, selection_rule='maximum full DEV canonical_PDMS; exact tie earliest update',
        training_seed=42, evaluation_seed=2101), output)


def validate_selection(path_or_data, *, stage='IL'):
    if stage not in ('IL', 'RL'):
        raise ValueError('Selection stage must be IL or RL')
    data = json.loads(Path(path_or_data).read_text()) if isinstance(path_or_data, (str, Path)) else dict(path_or_data)
    body = dict(data);saved = body.pop('sha256', None)
    schema = IL_SELECTION_SCHEMA if stage == 'IL' else RL_SELECTION_SCHEMA
    expected = IL_MILESTONES if stage == 'IL' else RL_MILESTONES
    endpoint_key = 'completed_IL_updates' if stage == 'IL' else 'completed_RL_updates'
    if (fingerprint(body) != saved or data.get('schema') != schema or data.get('Navtest_used') is not False
            or data.get('architecture_id') not in ('A', 'B', 'C', 'D')
            or data.get(endpoint_key) != expected[-1] or data.get('training_seed') != 42
            or data.get('evaluation_seed') != 2101):
        raise ValueError('Changed or non-DEV-only four-framework selection receipt')
    rows = data['evaluated_candidates']
    if tuple(row['updates'] for row in rows) != expected:
        raise ValueError('Selection receipt omits/reorders a required DEV checkpoint')
    if any(not math.isfinite(row['PDMS']) or not 0 <= row['PDMS'] <= 100 for row in rows):
        raise ValueError('Invalid DEV selection score')
    _shared(rows)
    selected = max(rows, key=lambda row:(row['PDMS'], -row['updates']))
    if selected != data['selected']:
        raise ValueError('Selection violates full-window PDMS/tie-earliest rule')
    return data


def validate_il_parent_selection(path, info):
    data = validate_selection(path, stage='IL')
    selected = data['selected']
    if (info['identity']['qualification'] or data['architecture_id'] != info['identity']['architecture_id']
            or selected['model_sha256'] != info['model_sha256']
            or selected['updates'] != info['seal']['updates']
            or selected['training_identity_sha256'] != fingerprint(info['identity'])
            or Path(selected['checkpoint']).resolve() != Path(info['path']).resolve()):
        raise ValueError('Selected own-architecture IL parent identity changed')
    return data


def validate_evaluation_selection(path, policy_role, export):
    stage = 'IL' if policy_role.startswith('il_') else 'RL'
    data = validate_selection(path, stage=stage)
    if policy_role not in ('il_endpoint', 'il_selected', 'rl_endpoint', 'rl_selected'):
        raise ValueError('Navtest policy role must be a locked endpoint or DEV-selected policy')
    row = data['evaluated_candidates'][-1] if policy_role.endswith('endpoint') else data['selected']
    rl_updates = export.get('RL_updates')
    if (data['architecture_id'] != export['architecture_id']
            or row['deployment_model_sha256'] != export['model_sha256']
            or row['model_sha256'] != export['IL_parent']['model_sha256']):
        raise ValueError('Navtest model is not the frozen DEV-observed policy')
    if stage == 'IL':
        if rl_updates is not None or export['IL_updates'] != row['updates']:
            raise ValueError('IL Navtest role was given an RL or different IL policy')
    elif (row['updates'] == 0 and rl_updates is not None) or (row['updates'] > 0 and rl_updates != row['updates']):
        raise ValueError('RL Navtest role/update changed')
    return dict(selection_sha256=data['sha256'], policy_role=policy_role,
                DEV_evaluation=row['evaluation'], DEV_evaluation_identity_sha256=row['evaluation_identity_sha256'],
                Navtest_used_for_selection=False)
