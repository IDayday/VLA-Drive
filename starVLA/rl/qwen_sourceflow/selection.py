"""Sealed DEV selection or an explicit, separately attributed user decision."""
import json
from pathlib import Path
from .contracts import file_sha,fingerprint


def validate_user_authorization(path,parent):
    record=json.loads(Path(path).read_text());body=dict(record);sha=body.pop('sha256')
    expected=dict(schema='native_IL_user_parent_authorization_v1',
        checkpoint=str(Path(parent['path']).resolve()),model_sha256=parent['files']['model.pt'],
        updates=parent['seal']['updates'],training_identity=parent['identity']['sha256'],
        Navtest_observed=True,IL_training_continues=True,RL_variants=['canonical','source'],
        RL_updates_per_variant=50000,training_seed=42,evaluation_seed=2101)
    if (fingerprint(body)!=sha or any(record.get(k)!=v for k,v in expected.items()) or
            not record.get('user_instruction') or not record.get('supersedes_selection_rule') or
            parent['identity']['qualification'] or parent['seal']['updates']<=0):
        raise ValueError('User authorization does not bind this complete, non-qualification IL parent')
    return record


def validate_selection(path,parent):
    if not path:raise ValueError('A sealed IL parent-selection receipt is required for formal RL')
    record=json.loads(Path(path).read_text());body=dict(record);sha=body.pop('sha256')
    if fingerprint(body)!=sha:
        raise ValueError('Invalid IL selection seal')
    if record['selected_model_sha256']!=parent['files']['model.pt']:
        raise ValueError('RL parent differs from locked IL selection')
    if record['schema']=='native_IL_user_selection_v1':
        if file_sha(record['authorization_path'])!=record['authorization_sha256']:
            raise ValueError('Changed user parent authorization')
        validate_user_authorization(record['authorization_path'],parent)
        selected=record['selected']
        expected=dict(checkpoint=str(Path(parent['path']).resolve()),updates=parent['seal']['updates'],
            model_sha256=parent['files']['model.pt'],training_identity=parent['identity']['sha256'])
        if (any(selected.get(k)!=v for k,v in expected.items()) or record.get('Navtest_used') is not True or
                record.get('selection_rule')!='explicit user-selected IL parent; observed Navtest disclosed'):
            raise ValueError('User-selected parent identity or selection disclosure differs')
        if file_sha(record['observed_summary_path'])!=record['observed_summary_sha256']:
            raise ValueError('Changed observed Navtest evidence')
        summary=json.loads(Path(record['observed_summary_path']).read_text())
        expected_summary=dict(status='COMPLETE',policy_type='IL',model_sha256=parent['files']['model.pt'],
            cohort='NAVTEST',scenes=12146,logs=136,metric='canonical_NAVSIM_v1.1',seed=2101,
            inference='single_candidate_source_shift5_ODE_K10',qualification=False,full_population=True,
            rl_export=None)
        if any(summary.get(k)!=v for k,v in expected_summary.items()):
            raise ValueError('User selection requires the exact observed complete IL Navtest evidence')
        return record
    if record['schema']!='native_IL_DEV_selection_v1':raise ValueError('Unknown IL selection protocol')
    if set(x['updates'] for x in record['candidates'])!={20000,40000,60000,80000,100000}:
        raise ValueError('Incomplete registered IL selection window')
    for candidate in record['candidates']:
        if file_sha(candidate['summary_path'])!=candidate['summary_sha256']:
            raise ValueError('Changed DEV evidence')
        summary=json.loads(Path(candidate['summary_path']).read_text())
        if summary.get('policy_type','IL')!='IL' or summary.get('rl_export') is not None:
            raise ValueError('RL evaluation cannot become IL selection evidence')
        if (summary['cohort']!='DEV' or summary['status']!='COMPLETE' or summary['qualification'] or
            summary['metric']!='canonical_NAVSIM_v1.1' or summary['seed']!=2101 or
            summary['inference']!='single_candidate_source_shift5_ODE_K10' or not summary['full_population']):
            raise ValueError('Parent selection must use complete registered DEV only')
    return record
