"""Explicit user-requested intermediate Navtest, separate from IL selection."""
import json
from pathlib import Path
from .contracts import fingerprint


def validate_interim_registration(path, parent, data, tokens, output):
    record=json.loads(Path(path).read_text());body=dict(record);seal=body.pop('sha256')
    if fingerprint(body)!=seal or record['schema']!='native_user_interim_NAVTEST_v1':
        raise ValueError('Invalid interim evaluation registration')
    expected=dict(checkpoint=parent['path'],model_sha256=parent['files']['model.pt'],
        updates=parent['seal']['updates'],data_sha256=data.sha,token_sha256=fingerprint(tokens),
        count=len(tokens),logs=len({data.records[t]['log'] for t in tokens}),
        seed=2101,metric='canonical_NAVSIM_v1.1',inference='single_candidate_source_shift5_ODE_K10',
        output=str(Path(output).resolve()))
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('Interim registration differs from model/data/output/protocol')
    if (record['count']!=12146 or record['logs']!=136 or data.manifest['cohort']!='NAVTEST' or
        tokens!=data.tokens or record.get('used_for_parent_selection') is not False or
        record.get('used_for_training_or_tuning') is not False):
        raise ValueError('Interim Navtest requires full population and no selection/tuning use')
    return record
