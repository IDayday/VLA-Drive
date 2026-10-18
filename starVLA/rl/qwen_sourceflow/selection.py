"""Immutable DEV-only IL parent selection after the fixed training window."""
import json
from pathlib import Path
from .contracts import file_sha,fingerprint


def validate_selection(path,parent):
    if not path:raise ValueError('A sealed DEV-only IL selection is required for formal RL')
    record=json.loads(Path(path).read_text());body=dict(record);sha=body.pop('sha256')
    if fingerprint(body)!=sha or record['schema']!='native_IL_DEV_selection_v1':
        raise ValueError('Invalid IL selection seal')
    if record['selected_model_sha256']!=parent['files']['model.pt']:
        raise ValueError('RL parent differs from locked IL selection')
    if set(x['updates'] for x in record['candidates'])!={20000,40000,60000,80000,100000}:
        raise ValueError('Incomplete registered IL selection window')
    for candidate in record['candidates']:
        if file_sha(candidate['summary_path'])!=candidate['summary_sha256']:
            raise ValueError('Changed DEV evidence')
        summary=json.loads(Path(candidate['summary_path']).read_text())
        if (summary['cohort']!='DEV' or summary['status']!='COMPLETE' or summary['qualification'] or
            summary['metric']!='canonical_NAVSIM_v1.1' or summary['seed']!=2101 or
            summary['inference']!='single_candidate_source_shift5_ODE_K10' or not summary['full_population']):
            raise ValueError('Parent selection must use complete registered DEV only')
    return record
