"""Explicit user-requested IL Navtest, separate from completed-IL/RL selection."""
import json
from pathlib import Path
from .contracts import fingerprint


def validate_interim(path, *, parent, model_sha256, data_sha256, tokens,
                     population_sha256, protocol_sha256, output, seed):
    record = json.loads(Path(path).read_text()); body = dict(record)
    if body.pop('sha256', None) != fingerprint(body): raise ValueError('Changed interim registration')
    expected = dict(schema='world_postcot_user_interim_NAVTEST_v1',
        checkpoint=parent['path'], native_model_sha256=parent['model_sha256'],
        model_sha256=model_sha256, updates=parent['seal']['updates'],
        checkpoint_files_sha256=parent['seal']['files_sha256'], data_sha256=data_sha256,
        token_sha256=fingerprint(tokens), population_sha256=population_sha256,
        protocol_sha256=protocol_sha256, output=str(Path(output).resolve()),
        seed=2101, count=12146, prediction_type='x0',
        inference='single_candidate_cleanx_K10_shift5',
        used_for_parent_selection=False, used_for_training_or_tuning=False)
    if (seed != 2101 or len(tokens) != 12146 or len(set(tokens)) != 12146 or
            parent['identity']['qualification'] or parent['seal']['updates'] not in (37000, 40000) or
            any(record.get(k) != value for k, value in expected.items())):
        raise ValueError('Interim Navtest differs from the exact user-pinned full IL37k/40k policy')
    return record
