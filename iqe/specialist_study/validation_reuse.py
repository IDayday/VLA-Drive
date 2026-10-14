"""Read the immutable trajectories and labels which actually selected an endpoint.

Pool construction is an analysis of completed validation, not another numerical
inference run. A missing/corrupt record is an error, never a new prediction.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from iqe.contracts import CandidateRecord, ScoreRecord, require, strict_record
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.internal_baseline.bank import protocol


def completed_endpoint_records(config, item, scenes):
    require(scenes and all(s.split_role == 'stage_val' for s in scenes),
            'completed endpoint selection only reads fixed stage_val')
    checkpoint = Path(item['checkpoint'])
    receipt = read_json(str(checkpoint)+'.COMPLETE.json')
    require(file_hash(checkpoint) == item['checksum'] == receipt['checksum']
            and receipt['model_hash'] == item['model_hash']
            and receipt['step'] == item['step'] and receipt['endpoint'] == item['endpoint'],
            'selected endpoint checksum/identity mismatch')
    validation_path = config.root/'experts'/item['cell']/f"validation_{item['step']:06d}.json"
    validation = read_json(validation_path)['endpoints'][item['endpoint']]
    require(validation['candidate_hash'] == item['model_hash']
            and validation['scenes'] == len(scenes), 'completed validation population/endpoint changed')
    binding = read_json(config.root/'BASE_BINDING.json')
    contract = read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']
    proto = protocol(config)
    expert_id = item['cell']+'_'+item['endpoint']
    folder = config.root/'candidates'/expert_id/item['model_hash']
    pairs, evidence = [], []
    for scene in scenes:
        path = folder/(scene.scene_id+'.json')
        require(path.is_file(), 'completed validation candidate missing: '+scene.scene_id)
        candidate = strict_record(CandidateRecord, read_json(path))
        require(candidate.scene_id == scene.scene_id and candidate.expert_id == expert_id
                and candidate.expert_checkpoint_hash == item['model_hash']
                and candidate.feature_contract_hash == binding['contract_hash']
                and candidate.generation_config_hash == config.identity and candidate.finite_valid,
                'completed validation candidate identity mismatch')
        trajectory = path.with_suffix('.npz')
        require(Path(candidate.trajectory_ref) == trajectory
                and file_hash(trajectory) == candidate.trajectory_hash,
                'completed validation trajectory corruption')
        require(candidate.horizon == contract['horizon'] and candidate.dt == contract['dt']
                and candidate.raw_action_representation == contract['raw_representation']
                and candidate.physical_frame == contract['frame'], 'completed trajectory contract changed')
        with np.load(trajectory) as stored:
            value = stored['trajectory']
            require(value.dtype == np.float32 and value.shape == (contract['horizon'], 3)
                    and np.isfinite(value).all(), 'completed trajectory is invalid')
        seed = int(digest({'scene_id': scene.scene_id, 'seed': 42, 'repetition': 0})[:8], 16)
        request = {'candidate': asdict(candidate), 'scene': asdict(scene),
                   'protocol_hash': proto['protocol_hash'], 'navsim_root': proto['navsim_root'],
                   'source_hash': proto['navsim_source_hash'], 'config': proto['config'],
                   'seed': seed, 'eval_seed': 42, 'repetition': 0}
        score_path = config.root/'scores'/(digest([candidate.key, scene.metric_context_hash,
                                                  proto['protocol_hash'], seed])+'.json')
        require(score_path.is_file() and score_path.with_suffix('.COMPLETE.json').is_file(),
                'completed official validation receipt missing: '+scene.scene_id)
        score_receipt = read_json(score_path.with_suffix('.COMPLETE.json'))
        require(score_receipt['checksum'] == file_hash(score_path)
                and score_receipt['request_hash'] == digest(request),
                'completed validation observation/context/request changed')
        score = strict_record(ScoreRecord, read_json(score_path))
        require(score.score_valid and score.candidate_key == candidate.key
                and score.metric_context_hash == scene.metric_context_hash
                and score.score_protocol_hash == proto['protocol_hash']
                and score.eval_seed == 42 and score.repetition_index == 0,
                'completed validation label invalid or stale')
        pairs.append((candidate, score))
        evidence.append({'scene_id': scene.scene_id, 'candidate_key': candidate.key,
                         'record_hash': file_hash(path), 'score_hash': score_receipt['checksum'],
                         'request_hash': score_receipt['request_hash']})
    observed_mean = float(np.mean([s.total_score_01 for _, s in pairs]))
    require(abs(observed_mean-validation['fixed_policy_01'][1]) < 1e-12,
            'completed validation labels disagree with endpoint selection result')
    audit = {'schema': 'completed_validation_reuse_v1', 'cell': item['cell'],
             'step': item['step'], 'endpoint': item['endpoint'], 'checkpoint_hash': item['checksum'],
             'validation_hash': file_hash(validation_path), 'Base_binding_hash': file_hash(config.root/'BASE_BINDING.json'),
             'scene_manifest_hash': digest([asdict(s) for s in scenes]), 'records': evidence,
             'model_forward_calls': 0, 'official_backend_calls': 0, 'original_artifacts_modified': False}
    atomic_json(config.root/'runtime/validation_reuse'/
                f"{item['cell']}_{item['step']:06d}_{item['endpoint']}.json", audit, immutable=True)
    return pairs
