"""Identity-only adapter for the fixed C1 TRAIN road-failure diagnostic.

This module does not select new scenes, run a proxy, open metric caches, or score.
The evidence owner fixes the frame and binds candidates before registration.
Actual saved tensors/records are revalidated through the caller's bounded reader.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

from starVLA.model.modules.st_world.consequence import (
    ConsequenceRequest, METRICS, array_sha256, fingerprint, validated_reference_input)

SCHEMA = 'road_failure_direction_selection_v1'
EXECUTION_SCHEMA = 'road_failure_direction_execution_v1'
FRAME_SCHEMA = 'road_failure_direction_frame_v1'
SOURCE_RUNTIME_COMMIT = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
SOURCE_RUN_NAME = 'C1_fulltrain_seed42_official_4096_v1'
DOMAIN = 'st-road-failure-direction-v1'
SEED = 42
REF_NAMES = ('requests', 'results', 'complete', 'admitted', 'rank_complete', 'rollout')
MAPPING_KEYS = ('scene_id', 'source_log', 'role', 'rank')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def seeded_candidate_key(candidate_id):
    return hashlib.sha256('\0'.join((DOMAIN, str(SEED), 'candidate', candidate_id)).encode()).hexdigest()


def _hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Invalid exact SHA256')
    return value


def _ref(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256', 'bytes'}
            or not isinstance(value['path'], str) or not Path(value['path']).is_absolute()
            or type(value['bytes']) is not int or value['bytes'] <= 0):
        raise ValueError('Missing bounded immutable artifact provenance')
    _hash(value['sha256'])
    return value


def _candidate(item, candidate):
    if type(candidate.get('candidate')) is not int or candidate['candidate'] not in range(4):
        raise ValueError('Candidate must be an actual saved index 0..3')
    identity = candidate['request']
    expected = f"{item['rollout_id']}:{item['slot']}:{candidate['candidate']}"
    if (identity['candidate_id'] != expected or identity['scene_id'] != item['scene_id']
            or identity['source_log'] != item['source_log']
            or identity['policy_version'] != str(item['boundary'])
            or identity.get('generator_kind') != 'policy_snapshot'):
        raise ValueError('Wrong candidate/scene/source/policy identity')
    origin = identity.get('generator_identity')
    if (not isinstance(origin, dict) or origin.get('kind') != 'in_memory_parameter_snapshot'
            or origin.get('optimizer_step') != item['boundary']
            or origin.get('policy_version') != str(item['boundary'])
            or origin.get('state_sha256') != item['policy']['state_hash']
            or origin.get('numeric_execution_hash') != identity['inference_runtime_hash']
            or fingerprint(origin) != identity['generator_source_hash']
            or identity.get('generator_checkpoint_sha256') is not None):
        raise ValueError('Candidate source is not the actual C1 old-policy snapshot')
    for key in ('parent_checkpoint_sha256', 'parent_receipt_sha256', 'numeric_execution_hash'):
        _hash(origin[key])
    for key in ('trajectory_sha256', 'metric_cache_sha256', 'score_protocol_hash',
                'inference_runtime_hash', 'generator_source_hash'):
        _hash(identity[key])
    cache = Path(identity['metric_cache_path'])
    if (not cache.is_absolute() or list(cache.parts[-4:]) !=
            [item['source_log'], 'unknown', item['scene_id'], 'metric_cache.pkl']):
        raise ValueError('Wrong candidate context path')
    if (candidate['valid'] is not True or candidate['score_units'] != '0..1'
            or set(candidate['metrics']) != set(METRICS)
            or set(candidate['metrics_valid']) != set(METRICS)
            or any(candidate['metrics_valid'][k] is not True for k in METRICS)
            or any(not math.isfinite(candidate['metrics'][k])
                   or not 0 <= candidate['metrics'][k] <= 1 for k in METRICS)
            or type(candidate.get('native_t0')) is not bool):
        raise ValueError('Invalid original canonical labels/native t0 provenance')
    for key in ('raw_sha256', 'context_hash', 'reference_input_sha256'):
        _hash(candidate[key])
    return candidate


def validate_selection(evidence, frame):
    """Validate content, not a path-only selection or a metadata-only frame."""
    if (evidence.get('schema') != SCHEMA or evidence.get('status') != 'BOUND_BEFORE_PROXY'
            or evidence.get('source_runtime_commit') != SOURCE_RUNTIME_COMMIT
            or Path(evidence['source_run']).name != SOURCE_RUN_NAME
            or evidence.get('domain') != DOMAIN or evidence.get('seed') != SEED):
        raise ValueError('Missing final C1 TRAIN failure-selection contract')
    if (frame.get('schema') != FRAME_SCHEMA or frame.get('domain') != DOMAIN
            or frame.get('seed') != SEED or frame.get('run') != evidence['source_run']):
        raise ValueError('Foreign frozen source frame')
    selected = evidence['selected']
    if (not 1 <= len(selected) <= 32 or len(selected) != evidence['selection_count']
            or len(selected) != len(frame['selected'])
            or len({item['scene_id'] for item in selected}) != len(selected)
            or canonical_hash(selected) != evidence['selection_sha256_before_proxy']):
        raise ValueError('Selection content hash/count/unique-scene mismatch')
    for item, original in zip(selected, frame['selected']):
        if (any(item[k] != original[k] for k in MAPPING_KEYS)
                or item['boundary'] != original['policy_version']
                or item['slot'] != original['local_slot']
                or item['rollout_id'] != original['original_rollout_id']
                or item['policy'] != original['original_policy_identity']
                or item['role'] != 'train' or type(item['boundary']) is not int
                or not 0 <= item['boundary'] < 4096
                or type(item['rank']) is not int or item['rank'] not in range(4)
                or type(item['slot']) is not int or item['slot'] not in range(8)
                or original['global_scene_slot'] != item['rank'] * 8 + item['slot']):
            raise ValueError('Fixed frame boundary/rank/slot/scene/source changed')
        if (item['policy']['role'] != 'old' or item['policy']['version'] != item['boundary']
                or item['policy']['optimizer_step'] != item['boundary']):
            raise ValueError('Wrong original policy boundary')
        _hash(item['rollout_id']); _hash(item['observation_hash']); _hash(item['condition_hash'])
        for name in REF_NAMES:
            _ref(item['artifacts'][name])
        mapping = {'requests': 'original_requests', 'results': 'original_feedback_payload',
                   'rank_complete': 'original_rank_receipt', 'rollout': 'original_rollout_payload'}
        for name, frame_name in mapping.items():
            if any(item['artifacts'][name][k] != original[frame_name][k] for k in ('path', 'sha256', 'bytes')):
                raise ValueError('Source frame artifact changed')
        paths = item['artifacts']
        run = Path(evidence['source_run'])
        ledger = original['ledger']
        if (Path(paths['requests']['path']) != run/'official_feedback'/ledger/'REQUESTS.json'
                or Path(paths['results']['path']).name != 'RESULTS.pt'
                or Path(paths['results']['path']).parent.parent != run/'official_feedback'/ledger
                or Path(paths['complete']['path']) != Path(paths['results']['path']).parent/'COMPLETE.json'
                or Path(paths['admitted']['path']) != Path(paths['results']['path']).parent/'ADMITTED.json'
                or Path(paths['rank_complete']['path']) != run/'rollouts'/f"boundary_{item['boundary']:06d}"/f"rank_{item['rank']}_COMPLETE.json"
                or Path(paths['rollout']['path']) != run/'rollouts'/f"boundary_{item['boundary']:06d}"/f"rank_{item['rank']}.pt"):
            raise ValueError('Artifacts do not belong to the original C1 boundary/ledger')
        for name, key in (('requests', 'REQUESTS'), ('results', 'RESULTS'),
                          ('complete', 'COMPLETE'), ('admitted', 'ADMITTED')):
            if paths[name]['sha256'] != original['original_input_hashes'][key]:
                raise ValueError('Original ledger receipt hash changed')
        failed = _candidate(item, item['failed'])
        sibling = _candidate(item, item['safe_sibling'])
        if (failed['candidate'] == sibling['candidate']
                or failed['metrics']['drivable_area_compliance'] != 0.0
                or sibling['metrics']['drivable_area_compliance'] != 1.0):
            raise ValueError('Missing actual DAC-failed candidate and DAC-safe sibling')
        common = ('context_hash', 'reference_input_sha256')
        if any(failed[k] != sibling[k] for k in common):
            raise ValueError('Failed candidate and safe sibling use different context/reference')
        if sibling['native_t0']:raise ValueError('DAC-safe sibling cannot have a native t0 road failure')
        for key in ('metric_cache_path', 'metric_cache_sha256', 'score_protocol_hash',
                    'inference_runtime_hash', 'generator_identity'):
            if failed['request'][key] != sibling['request'][key]:
                raise ValueError('Failed candidate and safe sibling context/source mismatch')
    _ref(evidence['frame']); _ref(evidence['train_index']); _ref(evidence['source_evidence'])
    if len(evidence['source_run_artifacts']) != 3:
        raise ValueError('Missing original C1 registration/run/result provenance')
    for ref in evidence['source_run_artifacts']:_ref(ref)
    return evidence


def prepare(path):
    path = Path(path)
    if path.stat().st_size > 2 * 2**20:
        raise ValueError('Failure selection control exceeds 2MiB')
    raw = path.read_bytes(); evidence = json.loads(raw)
    frame_ref = _ref(evidence['frame'])
    if frame_ref['bytes'] > 2 * 2**20:
        raise ValueError('Failure frame control exceeds 2MiB')
    with Path(frame_ref['path']).open('rb') as stream:
        frame_raw = stream.read(frame_ref['bytes'] + 1)
    if len(frame_raw) != frame_ref['bytes'] or hashlib.sha256(frame_raw).hexdigest() != frame_ref['sha256']:
        raise ValueError('Frozen source frame checksum mismatch')
    validate_selection(evidence, json.loads(frame_raw))
    return {'evidence': {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()},
            'selection_hash': evidence['selection_sha256_before_proxy'],
            'scene_count': len(evidence['selected']),
            'existing_DAC': {'0.0': len(evidence['selected'])}}


def native_t0(record):
    import numpy as np
    flags = record['consequences']['road']['native_flag']
    value = np.asarray(flags)
    if value.dtype != np.dtype('bool') or value.shape != (41,):
        raise ValueError('Missing exact canonical native road bool[41]')
    dac = 1.0 - float(value.any())
    if record['metrics']['drivable_area_compliance'] != dac:
        raise ValueError('Original native road trace disagrees with DAC')
    return bool(value[0])


def _record(candidate, request_map, record_map):
    identity = candidate['request']
    if request_map.get(identity['candidate_id']) != identity:
        raise ValueError('Original request/selection mismatch')
    record = record_map[identity['candidate_id']]
    request = ConsequenceRequest(**identity, trajectory=record['requested_trajectory'], observe=True)
    inputs = validated_reference_input(record, request)
    if (array_sha256(record['requested_trajectory']) != identity['trajectory_sha256']
            or record['metrics'] != candidate['metrics']
            or record['metrics_valid'] != candidate['metrics_valid']
            or record.get('score_units') != candidate['score_units']
            or record.get('context_hash') != candidate['context_hash']
            or fingerprint(inputs) != candidate['reference_input_sha256']
            or native_t0(record) != candidate['native_t0']):
        raise ValueError('Original label/context/reference/native trace changed')
    return request, record


def load_inputs(evidence, budget):
    """Load one immutable source at most once; never read weights/images/cache."""
    import torch
    from starVLA.model.modules.flow_posttraining.rollout_contract import RolloutBatch
    raw_by_ref = {}
    def read(ref):
        key = (ref['path'], ref['sha256'])
        if key not in raw_by_ref:
            raw = budget.read(ref['path'], ref['sha256'], ref['bytes'])
            if len(raw) != ref['bytes']:
                raise ValueError('Immutable input byte count mismatch')
            raw_by_ref[key] = raw
        return raw_by_ref[key]
    frame = json.loads(read(evidence['frame']))
    validate_selection(evidence, frame)
    for ref in evidence['source_run_artifacts']:read(ref)
    read(evidence['source_evidence'])
    index = json.loads(read(evidence['train_index']))
    if fingerprint(index) != evidence['train_index_identity']['index_sha256']:
        raise ValueError('Train index content identity mismatch')
    train = {r['token']: r['log'] for r in index}
    rollouts = {}; ledgers = {}; loaded = []
    for item in evidence['selected']:
        refs = item['artifacts']; key = (item['boundary'], item['rank'])
        if key not in rollouts:
            receipt = json.loads(read(refs['rank_complete']))
            payload = torch.load(io.BytesIO(read(refs['rollout'])), map_location='cpu', weights_only=False)
            if (receipt['payload'] != {k: refs['rollout'][k] for k in ('path', 'sha256')}
                    or payload['schema'] != 'st_flow_update_feedback_v1'
                    or payload['identity'] != receipt['identity']):
                raise ValueError('Original rollout receipt/payload mismatch')
            rollout = RolloutBatch.from_state_dict(payload['rollout'])
            if len(rollout.scene_ids) != 8 or rollout.policy.__dict__ != item['policy']:
                raise ValueError('Wrong saved C1 rollout population/policy')
            if (receipt['identity']['rollout_id'] != rollout.rollout_id
                    or receipt['identity']['old_policy'] != item['policy']):
                raise ValueError('Original rollout identity mismatch')
            rollouts[key] = rollout
        rollout = rollouts[key]; slot = item['slot']
        if (rollout.rollout_id != item['rollout_id'] or rollout.scene_ids[slot] != item['scene_id']
                or rollout.observation_hashes[slot] != item['observation_hash']
                or rollout.condition_hashes[slot] != item['condition_hash']
                or train.get(item['scene_id']) != item['source_log']):
            raise ValueError('Selected scene/raw/provenance mismatch')
        ledger_key = refs['results']['path']
        if ledger_key not in ledgers:
            complete = json.loads(read(refs['complete']))
            if (complete['result'] != {k: refs['results'][k] for k in ('path', 'sha256')}
                    or complete['admission_sha256'] != refs['admitted']['sha256']):
                raise ValueError('Original official COMPLETE/ADMITTED mismatch')
            read(refs['admitted'])
            requests = json.loads(read(refs['requests']))
            if requests['policy_version'] != str(item['boundary']):
                raise ValueError('Wrong original request boundary')
            records = torch.load(io.BytesIO(read(refs['results'])), map_location='cpu', weights_only=False)
            request_map = {r['candidate_id']: r for r in requests['requests']}
            record_map = {r['identity']['candidate_id']: r for r in records}
            if len(request_map) != len(requests['requests']) or len(record_map) != len(records):
                raise ValueError('Duplicate original request/record candidate ID')
            ledgers[ledger_key] = request_map, record_map
        request_map, record_map = ledgers[ledger_key]
        candidates = []
        for candidate in (item['failed'], item['safe_sibling']):
            request, record = _record(candidate, request_map, record_map)
            raw = rollout.raw_actions[slot, candidate['candidate']].detach().cpu()
            if hashlib.sha256(raw.numpy().tobytes()).hexdigest() != candidate['raw_sha256']:
                raise ValueError('Selected exact FP32 raw candidate mismatch')
            candidates.append((request, record, raw))
        # Validate the fixed pre-proxy seeded candidate rule against all four
        # actual canonical records, not just the two retained witnesses.
        group = []
        for c in range(4):
            cid = f"{rollout.rollout_id}:{slot}:{c}"
            identity = request_map[cid]; record = record_map[cid]
            request = ConsequenceRequest(**identity, trajectory=record['requested_trajectory'], observe=True)
            inputs = validated_reference_input(record, request)
            if (record.get('metrics_valid', {}).get('drivable_area_compliance') is not True
                    or record.get('context_hash') != item['failed']['context_hash']
                    or fingerprint(inputs) != item['failed']['reference_input_sha256']):
                raise ValueError('Original group is not one exact official context')
            group.append((cid, record['metrics']['drivable_area_compliance']))
        for name, dac in (('failed', 0.0), ('safe_sibling', 1.0)):
            ids = [cid for cid, value in group if value == dac]
            if not ids or min(ids, key=seeded_candidate_key) != item[name]['request']['candidate_id']:
                raise ValueError('Fixed seeded failed/safe-sibling candidate rule changed')
        request, record, raw = candidates[0]
        normalized = {**item, **item['failed']['request'],
                      'candidate': item['failed']['candidate'],
                      'raw_sha256': item['failed']['raw_sha256']}
        loaded.append((normalized, raw, request, record))
    return loaded


def bind_frame(frame_path, frame_sha256, legacy_evidence_path, budget, decode_gate):
    """Bounded, explicit preparation only; no proxy, simulator, cache or model.

    The frozen metadata frame determines scene order. Whole native containers
    must be deserialized, but only the selected rank/slot's four records are
    examined. A failure never substitutes another scene or candidate rule.
    """
    import torch
    from starVLA.model.modules.flow_posttraining.rollout_contract import RolloutBatch
    frame_raw=budget.read(frame_path,_hash(frame_sha256),2*2**20)
    frame=json.loads(frame_raw)
    if (frame.get('schema')!=FRAME_SCHEMA or frame.get('domain')!=DOMAIN
            or frame.get('seed')!=SEED or not 1<=len(frame['selected'])<=32
            or Path(frame['run']).name!=SOURCE_RUN_NAME
            or len({item['scene_id'] for item in frame['selected']})!=len(frame['selected'])
            or canonical_hash(frame['selected'])!=frame['selection_frame_sha256']):
        raise ValueError('Missing immutable fixed unique TRAIN source frame')
    budget.binding_requested_scene_ids=[row['scene_id'] for row in frame['selected']]
    budget.binding_bound_scene_ids=[]
    legacy_path=Path(legacy_evidence_path)
    if legacy_path.stat().st_size>2*2**20:raise ValueError('Oversized original evidence metadata')
    budget.check()
    with legacy_path.open('rb') as stream:legacy_raw=stream.read(2*2**20+1)
    legacy_sha=hashlib.sha256(legacy_raw).hexdigest()
    # Discovery of this small control checksum is an actual read as well as
    # the subsequent pinned verification; account both, never a hidden scan.
    budget.read_bytes+=len(legacy_raw)
    budget.reads.append({'path':str(legacy_path),'bytes':len(legacy_raw),'sha256':legacy_sha,'purpose':'small_control_checksum_discovery'})
    if len(legacy_raw)>2*2**20:raise ValueError('Original evidence control grew beyond bound')
    if budget.read_bytes>budget.limits['asset_read_bytes']:raise ValueError('Binding asset read cap')
    legacy=json.loads(budget.read(legacy_path,legacy_sha,2*2**20))
    if legacy.get('source_runtime_commit')!=SOURCE_RUNTIME_COMMIT:
        raise ValueError('Foreign original canonical runtime')
    entries={entry['path']:entry for entry in legacy['budget']['entries']}
    run=Path(frame['run']); source_refs=[]
    for filename in ('identity.json','RESULT.json'):
        entry=entries[str(run/filename)]
        source_refs.append(_ref({k:entry[k] for k in ('path','sha256','bytes')}))
        budget.read(entry['path'],entry['sha256'],entry['bytes'])
    registration_entry=next(entry for entry in entries.values()
        if Path(entry['path']).name=='REGISTRATION.json' and Path(entry['path']).parent.name==SOURCE_RUN_NAME)
    source_refs.append(_ref({k:registration_entry[k] for k in ('path','sha256','bytes')}))
    budget.read(registration_entry['path'],registration_entry['sha256'],registration_entry['bytes'])
    index_entry=next(entry for entry in entries.values() if entry['path'].endswith('/student_train_v1/index.json'))
    index_ref=_ref({k:index_entry[k] for k in ('path','sha256','bytes')})
    index=json.loads(budget.read(index_ref['path'],index_ref['sha256'],index_ref['bytes']))
    if fingerprint(index)!=legacy['train_index_identity']['index_sha256']:
        raise ValueError('Original TRAIN index identity changed')
    train={row['token']:row['log'] for row in index}
    raw_cache={};rollouts={};ledgers={};selected=[];decode_audits=[]
    def read(ref):
        ref=_ref({k:ref[k] for k in ('path','sha256','bytes')})
        key=(ref['path'],ref['sha256'])
        if key not in raw_cache:
            raw=budget.read(ref['path'],ref['sha256'],ref['bytes'])
            if len(raw)!=ref['bytes']:raise ValueError('Source byte count changed')
            raw_cache[key]=raw
        return raw_cache[key]
    for original in frame['selected']:
        budget.binding_active_scene_id=original['scene_id']
        if train.get(original['scene_id'])!=original['source_log'] or original['role']!='train':
            raise ValueError('Frozen scene/source is not an original TRAIN observation')
        boundary=original['policy_version'];rank=original['rank'];slot=original['local_slot']
        refs={name:{k:original[field][k] for k in ('path','sha256','bytes')}
              for name,field in (('requests','original_requests'),('results','original_feedback_payload'),
                                 ('rank_complete','original_rank_receipt'),('rollout','original_rollout_payload'))}
        for name,filename in (('complete','COMPLETE.json'),('admitted','ADMITTED.json')):
            path=Path(refs['results']['path']).parent/filename
            refs[name]={'path':str(path),'sha256':original['original_input_hashes'][filename.split('.')[0]],
                        'bytes':path.stat().st_size}
        key=(boundary,rank)
        if key not in rollouts:
            receipt=json.loads(read(refs['rank_complete']))
            payload=torch.load(io.BytesIO(read(refs['rollout'])),map_location='cpu',weights_only=False)
            if (payload.get('schema')!='st_flow_update_feedback_v1' or payload['identity']!=receipt['identity']
                    or receipt['payload']!={k:refs['rollout'][k] for k in ('path','sha256')}):
                raise ValueError('Original rank payload/COMPLETE mismatch')
            rollout=RolloutBatch.from_state_dict(payload['rollout'])
            if len(rollout.scene_ids)!=8:raise ValueError('Wrong original rank population')
            rollouts[key]=rollout
        rollout=rollouts[key]
        if (rollout.rollout_id!=original['original_rollout_id']
                or rollout.policy.__dict__!=original['original_policy_identity']
                or rollout.scene_ids[slot]!=original['scene_id']):
            raise ValueError('Frozen frame and actual C1 rollout differ')
        ledger_key=refs['results']['path']
        if ledger_key not in ledgers:
            complete=json.loads(read(refs['complete']));read(refs['admitted'])
            if (complete['admission_sha256']!=refs['admitted']['sha256']
                    or complete['result']!={k:refs['results'][k] for k in ('path','sha256')}):
                raise ValueError('Original official ledger receipt mismatch')
            requests=json.loads(read(refs['requests']))
            records=torch.load(io.BytesIO(read(refs['results'])),map_location='cpu',weights_only=False)
            request_map={row['candidate_id']:row for row in requests['requests']}
            record_map={row['identity']['candidate_id']:row for row in records}
            if (len(request_map)!=len(requests['requests']) or len(record_map)!=len(records)
                    or requests['policy_version']!=str(boundary)):
                raise ValueError('Original official ledger candidates/version invalid')
            ledgers[ledger_key]=(request_map,record_map)
        request_map,record_map=ledgers[ledger_key]
        item={k:original[k] for k in MAPPING_KEYS}
        item.update(boundary=boundary,slot=slot,policy=original['original_policy_identity'],
                    rollout_id=rollout.rollout_id,artifacts=refs,
                    observation_hash=rollout.observation_hashes[slot],condition_hash=rollout.condition_hashes[slot])
        candidates=[];common=None
        for candidate in range(4):
            cid=f'{rollout.rollout_id}:{slot}:{candidate}'
            identity=request_map[cid];record=record_map[cid]
            request=ConsequenceRequest(**identity,trajectory=record['requested_trajectory'],observe=True)
            inputs=validated_reference_input(record,request)
            context=(record['context_hash'],fingerprint(inputs))
            if common is not None and context!=common:raise ValueError('Original sibling context/reference changed')
            common=context
            raw=rollout.raw_actions[slot,candidate].detach().cpu()
            value={'candidate':candidate,'request':identity,'raw_sha256':hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                'context_hash':context[0],'reference_input_sha256':context[1],
                'valid':record['valid'],'metrics':record['metrics'],'metrics_valid':record['metrics_valid'],
                'score_units':record['score_units'],'native_t0':native_t0(record)}
            _candidate(item,value)
            if array_sha256(record['requested_trajectory'])!=identity['trajectory_sha256']:
                raise ValueError('Original physical payload hash mismatch')
            candidates.append(value)
        for name,dac in (('failed',0.0),('safe_sibling',1.0)):
            values=[row for row in candidates if row['metrics']['drivable_area_compliance']==dac]
            if not values:raise ValueError('Frozen scene has no actual DAC-failed/safe-sibling witness')
            value=min(values,key=lambda row:seeded_candidate_key(row['request']['candidate_id']))
            item[name]=value
            raw=rollout.raw_actions[slot,value['candidate']].detach().cpu()
            audit=decode_gate(raw,record_map[value['request']['candidate_id']]['requested_trajectory'])
            decode_audits.append({'scene_id':item['scene_id'],'kind':name,'audit':audit})
        budget.check();selected.append(item);budget.binding_bound_scene_ids.append(item['scene_id'])
    evidence={'schema':SCHEMA,'status':'BOUND_BEFORE_PROXY','domain':DOMAIN,'seed':SEED,
        'source_run':str(run),'source_runtime':legacy['source_runtime'],
        'source_runtime_commit':SOURCE_RUNTIME_COMMIT,'sources':legacy['sources'],
        'source_run_artifacts':source_refs,'source_evidence':{'path':str(legacy_path.resolve()),'sha256':legacy_sha,'bytes':len(legacy_raw)},
        'frame':{'path':str(Path(frame_path).resolve()),'sha256':frame_sha256,'bytes':len(frame_raw)},
        'train_index':index_ref,'train_index_identity':legacy['train_index_identity'],
        'selection_count':len(selected),'selected':selected,
        'selection_sha256_before_proxy':canonical_hash(selected)}
    validate_selection(evidence,frame)
    return evidence,{'decode_audits':decode_audits,'requested_scenes':len(selected),
        'bound_scenes':len(selected),'source_ledgers':len(ledgers),'source_rank_payloads':len(rollouts),
        'new_official_calls':0,'proxy_calls':0,'model_loads':0,'cuda_initialized':torch.cuda.is_initialized()}


def execution_read_plan(evidence):
    """Stat only known referenced assets; no metric-cache content is read."""
    unique={}
    refs=[evidence['frame'],evidence['train_index'],evidence['source_evidence'],*evidence['source_run_artifacts']]
    for item in evidence['selected']:refs.extend(item['artifacts'].values())
    for ref in refs:
        key=ref['path']
        value={'path':key,'sha256':ref['sha256'],'bytes':ref['bytes']}
        if key in unique and unique[key]!=value:raise ValueError('Conflicting source reference')
        unique[key]=value
    for ref in evidence['sources']:
        unique[ref['path']]={**ref,'bytes':Path(ref['path']).stat().st_size}
    contexts={}
    for item in evidence['selected']:
        request=item['failed']['request'];path=Path(request['metric_cache_path'])
        size=path.stat().st_size
        if size>64*2**20:raise ValueError('Selected context exceeds unchanged per-cache read bound')
        contexts[str(path)]={'path':str(path),'sha256':request['metric_cache_sha256'],'bytes':size}
    selection_bytes=len((json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    total=sum(ref['bytes'] for ref in unique.values())+sum(ref['bytes'] for ref in contexts.values())+selection_bytes
    return {'scope':'worker logical immutable inputs plus one selected context read per scene; control/admission overhead separate',
        'unique_source_assets':list(unique.values()),'context_stat_only':list(contexts.values()),
        'selection_bytes':selection_bytes,'total_worker_asset_bytes':total,'metric_cache_content_reads':0}
