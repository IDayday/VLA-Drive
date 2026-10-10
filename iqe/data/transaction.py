"""Immutable round input snapshot and data commit. Pointers are reconstructible."""
from dataclasses import asdict
from pathlib import Path
from ..contracts import require
from ..io import atomic_json, digest, file_hash, read_json, lock, BlockedError
from .manifests import load_scenes, save_scenes


def committed_sources(root, before_round=None):
    records = {}
    for marker in sorted(Path(root).glob('rounds/round_*/DATA_COMMIT.json')):
        commit = read_json(marker)
        if before_round is not None and commit['round'] >= before_round:
            continue
        require(commit['commit_hash'] == digest({k:v for k,v in commit.items() if k != 'commit_hash'}), 'data commit checksum')
        for path, checksum in commit['outputs'].items():
            require(file_hash(path) == checksum, 'data commit output changed: ' + path)
        for scene in load_scenes(commit['admitted_manifest']):
            require(scene.scene_id not in records or records[scene.scene_id] == scene, 'duplicate admission conflict')
            records[scene.scene_id] = scene
    return list(records.values())


class RoundDataTransaction:
    def __init__(self, pipeline, number, old_pool, steps):
        self.pipeline, self.number = pipeline, number
        self.folder = pipeline.round_root(number)
        self.staging = self.folder/'data_staging'
        self.staging.mkdir(parents=True, exist_ok=True)
        self.old_scenes = pipeline.scenes(before_round=number)
        incoming = pipeline.config['data']['new_scene_manifest']
        previous = self.folder/'previous_serving.json'
        if not previous.exists():
            active = pipeline.root/'ACTIVE.json'
            temp = pipeline.round_root(number-1)/'bundle/BUNDLE.json'
            prior = ({'path': str(temp.parent), 'bundle_hash': read_json(temp)['bundle_hash']}
                if pipeline.mode == 'smoke' and temp.exists() else read_json(active) if active.exists() else {'policy':'locked_query_S0'})
            atomic_json(previous, prior, immutable=True)
        self.inputs = {'round':number, 'config_hash':pipeline.config['config_hash'], 'steps':steps,
            'old_scene_hash':digest([asdict(s) for s in self.old_scenes]), 'old_pool_hash':old_pool['pool_hash'],
            'old_experts':old_pool['experts'], 'new_source_manifest_hash':file_hash(incoming) if incoming else None,
            'previous_serving_hash':file_hash(previous), 'base_contract':digest(pipeline.contract)}
        path = self.staging/'INPUTS.json'
        if path.exists():
            require(read_json(path) == self.inputs, 'STALE_ROUND_DATA: manifest/pool/config content changed; use a new version')
        atomic_json(path,self.inputs,immutable=True)
        save_scenes(self.staging/'old_scenes.json',self.old_scenes)
        atomic_json(self.staging/'old_pool.json',old_pool,immutable=True)

    def point(self, name):
        callback = getattr(self.pipeline, '_data_fault_hook', None)
        if callback:
            callback(name)

    def audit_stage(self, new, quarantine):
        save_scenes(self.staging/'admitted.json',new)
        atomic_json(self.staging/'quarantine.json',quarantine,immutable=True)
        self.point('admission_staging_written')

    def commit(self):
        manifest = self.folder/'expert_manifest.json'
        require(manifest.exists(), 'expert manifest must precede data commit')
        required = [manifest, self.staging/'INPUTS.json', self.staging/'old_scenes.json',
                    self.staging/'old_pool.json', self.staging/'admitted.json', self.staging/'quarantine.json']
        required += list((self.staging/'old_pool_on_new_scenes').glob('*.json'))
        refs = []
        for s in load_scenes(self.staging/'admitted.json'):
            refs.extend(Path(p) for p in (s.observation_ref,s.target_ref,s.metric_context_ref) if Path(p).is_file())
        for pointer in (self.staging/'old_pool_on_new_scenes').glob('*.json'):
            row=read_json(pointer)
            candidate=self.pipeline.bank.candidate(row['candidate_key'])
            refs.extend([self.pipeline.bank.root/'candidates'/(row['candidate_key']+'.json'),
                         self.pipeline.bank.root/'scores'/(row['score_key']+'.json'),Path(candidate.trajectory_ref)])
        for scene in load_scenes(self.staging/'admitted.json'):
            cache=self.pipeline.features.root/self.pipeline.feature_record(scene).cache_key
            refs.extend([cache/'COMPLETE.json',cache/'features.pt'])
        outputs = {str(p.resolve()):file_hash(p) for p in required+refs}
        payload = {'schema_version':1,'round':self.number,'inputs_hash':digest(self.inputs),
                   'admitted_manifest':str((self.staging/'admitted.json').resolve()),'outputs':outputs}
        payload['commit_hash'] = digest(payload)
        with lock(self.pipeline.root/'DATA_ADMISSION.lock'):
            atomic_json(self.folder/'DATA_COMMIT.json',payload,immutable=True)
            self.point('committed_before_pointer')
            pointers = [{'path':str(p.resolve()),'checksum':file_hash(p)} for p in sorted(self.pipeline.root.glob('rounds/round_*/DATA_COMMIT.json'))]
            atomic_json(self.pipeline.root/'DATA_ADMISSION.json', {'commits':pointers,'identity':digest(pointers)})
        return payload


def external_inputs(pipeline):
    paths = [pipeline.config['data']['scene_manifest'] or pipeline.root/'scenes.json',
             pipeline.config['data']['new_scene_manifest']]
    official = pipeline.root/'OFFICIAL_SCENES.json'
    if official.exists():
        paths.extend([official,read_json(official)['path']])
    # Current admission pointer is deliberately excluded: its changing membership
    # is frozen per round in INPUTS.json, rather than an implicit old-pool input.
    return {str(p):file_hash(p) for p in paths if p is not None}


def stage_dependencies(pipeline, name, number):
    folder=pipeline.round_root(number)
    old=pipeline.round_root(max(0,number-1))
    kind='router' if pipeline.config['router_ablation']['enabled'] else 'scorer'
    prefix='router_' if kind=='router' else ''
    mapping={
        'build-round':[old/'scored.json'],
        'train-expert':[folder/'expert_manifest.json',folder/'DATA_COMMIT.json'],
        'audit-frozen':[folder/'expert/best.pt',folder/'frozen_before.json',folder/'frozen_reference.pt'],
        'export-candidates':[folder/'expert/best.pt'],
        'score-candidates':[folder/'candidates.json'],
        'score-old-pool':[folder/'candidates.json'],
        'evaluate-oracle':[folder/'scored.json',folder/'frozen_audit.json'],
        'train-'+kind:[folder/'scored.json'],
        'calibrate':[folder/kind/'best.pt',folder/'scored.json'],
        'evaluate':[folder/kind/'best.pt',folder/(prefix+'calibration.json'),folder/'scored.json'],
        'evaluate-fit-selector':[folder/kind/'best.pt',folder/(prefix+'calibration.json'),folder/'scored.json'],
        'export-bundle':[folder/kind/'best.pt',folder/(prefix+'calibration.json'),folder/'scored.json',folder/'oracle_stage_val.json',folder/(prefix+'evaluation_dev_report.json')],
        'reuse-accepted-pool':[old/'scored.json',old/'candidates.json']}
    files=mapping.get(name,[])
    hashes={str(p):file_hash(p) if p.exists() else None for p in files}
    # These hashes are immutable weight identities, independent of later status
    # transitions such as frozen -> candidate_only -> served/rejected.
    workflow=folder/'WORKFLOW.json'
    if workflow.exists():
        weights=read_json(workflow)['old_candidate_pool']
        pipeline.validate_expert_dependencies(weights)
        hashes['old_checkpoint_dependencies']=weights
    return hashes
