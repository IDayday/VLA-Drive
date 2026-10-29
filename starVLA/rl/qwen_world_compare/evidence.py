"""Exact-candidate native event evidence with atomic scene commits.

This adds labels to an immutable physical candidate bank; it never creates
on-policy probabilities, substitutes logged future features for counterfactuals,
or infers event time from a final metric score.
"""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np

from starVLA.model.modules.qwen_world_compare.contracts import first_event_class
from starVLA.model.modules.qwen_world_postcot.candidate_bank import (
    OFFICIAL_METRIC_NAMES, CandidateBank, fingerprint, file_sha, trajectory_hash,
)

SCHEMA = 'qwen_world_compare_native_candidate_evidence_v1'
EVENT_ORDER = ('NC', 'DAC', 'TTC')
EVENT_RULE = dict(horizon_seconds=4., simulation_interval_seconds=.1,
    boundaries='[0,.5],(.5,1],...,(3.5,4]; exact no-event class8',
    boundary_tolerance_seconds=1e-8, no_event_class=8,
    NC='official at-fault collision earliest index, including score0.5',
    DAC='first true in official NON_DRIVABLE_AREA body mask',
    TTC='official TTC infraction earliest index; not a minimum TTC value')


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.incomplete.' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def make_evidence_record(candidate, feedback, *, source_bank_hash, evaluator_identity,
                         score_tolerance=5e-12):
    """Validate an actual native replay and retain exact lineage and event time."""
    candidate.validate()
    if (feedback.get('status') != 'OK' or feedback['token'] != candidate.scene_id
            or feedback['log'] != candidate.source_log
            or feedback['protocol_sha256'] != candidate.score_protocol_sha256
            or feedback['cache']['sha256'] != candidate.context_sha256):
        raise ValueError('Native evidence replay changed scene, evaluator or metric cache')
    if trajectory_hash(feedback['requested_trajectory']) != candidate.trajectory_sha256:
        raise ValueError('Native evidence request is not the exact bank trajectory')
    m = feedback['metrics']
    for name in OFFICIAL_METRIC_NAMES:
        if not candidate.valid_masks[name] or not np.isfinite(m[name]) or not 0 <= m[name] <= 1:
            raise ValueError('Missing/invalid factor is not an ordinary training zero')
        if abs(m[name] - candidate.raw_factor_labels[name]) > score_tolerance:
            raise ValueError('Canonical factor changed during event extraction: '+name)
    if not np.isfinite(m['score']) or abs(m['score'] - candidate.raw_total_score) > score_tolerance:
        raise ValueError('Canonical total score changed during event extraction')
    nc = feedback['official_first_at_fault_collision_seconds']
    ttc = feedback['official_first_ttc_infraction_seconds']
    road = np.asarray(feedback['non_drivable_body_by_time'])
    if road.shape != (41,) or road.dtype != np.bool_:
        raise ValueError('Native road-body mask must cover t0..4s at0.1s')
    idx = np.flatnonzero(road)
    dac = None if len(idx) == 0 else float(idx[0] * .1)
    first = [None if t is None else float(t) for t in (nc, dac, ttc)]
    classes = [first_event_class(t) for t in first]
    # This checks semantic alignment; event labels still come exclusively from
    # native times/masks, never by filling a time bin based on these metrics.
    for name, t, key in zip(EVENT_ORDER, first, ('no_at_fault_collisions',
            'drivable_area_compliance', 'time_to_collision_within_bound')):
        if (m[key] < 1) != (t is not None):
            raise ValueError('Native event/metric disagreement: '+name)
    value = dict(schema=SCHEMA, record_id=candidate.record_id,
        old_record_id=candidate.record_id, scene_token=candidate.scene_id,
        source_log=candidate.source_log, split=candidate.split,
        timestamp_us=candidate.timestamp_us, observation_hash=candidate.observation_hash,
        trajectory_fp32=candidate.trajectory_fp32, trajectory_sha256=candidate.trajectory_sha256,
        coordinate_version=candidate.coordinate_version,
        normalization_version=candidate.normalization_version,
        source_bank_hash=source_bank_hash, source_evaluator_revision=candidate.evaluator_revision,
        evaluator_identity=evaluator_identity, evaluator_sha256=fingerprint(evaluator_identity),
        metric_cache=dict(feedback['cache']), score_protocol_sha256=feedback['protocol_sha256'],
        factor_labels={key: float(m[key]) for key in OFFICIAL_METRIC_NAMES},
        factor_valid={key: True for key in OFFICIAL_METRIC_NAMES},
        canonical_total_score=float(m['score']), total_valid=True,
        event_order=list(EVENT_ORDER), event_class=classes, event_valid=[True]*3,
        event_first_seconds=first, native_non_drivable_body_by_time=road.tolist(),
        event_rule_sha256=fingerprint(EVENT_RULE),
        simulated_states=_array_hash(feedback['simulated_states']),
        simulated_body_coordinates=_array_hash(feedback['simulated_body_coordinates']),
        raw_progress_meters=float(feedback['raw_progress_meters']),
        physical_scoring_seconds=float(feedback['scoring_seconds']),
        supervision_source='native simulator/scorer exact replay, no learned risk label')
    value['sha256'] = fingerprint(value)
    return value


def validate_record(value, *, identity=None):
    body = dict(value); expected = body.pop('sha256')
    if (fingerprint(body) != expected or value['schema'] != SCHEMA
            or value['record_id'] != value['old_record_id'] or value['split'] != 'TRAIN'
            or value['trajectory_sha256'] != trajectory_hash(value['trajectory_fp32'])
            or value['event_order'] != list(EVENT_ORDER)
            or value['event_rule_sha256'] != fingerprint(EVENT_RULE)
            or value['event_class'] != [first_event_class(t) for t in value['event_first_seconds']]
            or value['event_valid'] != [True]*3):
        raise ValueError('Altered or invalid native candidate evidence')
    expected_id = fingerprint([value['scene_token'], value['observation_hash'],
        value['trajectory_sha256'], value['score_protocol_sha256'], value['metric_cache']['sha256']])
    if (value['record_id'] != expected_id or set(value['factor_labels']) != set(OFFICIAL_METRIC_NAMES)
            or set(value['factor_valid']) != set(OFFICIAL_METRIC_NAMES)
            or not all(value['factor_valid'].values()) or value['total_valid'] is not True
            or not all(math.isfinite(t) and 0 <= t <= 1 for t in value['factor_labels'].values())
            or not math.isfinite(value['canonical_total_score']) or not 0 <= value['canonical_total_score'] <= 1):
        raise ValueError('Evidence factor/score/record identity contract changed')
    if identity is not None and any(value[key] != identity[key] for key in
            ('source_bank_hash', 'score_protocol_sha256', 'evaluator_sha256')):
        raise ValueError('Evidence record and bank identity differ')
    return value


class EvidenceWriter:
    """One part owner, one atomic SQLite transaction per complete scene."""
    def __init__(self, root, identity, scenes, *, resume=False):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/'writer.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.root/'COMPLETE.json').exists():
            raise FileExistsError('A complete evidence part is immutable')
        if not scenes or len(scenes) != len(set(scenes)):
            raise ValueError('Evidence scene list must be unique and nonempty')
        self.identity, self.scenes = dict(identity), list(scenes)
        self.build = dict(schema=SCHEMA, identity=self.identity, scenes=self.scenes)
        path = self.root/'BUILD_IDENTITY.json'
        if path.exists():
            if not resume or json.loads(path.read_text()) != self.build:
                raise ValueError('Evidence resume requires identical source/scenes/evaluator')
        else:
            atomic_json(path, self.build)
        self.db = sqlite3.connect(self.root/'index.sqlite3')
        self.db.execute('PRAGMA journal_mode=DELETE'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY,scene_token TEXT NOT NULL,payload TEXT NOT NULL,checksum TEXT NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS scene_idx ON records(scene_token)')
        self.db.execute('CREATE TABLE IF NOT EXISTS scenes(scene_token TEXT PRIMARY KEY,record_count INTEGER NOT NULL,scene_hash TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts(attempt_id TEXT PRIMARY KEY,scene_token TEXT NOT NULL,candidates INTEGER NOT NULL,start_unix REAL NOT NULL,status TEXT NOT NULL,returned_candidates INTEGER NOT NULL DEFAULT 0,wall_seconds REAL,simulator_batches INTEGER)')
        # An interrupted native call can have spent physical CPU even though no
        # scene transaction was committed. Preserve that uncertainty explicitly.
        self.db.execute("UPDATE attempts SET status='INTERRUPTED_UNKNOWN_PARTIAL' WHERE status='RUNNING'")
        self.db.commit()

    def has_scene(self, scene):
        return self.db.execute('SELECT 1 FROM scenes WHERE scene_token=?', (scene,)).fetchone() is not None

    def begin_attempt(self, scene, candidates):
        key = uuid.uuid4().hex
        self.db.execute('INSERT INTO attempts(attempt_id,scene_token,candidates,start_unix,status) VALUES (?,?,?,?,?)',
                        (key, scene, candidates, time.time(), 'RUNNING'))
        self.db.commit()
        return key

    def finish_attempt(self, key, *, status, returned_candidates, wall_seconds, simulator_batches):
        self.db.execute('UPDATE attempts SET status=?,returned_candidates=?,wall_seconds=?,simulator_batches=? WHERE attempt_id=?',
                        (status, returned_candidates, wall_seconds, simulator_batches, key))
        self.db.commit()

    def accounting(self):
        row = self.db.execute('SELECT COUNT(*),COALESCE(SUM(candidates),0),COALESCE(SUM(returned_candidates),0),COALESCE(SUM(wall_seconds),0),COALESCE(SUM(simulator_batches),0) FROM attempts').fetchone()
        unknown = self.db.execute("SELECT COUNT(*) FROM attempts WHERE status IN ('INTERRUPTED_UNKNOWN_PARTIAL','FAILED_UNKNOWN_PARTIAL')").fetchone()[0]
        return dict(native_scene_attempts=row[0], candidate_attempts=row[1],
                    completed_candidate_scores=row[2], known_scoring_wall_seconds=row[3],
                    simulator_batches=row[4], interrupted_partial_calls_unknown_cost=unknown)

    def append_scene(self, scene, values):
        if scene not in self.scenes or not values or len({v['record_id'] for v in values}) != len(values):
            raise ValueError('Incomplete or duplicate evidence scene')
        for value in values:
            validate_record(value, identity=self.identity)
            if value['scene_token'] != scene:
                raise ValueError('Cross-scene evidence join')
        scene_hash = fingerprint(sorted(v['sha256'] for v in values))
        prior = self.db.execute('SELECT record_count,scene_hash FROM scenes WHERE scene_token=?', (scene,)).fetchone()
        if prior is not None:
            if tuple(prior) != (len(values), scene_hash):
                raise ValueError('Committed evidence scene changed')
            return False
        try:
            for v in values:
                payload = json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
                self.db.execute('INSERT INTO records VALUES (?,?,?,?)',
                                (v['record_id'], scene, payload, hashlib.sha256(payload.encode()).hexdigest()))
            self.db.execute('INSERT INTO scenes VALUES (?,?,?)', (scene, len(values), scene_hash))
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise
        return True

    def counts(self):
        return dict(scenes=self.db.execute('SELECT COUNT(*) FROM scenes').fetchone()[0],
                    records=self.db.execute('SELECT COUNT(*) FROM records').fetchone()[0])

    def finish(self):
        actual = self.counts()
        accounting = self.accounting()
        if actual['scenes'] != len(self.scenes) or {r[0] for r in self.db.execute('SELECT scene_token FROM scenes')} != set(self.scenes):
            raise ValueError('Cannot publish partial native evidence')
        self.db.commit(); self.db.close(); self.db = None
        manifest = dict(schema=SCHEMA, status='COMPLETE', kind='part', identity=self.identity,
                        scenes=self.scenes, counts=actual, accounting=accounting,
                        index_sha256=file_sha(self.root/'index.sqlite3'))
        manifest['sha256'] = fingerprint(manifest)
        atomic_json(self.root/'MANIFEST.json', manifest)
        atomic_json(self.root/'COMPLETE.json', dict(status='COMPLETE', manifest_sha256=file_sha(self.root/'MANIFEST.json'),
                                                  bank_hash=manifest['sha256']))
        return manifest

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None
        if self.lock is not None:
            self.lock.close(); self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_manifest(root):
    root = Path(root); p = root/'MANIFEST.json'
    manifest = json.loads(p.read_text()); seal = json.loads((root/'COMPLETE.json').read_text())
    body = dict(manifest); expected = body.pop('sha256')
    if (manifest['schema'] != SCHEMA or manifest['status'] != 'COMPLETE'
            or fingerprint(body) != expected or seal['bank_hash'] != expected
            or seal['manifest_sha256'] != file_sha(p)):
        raise ValueError('Unsealed or altered evidence bank')
    return manifest


class EvidenceBank:
    """Read only completed evidence; get() uses the unchanged old record id."""
    def __init__(self, root, *, source_bank_hash=None, maximum_open_parts=8):
        self.root = Path(root); self.manifest = read_manifest(root)
        if source_bank_hash is not None and self.manifest['identity']['source_bank_hash'] != source_bank_hash:
            raise ValueError('Evidence attached to a different fixed candidate bank')
        if file_sha(self.root/'index.sqlite3') != self.manifest['index_sha256']:
            raise ValueError('Evidence index changed')
        self.db = sqlite3.connect(f'file:{self.root / "index.sqlite3"}?mode=ro&immutable=1', uri=True)
        self.parts = OrderedDict(); self.maximum_open_parts = maximum_open_parts
        self.identity = self.manifest['identity']; self.bank_hash = self.manifest['sha256']

    def _part(self, name):
        if name in self.parts:
            self.parts.move_to_end(name)
            return self.parts[name]
        path = self.root/name
        if not path.resolve().is_relative_to(self.root.resolve()) or name not in self.manifest['parts']:
            raise ValueError('Unregistered evidence part path')
        m = read_manifest(path)
        if file_sha(path/'MANIFEST.json') != self.manifest['parts'][name]['manifest_sha256']:
            raise ValueError('Evidence part manifest changed')
        if m['identity'] != self.identity or file_sha(path/'index.sqlite3') != m['index_sha256']:
            raise ValueError('Evidence part identity/content changed')
        db = sqlite3.connect(f'file:{path / "index.sqlite3"}?mode=ro&immutable=1', uri=True)
        self.parts[name] = db
        while len(self.parts) > self.maximum_open_parts:
            self.parts.popitem(last=False)[1].close()
        return db

    def get(self, record_id):
        if self.manifest['kind'] == 'part':
            row = self.db.execute('SELECT payload,checksum FROM records WHERE record_id=?', (record_id,)).fetchone()
        else:
            part = self.db.execute('SELECT part FROM records WHERE record_id=?', (record_id,)).fetchone()
            if part is None:
                raise KeyError('No native evidence for exact candidate '+record_id)
            row = self._part(part[0]).execute('SELECT payload,checksum FROM records WHERE record_id=?', (record_id,)).fetchone()
        if row is None:
            raise KeyError('No native evidence for exact candidate '+record_id)
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError('Evidence record checksum changed')
        return validate_record(json.loads(row[0]), identity=self.identity)

    def scene_records(self, token):
        ids = [row[0] for row in self.db.execute('SELECT record_id FROM records WHERE scene_token=? ORDER BY record_id', (token,))]
        if not ids:
            raise KeyError('Scene absent from native evidence '+token)
        return [self.get(record_id) for record_id in ids]

    def close(self):
        self.db.close()
        for db in self.parts.values():
            db.close()
        self.parts.clear()


def merge_evidence(root, part_roots, *, expected_scenes):
    """Build a small sealed record→part index after every complete part exists."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/'COMPLETE.json').exists():
        return read_manifest(root)
    with (root/'merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'COMPLETE.json').exists():
            return read_manifest(root)
        attempt = root/('index.sqlite3.incomplete.'+uuid.uuid4().hex)
        db = sqlite3.connect(attempt);db.execute('CREATE TABLE records(record_id TEXT PRIMARY KEY,scene_token TEXT NOT NULL,part TEXT NOT NULL)')
        db.execute('CREATE INDEX scene_idx ON records(scene_token)')
        identity, scenes, parts, accounting = None, set(), {}, {}
        try:
            for path in part_roots:
                path = Path(path); m = read_manifest(path)
                if m['kind'] != 'part' or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Evidence merge requires registered child parts')
                if identity is not None and m['identity'] != identity:
                    raise ValueError('Evidence part identities differ')
                identity = m['identity']
                if scenes & set(m['scenes']):
                    raise ValueError('Evidence parts overlap scenes')
                scenes.update(m['scenes'])
                if file_sha(path/'index.sqlite3') != m['index_sha256']:
                    raise ValueError('Evidence part checksum mismatch')
                name = str(path.relative_to(root))
                parts[name] = dict(manifest_sha256=file_sha(path/'MANIFEST.json'), records=m['counts']['records'])
                for key, value in m.get('accounting', {}).items():
                    accounting[key] = accounting.get(key, 0) + value
                source = sqlite3.connect(f'file:{path / "index.sqlite3"}?mode=ro&immutable=1', uri=True)
                try:
                    db.executemany('INSERT INTO records VALUES (?,?,?)',
                                   ((record_id, token, name) for record_id, token in source.execute('SELECT record_id,scene_token FROM records')))
                finally:
                    source.close()
            if scenes != set(expected_scenes) or len(expected_scenes) != len(scenes):
                raise ValueError('Complete evidence must cover exactly the sealed TRAIN scene set')
            db.commit(); count = db.execute('SELECT COUNT(*) FROM records').fetchone()[0]; db.close(); db = None
            os.replace(attempt, root/'index.sqlite3')
            m = dict(schema=SCHEMA, status='COMPLETE', kind='merged', identity=identity,
                     counts=dict(scenes=len(scenes), records=count), accounting=accounting,
                     scenes=list(expected_scenes), parts=parts,
                     index_sha256=file_sha(root/'index.sqlite3'))
            m['sha256'] = fingerprint(m);atomic_json(root/'MANIFEST.json', m)
            atomic_json(root/'COMPLETE.json', dict(status='COMPLETE', bank_hash=m['sha256'],
                                                  manifest_sha256=file_sha(root/'MANIFEST.json')))
            return m
        finally:
            if db is not None:
                db.close()
