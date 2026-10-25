"""Immutable physical8x3 candidates and true canonical-PDM labels.

The SQLite index addresses individual records in bounded JSONL shards; a training
reader never deserializes a complete shard. Teacher/actor hidden states are not
part of this cache. A missing/partial bank is an error, never an all-zero label.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import torch

from .post_action import METRIC_NAMES, OFFICIAL_METRIC_NAMES, choose_aux_scenes


SCHEMA = "qwen_world_postcot_physical_candidate_bank_v1"
COORDINATE_VERSION = "rear_axle_ego_x_forward_y_left_yaw_radian_8x0.5s_v1"
CODEC_VERSION = "qwen_sourceflow_absolute_minmax_8x3_v1"


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def trajectory_hash(trajectory: Any) -> str:
    array = np.asarray(trajectory)
    if array.shape != (8, 3) or not np.isfinite(array).all():
        raise ValueError("Candidate must be a finite physical8x3 trajectory")
    fp32 = np.ascontiguousarray(array, dtype="<f4")
    if not np.array_equal(fp32.astype(array.dtype), array):
        raise ValueError("Coordinates were not exact FP32; rescore their stored FP32 values")
    return hashlib.sha256(fp32.tobytes()).hexdigest()


def observation_fingerprint(row: Mapping[str, Any], source_manifest_sha256: str,
                            image_sha256: Mapping[str, str | None]) -> str:
    return fingerprint(dict(schema="world_postcot_current_observation_v1",
        source_manifest_sha256=source_manifest_sha256, token=row["token"],
        log=row["log"], timestamp=int(row["timestamp"]), proprio=row["proprio"],
        navigation=row["navigation"], current_image_sha256=dict(image_sha256)))


@dataclass(frozen=True)
class CandidateRecord:
    scene_id: str
    source_log: str
    split: str
    timestamp_us: int
    observation_hash: str
    trajectory_fp32: list[list[float]]
    trajectory_sha256: str
    raw_factor_labels: dict[str, float | None]
    valid_masks: dict[str, bool]
    raw_total_score: float
    score_protocol_sha256: str
    evaluator_revision: dict[str, Any]
    metric_version: str
    context_sha256: str
    candidate_source: str
    actor_checkpoint_sha256: str | None
    sampling: dict[str, Any]
    coordinate_version: str = COORDINATE_VERSION
    normalization_version: str = CODEC_VERSION
    cache_version: str = SCHEMA
    status: str = "valid_scored"
    source_parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "CandidateRecord":
        if self.split not in ("TRAIN", "DEV"):
            raise ValueError("Training candidate banks cannot read NAVTEST/final_test")
        if (self.status != "valid_scored" or self.cache_version != SCHEMA or
                self.coordinate_version != COORDINATE_VERSION or
                self.normalization_version != CODEC_VERSION):
            raise ValueError("Candidate validity/coordinate/codec contract mismatch")
        if not all((self.scene_id, self.source_log, self.observation_hash,
                    self.score_protocol_sha256, self.context_sha256, self.candidate_source)):
            raise ValueError("Candidate is missing scene, observation or scorer provenance")
        if self.candidate_source not in ('logged_gt', 'coherent_variant', 'native_qwen_il',
                                          'native_qwen_rl', 'current_student_snapshot'):
            raise ValueError("External scorer/teacher candidates are outside this bank contract")
        if self.timestamp_us <= 0 or not self.evaluator_revision:
            raise ValueError("Candidate requires the real timestamp and evaluator revision")
        if trajectory_hash(self.trajectory_fp32) != self.trajectory_sha256:
            raise ValueError("Candidate coordinate checksum changed")
        if set(self.raw_factor_labels) != set(OFFICIAL_METRIC_NAMES) or set(self.valid_masks) != set(OFFICIAL_METRIC_NAMES):
            raise ValueError("All six named official factors and masks must be retained")
        for name in OFFICIAL_METRIC_NAMES:
            value, valid = self.raw_factor_labels[name], self.valid_masks[name]
            if type(valid) is not bool:
                raise ValueError("Metric validity is a bool, not a numeric target")
            if valid and (value is None or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Invalid canonical metric is not an ordinary zero label")
            if value is not None and not math.isfinite(value):
                raise ValueError("Nonfinite infrastructure feedback belongs in failed attempts, not this bank")
        if not math.isfinite(self.raw_total_score) or not 0 <= self.raw_total_score <= 1:
            raise ValueError("Invalid canonical total score")
        return self

    @property
    def record_id(self) -> str:
        return fingerprint([self.scene_id, self.observation_hash, self.trajectory_sha256,
                            self.score_protocol_sha256, self.context_sha256])

    @classmethod
    def from_feedback(cls, *, row: Mapping[str, Any], observation_hash: str,
                      feedback: Mapping[str, Any], evaluator_revision: dict,
                      source: str, actor_checkpoint_sha256: str | None = None,
                      sampling: dict | None = None, source_parameters: dict | None = None,
                      split: str = "TRAIN") -> "CandidateRecord":
        if feedback.get("status") != "OK" or feedback["token"] != row["token"] or feedback["log"] != row["log"]:
            raise ValueError("Scoring feedback does not match the exact requested scene")
        trajectory = np.asarray(feedback["requested_trajectory"])
        key = trajectory_hash(trajectory)
        metrics = feedback["metrics"]
        return cls(scene_id=row["token"], source_log=row["log"], split=split,
            timestamp_us=int(row["timestamp"]), observation_hash=observation_hash,
            trajectory_fp32=trajectory.astype(np.float32).tolist(), trajectory_sha256=key,
            raw_factor_labels={name: float(metrics[name]) for name in OFFICIAL_METRIC_NAMES},
            valid_masks={name: True for name in OFFICIAL_METRIC_NAMES},
            raw_total_score=float(metrics["score"]),
            score_protocol_sha256=feedback["protocol_sha256"], evaluator_revision=evaluator_revision,
            metric_version=feedback["protocol"], context_sha256=feedback["cache"]["sha256"],
            candidate_source=source, actor_checkpoint_sha256=actor_checkpoint_sha256,
            sampling=sampling or {}, source_parameters=source_parameters or {}).validate()


def _signature(record: CandidateRecord) -> tuple:
    return (record.scene_id, record.observation_hash, record.trajectory_sha256,
            record.score_protocol_sha256, record.context_sha256,
            record.raw_factor_labels, record.valid_masks, record.raw_total_score)


class CandidateBankWriter:
    """Single-owner shard transaction; many hosts build disjoint part banks.

    File bytes are fsynced before the SQLite transaction commits. A restart
    retains uncommitted tails as diagnostic evidence and truncates only those
    unpublished tails. A complete version can never be appended to or replaced.
    """
    def __init__(self, root: str | Path, identity: dict, scene_ids: Sequence[str], *,
                 minimum_candidates: int = 2, maximum_candidates: int = 32,
                 shard_bytes: int = 64 * 1024 * 1024, resume: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/"WRITER.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.root/"COMPLETE.json").exists():
            raise FileExistsError("Bank is immutable and complete; create a new version")
        if len(set(scene_ids)) != len(scene_ids) or not scene_ids:
            raise ValueError("Expected scenes must be a nonempty fixed unique list")
        if minimum_candidates < 2 or maximum_candidates < minimum_candidates or shard_bytes < 1024:
            raise ValueError("Invalid candidate/shard bounds")
        self.identity = dict(identity)
        self.scenes = set(scene_ids)
        self.state = dict(schema=SCHEMA, identity=self.identity, scene_ids=list(scene_ids),
            minimum_candidates=minimum_candidates, maximum_candidates=maximum_candidates,
            shard_bytes=shard_bytes)
        state_path = self.root/"BUILD_IDENTITY.json"
        if state_path.exists():
            if not resume or json.loads(state_path.read_text()) != self.state:
                raise ValueError("Explicit resume requires identical data/protocol/build identity")
        else:
            atomic_json(state_path, self.state)
        self.shard_bytes, self.maximum = shard_bytes, maximum_candidates
        (self.root/"shards").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root/"index.sqlite3")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY,scene_id TEXT NOT NULL,trajectory_hash TEXT NOT NULL,shard TEXT NOT NULL,offset INTEGER NOT NULL,length INTEGER NOT NULL,checksum TEXT NOT NULL,UNIQUE(scene_id,trajectory_hash))")
        self.db.execute("CREATE INDEX IF NOT EXISTS scenes ON records(scene_id)")
        self.db.commit()
        self.handle = None
        self.current = None
        self._recover_tails()

    def _recover_tails(self):
        for path in sorted((self.root/"shards").glob("*.jsonl")):
            relative = str(path.relative_to(self.root))
            committed = self.db.execute("SELECT MAX(offset+length) FROM records WHERE shard=?", (relative,)).fetchone()[0] or 0
            size = path.stat().st_size
            if size < committed:
                raise ValueError("Committed candidate bytes are missing")
            if size > committed:
                diagnostic = self.root/("UNCOMMITTED_"+path.name+"_"+uuid.uuid4().hex)
                with path.open("rb") as source, diagnostic.open("xb") as target:
                    source.seek(committed)
                    while data := source.read(1024 * 1024):
                        target.write(data)
                with path.open("r+b") as stream:
                    stream.truncate(committed)
            self.current = path

    def _open(self, length):
        if self.current is None or self.current.stat().st_size + length > self.shard_bytes:
            if self.handle is not None:
                self.handle.close()
            index = len(list((self.root/"shards").glob("*.jsonl")))
            self.current = self.root/"shards"/f"{index:06d}.jsonl"
            self.handle = self.current.open("ab")
        elif self.handle is None:
            self.handle = self.current.open("ab")
        return self.handle

    def append_many(self, records: Iterable[CandidateRecord]) -> dict[str, int]:
        added = reused = 0
        dirty = set()
        try:
            for record in records:
                record.validate()
                if (record.scene_id not in self.scenes or record.split != self.identity["split"] or
                        record.score_protocol_sha256 != self.identity["score_protocol_sha256"]):
                    raise ValueError("Candidate crosses fixed scene/split/scoring dependencies")
                previous = self.db.execute("SELECT record_id,shard,offset,length FROM records WHERE scene_id=? AND trajectory_hash=?",
                                           (record.scene_id, record.trajectory_sha256)).fetchone()
                if previous:
                    with (self.root/previous[1]).open("rb") as stream:
                        stream.seek(previous[2]); payload = stream.read(previous[3])
                    old = CandidateRecord(**json.loads(payload)).validate()
                    if previous[0] != record.record_id or _signature(old) != _signature(record):
                        raise ValueError("Same coordinates have conflicting observations or true labels")
                    reused += 1
                    continue
                count = self.db.execute("SELECT COUNT(*) FROM records WHERE scene_id=?", (record.scene_id,)).fetchone()[0]
                if count >= self.maximum:
                    raise ValueError("More candidates than the locked per-scene budget")
                payload = (json.dumps(asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
                stream = self._open(len(payload))
                offset = stream.tell(); stream.write(payload); stream.flush()
                dirty.add(self.current)
                relative = str(self.current.relative_to(self.root))
                self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)", (record.record_id,
                    record.scene_id, record.trajectory_sha256, relative, offset, len(payload),
                    hashlib.sha256(payload).hexdigest()))
                added += 1
            for path in dirty:
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            self.db.commit()
        except BaseException:
            self.db.rollback()
            if self.handle is not None:
                self.handle.close(); self.handle = None
            self._recover_tails()
            raise
        return dict(added=added, reused=reused)

    def scene_count(self, token: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM records WHERE scene_id=?", (token,)).fetchone()[0]

    def finalize(self) -> dict:
        counts = dict(self.db.execute("SELECT scene_id,COUNT(*) FROM records GROUP BY scene_id"))
        if set(counts) != self.scenes or any(value < self.state["minimum_candidates"] or value > self.maximum for value in counts.values()):
            raise ValueError("Incomplete candidate coverage; cannot publish COMPLETE")
        if self.handle is not None:
            self.handle.flush(); os.fsync(self.handle.fileno()); self.handle.close(); self.handle = None
        self.db.commit(); self.db.close(); self.db = None
        return _publish_manifest(self.root, self.identity, self.state["scene_ids"], counts)

    def close(self):
        if self.handle is not None:
            self.handle.close(); self.handle = None
        if self.db is not None:
            self.db.close(); self.db = None
        if self.lock is not None:
            self.lock.close(); self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _publish_manifest(root: Path, identity: dict, scene_ids: list[str], counts: dict) -> dict:
    db = sqlite3.connect(f"file:{root/'index.sqlite3'}?mode=ro", uri=True)
    shards = {}
    for (name,) in db.execute("SELECT DISTINCT shard FROM records"):
        path = root/name
        shards[name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    record_count = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    db.close()
    body = dict(schema=SCHEMA, status="COMPLETE", identity=identity,
        scene_ids=scene_ids, scene_count=len(scene_ids), record_count=record_count,
        candidate_counts=counts, shards=shards, index_sha256=file_sha(root/"index.sqlite3"))
    body["sha256"] = fingerprint(body)
    if (root/"MANIFEST.json").exists():
        if json.loads((root/"MANIFEST.json").read_text()) != body:
            raise FileExistsError("Never overwrite a previous immutable bank manifest")
    else:
        atomic_json(root/"MANIFEST.json", body)
    atomic_json(root/"COMPLETE.json", dict(status="COMPLETE", manifest_sha256=file_sha(root/"MANIFEST.json"), bank_hash=body["sha256"]))
    return body


@dataclass
class CandidatePairBatch:
    trajectory: torch.Tensor
    labels: torch.Tensor
    valid: torch.Tensor
    pair_mask: torch.Tensor
    record_ids: list[list[str]]
    scene_ids: list[str]
    bank_hash: str

    def to(self, device: torch.device | str) -> "CandidatePairBatch":
        return CandidatePairBatch(self.trajectory.to(device), self.labels.to(device),
            self.valid.to(device), self.pair_mask.to(device), self.record_ids,
            self.scene_ids, self.bank_hash)


class CandidateBank:
    def __init__(self, manifest: str | Path, *, expected_identity: Mapping[str, Any] | None = None,
                 allowed_splits: Sequence[str] = ("TRAIN",), seed: int = 42,
                 max_open_shards: int = 8, max_cached_scenes: int = 64):
        manifest = Path(manifest)
        if manifest.is_dir():
            manifest = manifest/"MANIFEST.json"
        self.root = manifest.parent
        self.manifest = json.loads(manifest.read_text())
        body = dict(self.manifest); expected = body.pop("sha256")
        complete = json.loads((self.root/"COMPLETE.json").read_text())
        if (self.manifest["schema"] != SCHEMA or self.manifest["status"] != "COMPLETE" or
                fingerprint(body) != expected or complete["bank_hash"] != expected or
                complete["manifest_sha256"] != file_sha(manifest)):
            raise ValueError("Partial or altered candidate bank")
        identity = self.manifest["identity"]
        if identity["split"] not in allowed_splits:
            raise ValueError("Candidate bank data role cannot enter this training call")
        if expected_identity and any(identity.get(k) != v for k, v in expected_identity.items()):
            raise ValueError("Candidate bank observation/codec/protocol identity mismatch")
        if file_sha(self.root/"index.sqlite3") != self.manifest["index_sha256"]:
            raise ValueError("Candidate index checksum mismatch")
        self.db = sqlite3.connect(f"file:{self.root/'index.sqlite3'}?mode=ro&immutable=1", uri=True)
        self.manifest_path, self.bank_hash = str(manifest.resolve()), expected
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.max_open, self.max_cached = max_open_shards, max_cached_scenes
        if self.max_open < 1 or self.max_cached < 0:
            raise ValueError("Reader resources must be bounded")
        self.handles, self.verified, self.cache = OrderedDict(), {}, OrderedDict()
        self.sampled_scenes = self.sampled_records = 0

    @staticmethod
    def _stat(path_or_fd):
        value = os.fstat(path_or_fd) if isinstance(path_or_fd, int) else Path(path_or_fd).stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def _handle(self, shard):
        path = self.root/shard
        if not path.resolve().is_relative_to(self.root.resolve()) or shard not in self.manifest["shards"]:
            raise ValueError("Unregistered candidate shard path")
        if shard in self.handles:
            handle = self.handles.pop(shard)
            if self._stat(handle.fileno()) != self.verified[shard]:
                raise ValueError("Previously verified immutable candidate shard changed")
        else:
            current = self._stat(path)
            if shard in self.verified and self.verified[shard] != current:
                raise ValueError("Immutable candidate shard was replaced or modified")
            if shard not in self.verified:
                expected = self.manifest["shards"][shard]
                if current[2] != expected["bytes"] or file_sha(path) != expected["sha256"]:
                    raise ValueError("Candidate shard checksum mismatch")
                self.verified[shard] = current
            handle = path.open("rb")
        self.handles[shard] = handle
        while len(self.handles) > self.max_open:
            self.handles.popitem(last=False)[1].close()
        return handle

    def records(self, token: str, *, observation_hash: str | None = None) -> list[CandidateRecord]:
        if token in self.cache:
            records, shards = self.cache.pop(token)
            for shard in shards:
                self._handle(shard)
            self.cache[token] = (records, shards)
        else:
            rows = self.db.execute("SELECT shard,offset,length,checksum,record_id FROM records WHERE scene_id=? ORDER BY record_id", (token,)).fetchall()
            if len(rows) < 2:
                raise KeyError("Missing scene/two-candidate coverage in the locked bank: "+token)
            records = []
            for shard, offset, length, checksum, record_id in rows:
                handle = self._handle(shard); handle.seek(offset); payload = handle.read(length)
                if hashlib.sha256(payload).hexdigest() != checksum:
                    raise ValueError("Candidate record checksum mismatch")
                record = CandidateRecord(**json.loads(payload)).validate()
                if record.record_id != record_id or record.scene_id != token:
                    raise ValueError("Candidate index/scene join mismatch")
                records.append(record)
            if self.max_cached:
                self.cache[token] = (records, tuple(sorted({row[0] for row in rows})))
                while len(self.cache) > self.max_cached:
                    self.cache.popitem(last=False)
        if observation_hash is not None and any(r.observation_hash != observation_hash for r in records):
            raise ValueError("Candidate observation hash differs from the actual current input")
        return records

    def sample_pairs(self, scene_ids: Sequence[str], *, generator: torch.Generator | None = None,
                     observation_hashes: Sequence[str] | None = None) -> CandidatePairBatch:
        generator = self.generator if generator is None else generator
        if generator.device.type != "cpu":
            raise ValueError("Candidate sampler uses a separately restorable CPU RNG")
        if observation_hashes is not None and len(observation_hashes) != len(scene_ids):
            raise ValueError("Observation IDs must align with requested scene IDs")
        selected, masks = [], []
        for position, token in enumerate(scene_ids):
            rows = self.records(token, observation_hash=None if observation_hashes is None else observation_hashes[position])
            eligible = []
            for i in range(len(rows)):
                for j in range(i+1, len(rows)):
                    mask = [rows[i].valid_masks[name] and rows[j].valid_masks[name]
                            and rows[i].raw_factor_labels[name] in (0., 1.)
                            and rows[j].raw_factor_labels[name] in (0., 1.)
                            and rows[i].raw_factor_labels[name] != rows[j].raw_factor_labels[name]
                            for name in OFFICIAL_METRIC_NAMES[:2]]
                    if any(mask):
                        a, b = np.asarray(rows[i].trajectory_fp32), np.asarray(rows[j].trajectory_fp32)
                        distance = float(np.linalg.norm(a[:, :2]-b[:, :2], axis=1).mean())
                        eligible.append((distance, i, j, mask))
            if eligible:
                # Prefer geometrically close boundary pairs while exposing up
                # to four alternatives instead of repeating one pair forever.
                eligible.sort(key=lambda row: (row[0], rows[row[1]].record_id, rows[row[2]].record_id))
                index = int(torch.randint(min(4, len(eligible)), (), generator=generator))
                _, i, j, mask = eligible[index]
            else:
                indices = torch.randperm(len(rows), generator=generator)[:2].tolist()
                i, j = indices; mask = [False, False]
            selected.append([rows[i], rows[j]]); masks.append(mask)
        shape = (len(selected), 2, 8, 3)
        trajectories = torch.tensor([[r.trajectory_fp32 for r in pair] for pair in selected], dtype=torch.float32).reshape(shape)
        labels = torch.tensor([[[r.raw_factor_labels[name] if r.raw_factor_labels[name] is not None else float('nan')
                                for name in OFFICIAL_METRIC_NAMES] for r in pair] for pair in selected], dtype=torch.float32).reshape(len(selected), 2, 6)
        valid = torch.tensor([[[r.valid_masks[name] for name in OFFICIAL_METRIC_NAMES]
                               for r in pair] for pair in selected], dtype=torch.bool).reshape(len(selected), 2, 6)
        self.sampled_scenes += len(selected); self.sampled_records += 2*len(selected)
        return CandidatePairBatch(trajectories.detach(), labels.detach(), valid.detach(),
            torch.tensor(masks, dtype=torch.bool).reshape(len(selected), 2),
            [[r.record_id for r in pair] for pair in selected], list(scene_ids), self.bank_hash)

    def auxiliary_batch(self, scene_ids: Sequence[str], *, fraction: float = 0.5,
                        observation_hashes: Sequence[str] | None = None):
        indices = choose_aux_scenes(scene_ids, fraction, generator=self.generator)
        positions = indices.tolist()
        return indices, self.sample_pairs([scene_ids[i] for i in positions],
            observation_hashes=None if observation_hashes is None else [observation_hashes[i] for i in positions])

    def state_dict(self):
        return dict(schema=SCHEMA, manifest_path=self.manifest_path, bank_hash=self.bank_hash,
            generator_state=self.generator.get_state(), sampled_scenes=self.sampled_scenes,
            sampled_records=self.sampled_records)

    def load_state_dict(self, state):
        if state['schema'] != SCHEMA or state['bank_hash'] != self.bank_hash or state['manifest_path'] != self.manifest_path:
            raise ValueError("Resume must restore its exact candidate-bank version")
        self.generator.set_state(state['generator_state'])
        self.sampled_scenes, self.sampled_records = state['sampled_scenes'], state['sampled_records']

    def close(self):
        self.db.close()
        for handle in self.handles.values():
            handle.close()
        self.handles.clear(); self.cache.clear()


def merge_bank_parts(output: str | Path, parts: Sequence[str | Path], *,
                     identity: dict, scene_ids: Sequence[str]) -> dict:
    """Publish one global index over complete disjoint shard producers."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/'MERGE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'COMPLETE.json').exists():
            bank = CandidateBank(output, allowed_splits=(identity['split'],), expected_identity=identity)
            try:
                if bank.manifest['scene_ids'] != list(scene_ids):
                    raise ValueError('Completed merge population changed')
                return bank.manifest
            finally:
                bank.close()
        temporary = output/('index.building.'+uuid.uuid4().hex)
        destination = sqlite3.connect(temporary)
        destination.execute('CREATE TABLE records(record_id TEXT PRIMARY KEY,scene_id TEXT NOT NULL,trajectory_hash TEXT NOT NULL,shard TEXT NOT NULL,offset INTEGER NOT NULL,length INTEGER NOT NULL,checksum TEXT NOT NULL,UNIQUE(scene_id,trajectory_hash))')
        destination.execute('CREATE INDEX scenes ON records(scene_id)')
        counts, seen = {}, set()
        try:
            for part in parts:
                part = Path(part).resolve()
                if not part.is_relative_to(output.resolve()):
                    raise ValueError('Part shards must be within the published bank root')
                reader = CandidateBank(part, expected_identity=identity, allowed_splits=(identity['split'],), max_cached_scenes=0)
                try:
                    for name in reader.manifest['shards']:
                        reader._handle(name)
                    tokens = set(reader.manifest['scene_ids'])
                    if tokens & seen:
                        raise ValueError('Duplicate scene ownership across candidate producers')
                    seen |= tokens; counts.update(reader.manifest['candidate_counts'])
                    prefix = part.relative_to(output.resolve())
                    rows = reader.db.execute('SELECT record_id,scene_id,trajectory_hash,shard,offset,length,checksum FROM records')
                    for row in rows:
                        destination.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?)', (*row[:3], str(prefix/row[3]), *row[4:]))
                finally:
                    reader.close()
            if seen != set(scene_ids):
                raise ValueError('Candidate parts do not cover the fixed complete population')
            destination.commit(); destination.close()
            os.replace(temporary, output/'index.sqlite3')
            return _publish_manifest(output, identity, list(scene_ids), counts)
        except BaseException:
            destination.close()
            # Retain the failed building index as evidence; no COMPLETE marker.
            raise
