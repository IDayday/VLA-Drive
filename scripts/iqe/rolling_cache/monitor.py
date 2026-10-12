#!/usr/bin/env python3
"""Read-only, bounded rolling-cache supply/consumption observer (stdlib only).

No signals, cache writes, GPU allocation, sampler changes or automatic launches.
Completed optimizer-update log events are consumption; prepared_step is not.
Shard receipts measure committed production even after pool garbage collection.
This metadata observer does not replace the consumer's record SHA256 checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


class Metadata:
    """No tensor/payload reads; unchanged metadata is cached by file identity."""
    def __init__(self, max_bytes=128 * 1024**2, max_entries=500000):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.cache = OrderedDict()
        self.reset()

    def reset(self):
        self.bytes_read = self.entries = 0

    def entry(self):
        self.entries += 1
        require(self.entries <= self.max_entries, "METADATA_ENTRY_BUDGET_EXHAUSTED")

    def read(self, path, *, sealed=False, sha256=None):
        path = Path(path)
        self.entry()
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = str(path)
        old = self.cache.pop(key, None)
        if old is None or old[0] != stamp:
            require(self.bytes_read + stat.st_size <= self.max_bytes, "METADATA_BYTE_BUDGET_EXHAUSTED")
            raw = path.read_bytes()
            self.bytes_read += len(raw)
            later = path.stat()
            require(stamp == (later.st_dev, later.st_ino, later.st_size, later.st_mtime_ns,
                              later.st_ctime_ns), "METADATA_CHANGED_DURING_READ")
            old = (stamp, json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())
        self.cache[key] = old
        while len(self.cache) > 8192:
            self.cache.popitem(last=False)
        value = old[1]
        if sealed:
            require(value.get("content_hash") == digest({k: v for k, v in value.items()
                                                       if k != "content_hash"}), "METADATA_CHECKSUM_MISMATCH: " + key)
        if sha256 is not None:
            require(old[2] == sha256, "METADATA_FILE_HASH_MISMATCH: " + key)
        return value


def read_updates(path, checkpoint_step, request_time, limit, max_bytes=16 * 1024**2):
    """Bounded tail, handling partial final writes and appended resume segments."""
    path = Path(path)
    stat = path.stat()
    if stat.st_mtime + 1 < request_time:
        return [], {"status": "LOG_PREDATES_ATTEMPT", "mtime": stat.st_mtime}
    with path.open("rb") as stream:
        start = max(0, stat.st_size - max_bytes)
        stream.seek(start)
        data = stream.read(max_bytes)
    lines = data.splitlines(keepends=True)
    if start and lines:
        lines = lines[1:]
    rows = []
    for raw in lines:
        if not raw.endswith(b"\n"):
            continue  # A currently-written event is not a completed update.
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue  # Library diagnostics are interleaved with JSON events.
        if not isinstance(row, dict) or row.get("event") != "update":
            continue
        step = row.get("optimizer_step")
        require(type(step) is int, "INVALID_UPDATE_STEP")
        if rows and step <= rows[-1]["optimizer_step"]:
            rows = []  # New resume segment; never retain an older larger step.
        if step <= checkpoint_step:
            continue
        require(not rows or step == rows[-1]["optimizer_step"] + 1, "NONCONTIGUOUS_UPDATE_LOG")
        for key in ("seconds", "global_batch"):
            require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and row[key] > 0,
                    "INVALID_UPDATE_" + key)
        require(math.isfinite(row.get("wait_seconds", 0)) and row.get("wait_seconds", 0) >= 0,
                "INVALID_UPDATE_WAIT")
        require(all(math.isfinite(v) for v in row.get("losses", {}).values()) and
                math.isfinite(row.get("gradient_norm", 0)), "NONFINITE_TRAINING_UPDATE")
        rows.append(row)
    return rows[-limit:], {"status": "OBSERVED_LOG", "mtime": stat.st_mtime,
                          "tail_bytes": len(data), "truncated_tail": bool(start)}


def file_stamp(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class Inventory:
    """Compact receipt ledger independent of the small parsed-JSON LRU.

    Every remembered immutable receipt is stat-checked every observation; any
    changed stamp requires its original SHA256, including an atomic replacement.
    Archived generation directories are not walked/read again unless their
    stamp changes. New generations on checkpoint rewind count as new production.
    No historical payload, full-dataset index, ACCESS_STAGING or FAST file is read.
    A late cold start initializes over multiple byte-budgeted observations.
    """
    def __init__(self):
        self.generations = {}
        self.data_dirs = {}
        self.windows = []
        self.windows_stamp = None

    def discover(self, root, metadata):
        base = root / "windows"
        stamp = file_stamp(base)
        if stamp != self.windows_stamp:
            self.windows = [p for p in base.iterdir() if p.is_dir() and not p.is_symlink()]
            self.windows_stamp = stamp
        for window in self.windows:
            metadata.entry()
            data = window / "data"
            if not data.exists():
                continue
            require(not data.is_symlink(), "SYMLINK_POOL_DATA")
            stamp = file_stamp(data)
            if self.data_dirs.get(str(data)) == stamp:
                continue
            for generation in data.iterdir():
                metadata.entry()
                require(not generation.is_symlink(), "SYMLINK_POOL_GENERATION")
                if generation.is_dir():
                    self.generations.setdefault(str(generation), {"stamp": None, "archived": False,
                        "checks": {}, "shards": {}, "files": {}, "blocks": 0, "fast": []})
            self.data_dirs[str(data)] = stamp

    @staticmethod
    def immutable_name(name):
        return name == "COMPLETE.json" or name.endswith((".bin.COMPLETE.json", ".bin.INTENT.json"))

    def pin_receipt(self, path, state, metadata, visual_identity):
        row = metadata.read(path)
        stamp, _, raw_hash = metadata.cache[str(path)]
        if str(path) in state["checks"]:
            require(raw_hash == state["checks"][str(path)][1], "IMMUTABLE_RECEIPT_CHANGED: " + str(path))
        if path.name.endswith(".bin.COMPLETE.json"):
            require(row["identity_hash"] == visual_identity, "FOREIGN_SHARD_IDENTITY")
            require(type(row["bytes"]) is int and row["bytes"] >= 0 and len(row["sha256"]) == 64,
                    "INVALID_SHARD_RECEIPT")
            require(sum(v["length"] for v in row["records"].values()) == row["bytes"],
                    "INVALID_SHARD_RECORD_BYTES")
            state["shards"][str(path).removesuffix(".COMPLETE.json")] = {
                "sha256": row["sha256"], "bytes": row["bytes"], "record_count": len(row["records"])}
        state["checks"][str(path)] = (stamp, raw_hash)

    def scan(self, root, metadata, visual_identity):
        self.discover(root, metadata)
        complete = True
        for name, state in self.generations.items():
            generation = Path(name)
            metadata.entry()
            require(generation.is_dir() and not generation.is_symlink(), "HISTORICAL_GENERATION_REMOVED")
            # File content edits do not necessarily change the directory stamp.
            for path, (stamp, _) in list(state["checks"].items()):
                metadata.entry()
                require(Path(path).exists(), "IMMUTABLE_RECEIPT_REMOVED: " + path)
                if file_stamp(path) != stamp:
                    self.pin_receipt(Path(path), state, metadata, visual_identity)
            directory_stamp = file_stamp(generation)
            if state["archived"] and state["stamp"] == directory_stamp:
                continue
            try:
                paths = list(generation.iterdir())
                for path in paths:
                    metadata.entry()
                    require(not path.is_symlink(), "SYMLINK_POOL_ARTIFACT")
                    if self.immutable_name(path.name) and str(path) not in state["checks"]:
                        require(not state["archived"], "ARCHIVED_GENERATION_RECEIPT_ADDED")
                        self.pin_receipt(path, state, metadata, visual_identity)
                files, blocks, fast = {}, 0, []
                for path in paths:
                    if path.name.endswith((".bin", ".partial")):
                        try:
                            s = path.stat()
                        except FileNotFoundError:
                            continue  # Garbage collection may remove payloads.
                        files[str(path)] = s.st_size
                        blocks += s.st_blocks * 512
                    elif path.name.startswith("FAST_") and path.name.endswith(".json"):
                        # FAST metadata is disposable timing, never a counter.
                        if any(p.name.endswith((".bin", ".partial")) for p in paths):
                            fast.append({"path": str(path), **metadata.read(path)})
                # Evicted immutable generations keep their compact counters.
                state.update(files=files, blocks=blocks, fast=fast, stamp=directory_stamp,
                             archived=not files and str(generation / "COMPLETE.json") in state["checks"])
            except ValueError as exc:
                if str(exc) not in {"METADATA_BYTE_BUDGET_EXHAUSTED", "METADATA_ENTRY_BUDGET_EXHAUSTED"}:
                    raise
                complete = False
                break  # Retain initialization progress; never increase budget.
        files, shards, fast, blocks = {}, {}, [], 0
        for state in self.generations.values():
            files.update(state["files"]); shards.update(state["shards"])
            fast.extend(state["fast"]); blocks += state["blocks"]
        return {"scan_complete": complete, "generations": len(self.generations),
            "archived_generations": sum(s["archived"] for s in self.generations.values()),
            "pinned_receipts": sum(len(s["checks"]) for s in self.generations.values()),
            "resident_bytes": sum(files.values()), "allocated_blocks_bytes": blocks,
            "committed_bytes_all_generations": sum(s["bytes"] for s in shards.values()),
            "committed_records_all_generations": sum(s["record_count"] for s in shards.values()),
            "files": files, "shards": shards, "fast": fast}


def scan_pool(root, metadata, visual_identity):
    """One-shot helper; long-running Monitor retains its Inventory ledger."""
    return Inventory().scan(root, metadata, visual_identity)


def window_state(root, number, session, metadata, inventory):
    folder = root / "windows" / f"{number:06d}"
    if not (folder / "PLAN.json").exists():
        return {"number": number, "ready": False, "reason": "PLAN_NOT_YET_CREATED"}
    plan = metadata.read(folder / "PLAN.json", sealed=True)
    require(plan["session_hash"] == session["content_hash"] and plan["number"] == number,
            "FOREIGN_WINDOW_PLAN")
    expected_first = session["parent_step"] + number * session["window_updates"] + 1
    require(plan["first_step"] == expected_first and
            plan["last_step"] == expected_first + len(plan["batches"]) - 1, "WINDOW_STEP_MISMATCH")
    result = {"number": number, "ready": False, "plan": plan, "new_keys": None,
              "new_scene_ids": None, "reason": "INCOMPLETE_WINDOW"}
    ready_path = folder / "READY.json"
    if ready_path.exists() and not (folder / "EVICTING.json").exists():
        ready = metadata.read(ready_path, sealed=True)
        require(ready["session_hash"] == session["content_hash"] and ready["number"] == number and
                ready["plan_hash"] == plan["content_hash"] and ready["status"] == "WINDOW_COMPLETE",
                "STALE_WINDOW_READY")
        require(set(ready["refs"]) == {r["scene_id"] for r in plan["rows"]}, "INCOMPLETE_READY_SCENES")
        allowed = {v["root"]: v["index_sha256"] for v in session["reuse_sources"]}
        own = ready.get("owned_root")
        if own:
            require(Path(own).resolve().parent == (folder / "data").resolve(), "FOREIGN_OWNED_ROOT")
            allowed[own] = ready["sources"].get(own)
        require(set(ready["refs"].values()) == set(ready["sources"]), "READY_SOURCE_SET_MISMATCH")
        for source, sha in ready["sources"].items():
            require(source in allowed and sha == allowed[source], "FOREIGN_READY_SOURCE")
            complete = metadata.read(Path(source) / "COMPLETE.json")
            require(complete["status"] == "COMPLETE" and complete["index_sha256"] == sha,
                    "SOURCE_COMPLETION_MISMATCH")
        if own:
            index = metadata.read(Path(own) / "INDEX.json", sha256=ready["sources"][own])
            for name, spec in index["shards"].items():
                require(Path(name).name == name, "UNSAFE_SHARD_NAME")
                path = str(Path(own) / name)
                if inventory["files"].get(path) != spec["bytes"]:
                    result["reason"] = "OWNED_PAYLOAD_MISSING_OR_CONCURRENT_COMMIT"
                    return result
                receipt = inventory["shards"].get(path)
                require(receipt and receipt["sha256"] == spec["sha256"], "SHARD_RECEIPT_MISMATCH")
            result["new_keys"] = {sid: value["key"] for sid, value in index["access"].items()}
        else:
            result["new_keys"] = {}
        result.update(ready=True, reason="CHECKED_COMPLETE_METADATA",
                      new_scene_ids={s for s, source in ready["refs"].items() if source == own})
    else:
        staging = folder / "BUILDING.json"
        if staging.exists():
            generation = metadata.read(staging, sealed=True)
            require(generation["plan_hash"] == plan["content_hash"], "STALE_STAGING_PLAN")
            own = Path(generation["root"])
            require(own.resolve().parent == (folder / "data").resolve(), "FOREIGN_STAGING_ROOT")
            if (own / "PLAN.json").exists():
                native = metadata.read(own / "PLAN.json")
                require(native["plan_hash"] == digest({k: v for k, v in native.items() if k != "plan_hash"})
                        and native["rolling_window_hash"] == plan["content_hash"], "STALE_NATIVE_PLAN")
                result["new_scene_ids"] = {r["scene_id"] for r in native["rows"]}
    return result


# Only reads /proc. Running it over SSH never signals, attaches to, or launches
# a training process. Full argv + start ticks + cwd protect against PID reuse.
REMOTE_PROBE = r'''
import json,os,pathlib,socket,sys
d=json.loads(sys.argv[1]);out={'host':socket.gethostname(),'processes':[],'deleted_files':{}}
for c in d['children']:
 p=pathlib.Path('/proc')/str(c['pid']);r={'pid':c['pid'],'worker':c.get('worker')}
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split()
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  cwd=os.readlink(p/'cwd')
  stdout=os.readlink(p/'fd/1')
  r.update(state=st[0],start_ticks=st[19],cwd=cwd,stdout=stdout,gpu=c.get('gpu'),
   identity_match=(str(c['ticks'])==st[19] and argv==c['argv'] and (not d.get('cwd') or cwd==d['cwd'])
    and (not d.get('expected_log') or stdout==d['expected_log'])))
  r['alive']=st[0] not in ['Z','X']
 except (FileNotFoundError,ProcessLookupError):r.update(alive=False,identity_match=False)
 out['processes'].append(r)
# Include consumer DDP children when the receipt identifies a torchrun parent.
known={str(c['pid']) for c in d['children']};processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split();processes.append((p,st[1]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for _ in range(4):
 for p,parent in processes:
  if parent in known:known.add(p.name)
for pid in known:
 try:
  for f in (pathlib.Path('/proc')/pid/'fd').iterdir():
   try:
    target=os.readlink(f)
    if target.startswith(d['root']+'/') and target.endswith(' (deleted)'):
     st=f.stat();out['deleted_files'][target]={'bytes':st.st_size,'blocks':st.st_blocks*512}
   except (FileNotFoundError,PermissionError):pass
 except (FileNotFoundError,PermissionError):pass
print(json.dumps(out))
'''


def inspect_host(receipt, root, timeout):
    host = receipt["host"].removesuffix("-worker-0")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host), "INVALID_INSPECTION_HOST")
    children = receipt.get("children", [receipt])
    require(1 <= len(children) <= 16, "PROCESS_INSPECTION_BUDGET")
    for child in children:
        require(type(child["pid"]) is int and child["pid"] > 0 and str(child["ticks"]).isdigit(),
                "INVALID_PID_IDENTITY")
        require("iqe.rolling_cache.cli" in child["argv"], "UNRELATED_PROCESS_RECEIPT")
    for child in children:
        if "produce" in child["argv"]:
            argv = child["argv"]
            require("--root" in argv and Path(argv[argv.index("--root") + 1]).resolve() == root,
                    "PRODUCER_RECEIPT_POOL_MISMATCH")
    payload = json.dumps({"root": str(root), "cwd": receipt.get("cwd"), "children": children,
                          "expected_log": receipt.get("_expected_log")})
    args = [sys.executable, "-c", REMOTE_PROBE, payload]
    if socket.gethostname().removesuffix("-worker-0") != host:
        args = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(timeout, 10)}", host,
                shlex.join(["python3", "-c", REMOTE_PROBE, payload])]
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=True)
        value = json.loads(result.stdout)
        require(value["host"].removesuffix("-worker-0") == host, "INSPECTION_HOST_IDENTITY_MISMATCH")
        return {"status": "INSPECTED", **value}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"host": host, "status": "NOT_VERIFIED", "reason": str(exc), "processes": [], "deleted_files": {}}


def summarize(session, request, updates, log_info, windows, inventory, *, now, previous,
              disk_free, inspections, completed=False, low_watermark=20., stall_seconds=90.):
    consumed = updates[-1]["optimizer_step"] if updates else request["checkpoint_step"]
    require(session["parent_step"] <= consumed <= session["max_optimizer_steps"], "CONSUMED_STEP_OUT_OF_RANGE")
    rows = [w for w in windows if w.get("plan", {}).get("last_step", 0) > consumed]
    continuous = 0
    for w in rows:
        if not w["ready"] or w["plan"]["first_step"] > consumed + continuous + 1:
            break
        continuous += w["plan"]["last_step"] - max(consumed, w["plan"]["first_step"] - 1)
    durations = [r["seconds"] for r in updates]
    median = statistics.median(durations) if durations else None
    # Demand uses non-wait median, so supply stalls cannot disguise fast demand.
    compute_median = statistics.median([max(.001, r["seconds"] - r.get("wait_seconds", 0))
                                         for r in updates]) if updates else None
    batches = scenes = new_scenes = new_records = known_batches = 0
    upper_bound = False
    for w in rows:
        remaining = [b for i, b in enumerate(w["plan"]["batches"])
                     if w["plan"]["first_step"] + i > consumed]
        batches += len(remaining)
        scenes += sum(len(b["scene_ids"]) for b in remaining)
        if w.get("new_scene_ids") is None:
            continue
        known_batches += len(remaining)
        ids = {sid for b in remaining for sid in b["scene_ids"] if sid in w["new_scene_ids"]}
        new_scenes += len(ids)
        keys = w.get("new_keys")
        new_records += len({keys[sid] for sid in ids}) if keys is not None else len(ids)
        upper_bound |= keys is None
    needed_rate = (new_records / known_batches / compute_median) if known_batches and compute_median else None
    supply = None
    if previous:
        elapsed = now - previous["timestamp_unix"]
        delta = inventory["committed_records_all_generations"] - previous["production"]["committed_records_all_generations"]
        delta_bytes = inventory["committed_bytes_all_generations"] - previous["production"]["committed_bytes_all_generations"]
        require(elapsed > 0 and delta >= 0 and delta_bytes >= 0, "PRODUCTION_EVIDENCE_REMOVED_OR_TIME_REVERSED")
        supply = {"interval_seconds": elapsed, "records_per_second": delta / elapsed,
                  "bytes_per_second": delta_bytes / elapsed, "records": delta, "bytes": delta_bytes}
    high_water = bool(rows) and all(w["ready"] for w in rows) and batches >= min(
        session["max_optimizer_steps"] - consumed,
        session["lookahead"] * session["window_updates"] - (consumed - session["parent_step"]) % session["window_updates"])
    deleted = {}
    for host in inspections:
        deleted.update(host.get("deleted_files", {}))
    deleted_bytes = sum(v["bytes"] for v in deleted.values())
    alerts = []
    if disk_free < session["reserve_bytes"]:
        alerts.append("SHARED_STORAGE_RESERVE_BREACHED")
    if inventory["resident_bytes"] + deleted_bytes > session["capacity_bytes"]:
        alerts.append("POOL_CAPACITY_EXCEEDED")
    if any(h["status"] != "INSPECTED" for h in inspections):
        alerts.append("PROCESS_INSPECTION_NOT_VERIFIED")
    producers = [p for h in inspections if h.get("role") == "producer" for p in h["processes"]]
    if any(not p.get("alive") or not p.get("identity_match") for p in producers):
        alerts.append("PRODUCER_DEAD_OR_IDENTITY_MISMATCH")
    consumers = [p for h in inspections if h.get("role") == "consumer" for p in h["processes"]]
    if not completed and any(not p.get("alive") or not p.get("identity_match") for p in consumers):
        alerts.append("CONSUMER_DEAD_OR_IDENTITY_MISMATCH")
    if not completed and now - log_info["mtime"] > stall_seconds:
        alerts.append("CONSUMER_NO_RECENT_COMPLETED_UPDATE")
    if not completed and continuous == 0:
        alerts.append("NEXT_WINDOW_NOT_READY")
    seconds = continuous * compute_median if compute_median is not None else None
    if not completed and seconds is not None and seconds < low_watermark:
        alerts.append("LOW_CONTIGUOUS_BUFFER")
    if supply and supply["interval_seconds"] >= 10 and needed_rate is not None and not high_water and not completed:
        if supply["records_per_second"] < needed_rate:
            alerts.append("SUPPLY_BELOW_ESTIMATED_DEMAND")
    return {"schema": "iqe_rolling_monitor_v1", "timestamp_unix": now,
        "status": "ALERT" if alerts else "OK", "alerts": alerts,
        "session_hash": session["content_hash"], "attempt": request["attempt"],
        "consumer": {"completed_optimizer_step": consumed, "source": "completed_update_log_or_checkpoint",
            "prepared_is_consumption": False, "completed_run": completed, "recent_updates": len(updates),
            "mean_update_seconds_including_wait": statistics.mean(durations) if durations else None,
            "p50_update_seconds": median, "p50_seconds_minus_exposed_batch_wait": compute_median,
            "exposed_batch_wait_seconds": sum(r.get("wait_seconds", 0) for r in updates),
            "samples_per_second_including_wait": sum(r["global_batch"] for r in updates) / sum(durations) if durations else None,
            "seconds_since_log_write": max(0, now - log_info["mtime"]), "log": log_info},
        "buffer": {"continuous_ready_updates": continuous, "estimated_seconds_at_nonwait_p50": seconds,
            "high_watermark_reached": high_water, "windows": [{k: w[k] for k in ("number", "ready", "reason")} for w in windows]},
        "demand": {"planned_updates": batches, "planned_scene_exposures": scenes,
            "known_new_record_updates": known_batches, "new_unique_scenes": new_scenes,
            "new_payload_records": new_records, "record_count_is_upper_bound": upper_bound,
            "records_per_second_at_nonwait_p50": needed_rate,
            "bytes_per_second_upper_bound": needed_rate * session["record_max_bytes"] if needed_rate is not None else None},
        "production": {k: inventory[k] for k in ("committed_records_all_generations", "committed_bytes_all_generations")},
        "supply_measured_between_snapshots": supply,
        "storage": {"resident_payload_bytes": inventory["resident_bytes"],
            "allocated_payload_blocks_bytes": inventory["allocated_blocks_bytes"], "deleted_open_bytes": deleted_bytes,
            "deleted_open_files": deleted, "pool_capacity_bytes": session["capacity_bytes"],
            "shared_available_bytes": disk_free, "protected_reserve_bytes": session["reserve_bytes"]},
        "processes": inspections,
        "scheduling": {"monitor_actions": "NONE_READ_ONLY", "producer_policy": "existing REQUEST-driven bounded lookahead",
            "expected_producer_state": "HIGH_WATER_IDLE" if high_water else "FILL_REQUESTED_WINDOWS",
            "prepared_cursor_purpose": "producer lookahead only; never resume or completed training progress",
            "operator_advice": ["Inspect FAILED producer logs and host/PID/start-tick identities before retrying the same bounded producer command.",
                "Preserve committed shards; do not change the sampler, precision, global batch or learning-rate timeline.",
                "If reserve is breached, request operator action; this monitor never deletes checkpoints or stops jobs."]}}


class Monitor:
    def __init__(self, args, probe=inspect_host):
        self.args, self.probe = args, probe
        self.metadata = Metadata(args.metadata_mib * 1024**2, args.max_metadata_entries)
        self.inventory = Inventory()
        self.previous = None
        self.session_hash = None

    def snapshot(self, now=None):
        a, m = self.args, self.metadata
        now = time.time() if now is None else now
        m.reset()
        root = Path(a.root).resolve()
        session = m.read(root / "SESSION.json", sealed=True)
        require(session["schema"] == "iqe_remote_rolling_session_v1", "FOREIGN_POOL_SESSION")
        require(self.session_hash in (None, session["content_hash"]), "POOL_SESSION_CHANGED")
        self.session_hash = session["content_hash"]
        request = m.read(root / "REQUEST.json", sealed=True)
        require(request["session_hash"] == session["content_hash"], "FOREIGN_CONSUMER_REQUEST")
        qualification = m.read(session["qualification"], sha256=session["qualification_sha256"])
        launch = m.read(a.consumer_launch)
        require(launch["identity"]["production_cache_runtime"]["session_hash"] == session["content_hash"],
                "CONSUMER_LAUNCH_SESSION_MISMATCH")
        updates, log_info = read_updates(a.consumer_log, request["checkpoint_step"],
                                         (root / "REQUEST.json").stat().st_mtime, a.history_updates)
        consumed = updates[-1]["optimizer_step"] if updates else request["checkpoint_step"]
        first = (consumed - session["parent_step"]) // session["window_updates"]
        require(first >= 0, "CONSUMER_PRECEDES_POOL")
        inventory = self.inventory.scan(root, m, qualification["visual_identity_hash"])
        if not inventory["scan_complete"]:
            return {"timestamp_unix": now, "status": "NOT_VERIFIED", "read_only": True,
                "error": "METADATA_INITIALIZATION_IN_PROGRESS_WITHIN_BUDGET", "metadata_read_bytes": m.bytes_read,
                "pinned_receipts": inventory["pinned_receipts"], "generations": inventory["generations"],
                "production_rate": None, "storage_totals": None}
        total = math.ceil((session["max_optimizer_steps"] - session["parent_step"]) / session["window_updates"])
        windows = [window_state(root, n, session, m, inventory)
                   for n in range(first, min(total, first + session["lookahead"]))]
        receipts = [("producer", m.read(p)) for p in a.producer_receipt]
        if a.consumer_receipt:
            c = m.read(a.consumer_receipt)
            if "host" not in c:
                require(a.consumer_host, "CONSUMER_RECEIPT_REQUIRES_EXPLICIT_HOST")
                c = {**c, "host": a.consumer_host}
            receipts.append(("consumer", {**c, "_expected_log": str(Path(a.consumer_log).resolve())}))
        with ThreadPoolExecutor(max_workers=len(receipts)) as executor:
            futures = [(role, executor.submit(self.probe, r, root, a.ssh_timeout)) for role, r in receipts]
            inspections = [{"role": role, **future.result()} for role, future in futures]
        result_path = Path(a.consumer_launch).parent / "RESULT.json"
        completed = False
        if result_path.exists():
            result = m.read(result_path)
            completed = (result.get("status") == "COMPLETE" and result.get("final_step") == consumed and
                         result.get("identity", {}).get("production_cache_runtime", {}).get("session_hash") == session["content_hash"]
                         and result_path.stat().st_mtime >= (root / "REQUEST.json").stat().st_mtime)
        require(m.read(root / "REQUEST.json", sealed=True) == request, "CONSUMER_ATTEMPT_CHANGED_RETRY")
        fs = os.statvfs(root)
        snapshot = summarize(session, request, updates, log_info, windows, inventory, now=now,
            previous=self.previous, disk_free=fs.f_bavail * fs.f_frsize, inspections=inspections,
            completed=completed, low_watermark=a.low_watermark_seconds, stall_seconds=a.stall_seconds)
        snapshot["metadata_read_bytes"] = m.bytes_read
        snapshot["metadata_entries"] = m.entries
        snapshot["inventory_ledger"] = {k: inventory[k] for k in
            ("generations", "archived_generations", "pinned_receipts", "scan_complete")}
        snapshot["fast_timings"] = sorted(inventory["fast"], key=lambda r: r.get("updated_unix", 0))[-16:]
        snapshot["fast_timing_caveat"] = "inclusive worker-local sections; never sum parallel workers as critical-path time"
        self.previous = snapshot
        return snapshot


def atomic_json(path, value):
    path = Path(path)
    value = {**value, "content_hash": digest(value)}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".monitor-", mode="w", encoding="utf-8", delete=False) as out:
        temporary = Path(out.name)
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.write("\n"); out.flush(); os.fsync(out.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "consumer-log", "consumer-launch", "consumer-receipt"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--consumer-host", help="required only if consumer PID receipt lacks host")
    p.add_argument("--producer-receipt", action="append", required=True, help="repeat for the two producer hosts")
    p.add_argument("--interval", type=float, required=True)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--history-updates", type=int, default=100)
    p.add_argument("--ssh-timeout", type=int, default=10)
    p.add_argument("--low-watermark-seconds", type=float, default=20)
    p.add_argument("--stall-seconds", type=float, default=90)
    p.add_argument("--metadata-mib", type=int, default=128)
    p.add_argument("--max-metadata-entries", type=int, default=500000)
    p.add_argument("--max-history-mib", type=int, default=256)
    p.add_argument("--once", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require(5 <= args.interval <= 300 and 1 <= args.max_seconds <= 86400, "EXPLICIT_MONITOR_TIME_BUDGET")
        require(1 <= args.history_updates <= 2000 and 1 <= args.ssh_timeout <= 30, "INVALID_OBSERVATION_BUDGET")
        require(1 <= len(args.producer_receipt) <= 2 and 1 <= args.metadata_mib <= 512 and
                1 <= args.max_metadata_entries <= 1000000 and 1 <= args.max_history_mib <= 1024,
                "INVALID_METADATA_OR_HOST_BUDGET")
        require(math.isfinite(args.low_watermark_seconds) and args.low_watermark_seconds >= 0 and
                math.isfinite(args.stall_seconds) and args.stall_seconds > 0, "INVALID_ALERT_THRESHOLDS")
        root, output = Path(args.root).resolve(), Path(args.output).resolve()
        require(output != root and not output.is_relative_to(root), "MONITOR_OUTPUT_MUST_BE_OUTSIDE_POOL")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "monitor.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(output / "RESOLVED.json", {"args": vars(args), "started_unix": time.time(), "read_only": True})
            monitor = Monitor(args)
            deadline = time.monotonic() + args.max_seconds
            status = 0
            while True:
                began = time.monotonic()
                try:
                    row = monitor.snapshot()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row = {"timestamp_unix": time.time(), "status": "NOT_VERIFIED", "error": str(exc), "read_only": True}
                atomic_json(output / "STATUS.json", row)
                history = output / "history.jsonl"
                encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                require((history.stat().st_size if history.exists() else 0) + len(encoded) <= args.max_history_mib * 1024**2,
                        "MONITOR_HISTORY_BUDGET_EXHAUSTED")
                with history.open("ab") as stream:
                    stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps({k: row[k] for k in ("timestamp_unix", "status", "alerts", "error") if k in row}), flush=True)
                status = 0 if row["status"] == "OK" else 2
                if args.once or time.monotonic() >= deadline:
                    return status
                time.sleep(max(0, min(args.interval - (time.monotonic() - began), deadline - time.monotonic())))
                if time.monotonic() >= deadline:
                    return status
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc), "read_only": True}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
