"""Monitoring fixtures are metadata tests, never real-training evidence."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/iqe/rolling_cache/monitor.py"
spec = importlib.util.spec_from_file_location("rolling_observer", SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def write(path, value, sealed=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if sealed:
        value = {**value, "content_hash": monitor.digest(value)}
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return value


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(step, seconds=1., wait=0.):
    return {"event": "update", "optimizer_step": step, "seconds": seconds,
            "wait_seconds": wait, "global_batch": 4, "losses": {"IL": .5}, "gradient_norm": 1.}


def fixture(tmp_path):
    root = tmp_path / "pool"; qual = tmp_path / "qualified.json"
    write(qual, {"visual_identity_hash": "v" * 64})
    permanent = tmp_path / "permanent"
    write(permanent / "COMPLETE.json", {"status": "COMPLETE", "index_sha256": "permanent"})
    session = write(root / "SESSION.json", {"schema": "iqe_remote_rolling_session_v1", "parent_step": 10,
        "max_optimizer_steps": 20, "window_updates": 2, "lookahead": 2, "world_size": 2,
        "record_max_bytes": 10, "capacity_bytes": 1000, "reserve_bytes": 100,
        "qualification": str(qual), "qualification_sha256": hash_file(qual),
        "reuse_sources": [{"root": str(permanent), "index_sha256": "permanent"}]}, sealed=True)
    request = write(root / "REQUEST.json", {"session_hash": session["content_hash"], "checkpoint_step": 10,
        "attempt": "first", "world_size": 2}, sealed=True)
    for n in range(2):
        folder = root / "windows" / f"{n:06d}"; own = folder / "data" / "generation"
        ids = [f"s{n}_{i}" for i in range(8)]
        plan = write(folder / "PLAN.json", {"session_hash": session["content_hash"], "number": n,
            "first_step": 11 + n * 2, "last_step": 12 + n * 2,
            "batches": [{"scene_ids": ids[:4]}, {"scene_ids": ids[4:]}],
            "rows": [{"scene_id": sid} for sid in ids]}, sealed=True)
        own.mkdir(parents=True, exist_ok=True); (own / "shard.bin").write_bytes(b"12345678")
        records = {sid: {"offset": i * 2, "length": 2, "shard": "shard.bin"} for i, sid in enumerate(ids[:4])}
        sha = hash_file(own / "shard.bin")
        write(own / "shard.bin.COMPLETE.json", {"identity_hash": "v" * 64, "bytes": 8, "sha256": sha,
                                              "records": records})
        write(own / "INDEX.json", {"access": {sid: {"key": sid} for sid in ids[:4]},
                                 "shards": {"shard.bin": {"bytes": 8, "sha256": sha}}})
        index_hash = hash_file(own / "INDEX.json")
        write(own / "COMPLETE.json", {"status": "COMPLETE", "index_sha256": index_hash})
        write(folder / "READY.json", {"session_hash": session["content_hash"], "number": n,
            "status": "WINDOW_COMPLETE", "plan_hash": plan["content_hash"], "owned_root": str(own),
            "refs": {sid: str(own if i < 4 else permanent) for i, sid in enumerate(ids)},
            "sources": {str(own): index_hash, str(permanent): "permanent"}}, sealed=True)
    launch = tmp_path / "consumer" / "LAUNCH.json"
    write(launch, {"identity": {"production_cache_runtime": {"session_hash": session["content_hash"]}}})
    log = tmp_path / "consumer.log"; log.write_text("", encoding="utf-8")
    receipt = tmp_path / "producer.json"; write(receipt, {"host": "example", "children": []})
    consumer = tmp_path / "consumer.json"; write(consumer, {"host": "example", "pid": 999})
    args = SimpleNamespace(root=str(root), consumer_launch=str(launch), consumer_log=str(log),
        consumer_receipt=str(consumer), consumer_host=None, producer_receipt=[str(receipt)],
        metadata_mib=8, max_metadata_entries=10000, history_updates=100, ssh_timeout=1,
        low_watermark_seconds=0., stall_seconds=90.)
    def probe(receipt, root, timeout):
        return {"status": "INSPECTED", "host": "example", "processes": [
            {"pid": 999, "alive": True, "identity_match": True}], "deleted_files": {}}
    return monitor.Monitor(args, probe), root, log, session, request


def test_prepared_is_not_consumed_and_no_pool_mutation(tmp_path):
    obj, root, log, _, request = fixture(tmp_path)
    write(root / "progress" / request["attempt"] / "rank_0.json", {"prepared_step": 18})
    before = {str(p): hash_file(p) for p in root.rglob("*") if p.is_file()}
    result = obj.snapshot()
    assert result["consumer"]["completed_optimizer_step"] == 10
    assert result["consumer"]["prepared_is_consumption"] is False
    assert result["buffer"]["continuous_ready_updates"] == 4
    assert before == {str(p): hash_file(p) for p in root.rglob("*") if p.is_file()}


def test_real_consumption_fresh_demand_excludes_permanent_reuse(tmp_path):
    obj, _, log, _, _ = fixture(tmp_path)
    log.write_text(json.dumps(event(11, 11., 10.)) + "\n", encoding="utf-8")
    result = obj.snapshot()
    assert result["consumer"]["mean_update_seconds_including_wait"] == 11.
    assert result["consumer"]["p50_seconds_minus_exposed_batch_wait"] == 1.
    assert result["buffer"]["continuous_ready_updates"] == 3
    # Remaining batch of first window is all permanent; next window is half new.
    assert result["demand"]["planned_scene_exposures"] == 12
    assert result["demand"]["new_payload_records"] == 4
    assert result["demand"]["records_per_second_at_nonwait_p50"] == pytest.approx(4 / 3)


def test_do_not_jump_over_missing_window_or_trust_complete_with_missing_payload(tmp_path):
    obj, root, _, _, _ = fixture(tmp_path)
    (root / "windows/000000/data/generation/shard.bin").unlink()
    result = obj.snapshot()
    assert result["buffer"]["continuous_ready_updates"] == 0
    assert "NEXT_WINDOW_NOT_READY" in result["alerts"]
    assert result["buffer"]["windows"][1]["ready"]
    assert result["production"]["committed_records_all_generations"] == 8


def test_zero_supply_is_expected_at_high_water_not_false_alarm(tmp_path):
    obj, _, log, _, _ = fixture(tmp_path)
    log.write_text(json.dumps(event(11)) + "\n", encoding="utf-8")
    first = obj.snapshot(now=100.)
    second = obj.snapshot(now=130.)
    assert second["supply_measured_between_snapshots"]["records_per_second"] == 0
    assert second["scheduling"]["expected_producer_state"] == "HIGH_WATER_IDLE"
    assert "SUPPLY_BELOW_ESTIMATED_DEMAND" not in second["alerts"]


def test_waiting_stall_reserve_capacity_and_dead_producer_alert(tmp_path):
    obj, root, log, session, request = fixture(tmp_path)
    inv = monitor.scan_pool(root, obj.metadata, "v" * 64)
    windows = [monitor.window_state(root, i, session, obj.metadata, inv) for i in range(2)]
    windows[1]["ready"] = False
    prior = monitor.summarize(session, request, [event(11)], {"mtime": 0}, windows, inv, now=20,
        previous=None, disk_free=0, inspections=[], low_watermark=10.)
    bad = [{"role": "producer", "status": "INSPECTED", "processes": [{"alive": False, "identity_match": False}],
            "deleted_files": {"/pool/old.bin (deleted)": {"bytes": 2000, "blocks": 2000}}}]
    result = monitor.summarize(session, request, [event(11)], {"mtime": 0}, windows, inv, now=150,
        previous=prior, disk_free=0, inspections=bad, low_watermark=10.)
    assert {"SHARED_STORAGE_RESERVE_BREACHED", "POOL_CAPACITY_EXCEEDED", "LOW_CONTIGUOUS_BUFFER",
            "CONSUMER_NO_RECENT_COMPLETED_UPDATE", "PRODUCER_DEAD_OR_IDENTITY_MISMATCH",
            "SUPPLY_BELOW_ESTIMATED_DEMAND"} <= set(result["alerts"])


def test_metadata_hash_input_change_and_budget_reject(tmp_path):
    obj, root, _, _, _ = fixture(tmp_path)
    ready = root / "windows/000000/READY.json"
    data = json.loads(ready.read_text()); data["number"] = 1; write(ready, data)
    with pytest.raises(ValueError, match="CHECKSUM"):
        obj.snapshot()
    with pytest.raises(ValueError, match="BYTE_BUDGET"):
        monitor.Metadata(max_bytes=1).read(root / "SESSION.json")


def test_ascii_locale_cannot_break_utf8_metadata(tmp_path):
    p = tmp_path / "unicode.json"; write(p, {"instruction": "不改变训练"})
    script = ("import importlib.util; s=importlib.util.spec_from_file_location('m'," + repr(str(SCRIPT)) +
              ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
              "assert m.Metadata().read(" + repr(str(p)) + ")[\"instruction\"] == " + ascii("不改变训练"))
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"})


def test_partial_log_stale_log_and_resume_segment(tmp_path):
    p = tmp_path / "log"; lines = [event(11), event(12), event(11), event(12)]
    p.write_text("\n".join(json.dumps(v) for v in lines) + "\n" + json.dumps(event(13))[:20])
    rows, _ = monitor.read_updates(p, 10, 0, 100)
    assert [r["optimizer_step"] for r in rows] == [11, 12]
    rows, info = monitor.read_updates(p, 10, p.stat().st_mtime + 3, 100)
    assert rows == [] and info["status"] == "LOG_PREDATES_ATTEMPT"
    p.write_text(json.dumps(event(11, float("nan"))) + "\n")
    with pytest.raises(ValueError, match="INVALID_UPDATE_seconds"):
        monitor.read_updates(p, 10, 0, 100)


def test_completed_consumer_does_not_false_alarm_when_process_exits(tmp_path):
    obj, _, log, session, _ = fixture(tmp_path)
    log.write_text(json.dumps(event(11)) + "\n")
    write(Path(obj.args.consumer_launch).parent / "RESULT.json", {"status": "COMPLETE", "final_step": 11,
          "identity": {"production_cache_runtime": {"session_hash": session["content_hash"]}}})
    obj.probe = lambda receipt, root, timeout: {"status": "INSPECTED", "host": "example", "processes": [
        {"alive": "children" in receipt, "identity_match": "children" in receipt}], "deleted_files": {}}
    result = obj.snapshot(now=log.stat().st_mtime + 1000)
    assert result["consumer"]["completed_run"]
    assert "CONSUMER_DEAD_OR_IDENTITY_MISMATCH" not in result["alerts"]
    assert "CONSUMER_NO_RECENT_COMPLETED_UPDATE" not in result["alerts"]


def test_actual_proc_start_tick_command_identity_no_signals(tmp_path):
    p = Path("/proc") / str(os.getpid())
    argv = [v.decode() for v in (p / "cmdline").read_bytes().split(bytes([0])) if v]
    ticks = (p / "stat").read_text().rsplit(")", 1)[1].split()[19]
    request = {"root": str(tmp_path), "cwd": os.getcwd(), "children": [
        {"pid": os.getpid(), "ticks": ticks, "argv": argv},
        {"pid": os.getpid(), "ticks": str(int(ticks) + 1), "argv": argv}]}
    result = subprocess.run([sys.executable, "-c", monitor.REMOTE_PROBE, json.dumps(request)],
                            check=True, capture_output=True, text=True)
    rows = json.loads(result.stdout)["processes"]
    assert rows[0]["alive"] and rows[0]["identity_match"]
    assert rows[1]["alive"] and not rows[1]["identity_match"]


def test_cli_requires_budgets_and_forbids_pool_output(tmp_path):
    with pytest.raises(SystemExit):
        monitor.main([])
    obj, root, log, _, _ = fixture(tmp_path)
    args = ["--root", str(root), "--output", str(root / "bad"), "--consumer-log", str(log),
            "--consumer-launch", obj.args.consumer_launch, "--consumer-receipt", obj.args.consumer_receipt,
            "--producer-receipt", obj.args.producer_receipt[0], "--interval", "5", "--max-seconds", "1", "--once"]
    assert monitor.main(args) == 1
    assert not (root / "bad").exists()


def test_snapshot_outputs_atomic_checksum_and_append_history(tmp_path, monkeypatch):
    obj, root, log, _, _ = fixture(tmp_path)
    monkeypatch.setattr(monitor, "Monitor", lambda args: obj)
    output = tmp_path / "observations"
    args = ["--root", str(root), "--output", str(output), "--consumer-log", str(log),
            "--consumer-launch", obj.args.consumer_launch, "--consumer-receipt", obj.args.consumer_receipt,
            "--producer-receipt", obj.args.producer_receipt[0], "--interval", "5", "--max-seconds", "1", "--once"]
    assert monitor.main(args) == 0
    assert monitor.main(args) == 0
    status = monitor.Metadata().read(output / "STATUS.json", sealed=True)
    assert status["scheduling"]["monitor_actions"] == "NONE_READ_ONLY"
    assert len((output / "history.jsonl").read_text().splitlines()) == 2


def archive_window(root, number):
    folder = root / "windows" / f"{number:06d}"
    (folder / "READY.json").unlink()
    for p in folder.glob("data/*/*.bin"):
        p.unlink()
    write(folder / "EVICTED_fixture.json", {"status": "EVICTED"})


def test_archived_receipt_counters_survive_lru_eviction_without_reparse(tmp_path):
    _, root, _, _, _ = fixture(tmp_path)
    inventory, meta = monitor.Inventory(), monitor.Metadata()
    before = inventory.scan(root, meta, "v" * 64)
    archive_window(root, 0); archive_window(root, 1)
    meta.reset()
    gc = inventory.scan(root, meta, "v" * 64)
    assert gc["resident_bytes"] == 0 and gc["archived_generations"] == 2
    assert gc["committed_records_all_generations"] == before["committed_records_all_generations"] == 8
    meta.cache.clear(); meta.reset()
    after = inventory.scan(root, meta, "v" * 64)
    assert after["committed_records_all_generations"] == 8
    assert meta.bytes_read == 0  # Independent of 8192-entry JSON LRU, any history length.


@pytest.mark.parametrize("change", ["edit", "atomic_replace", "remove", "append_receipt"])
def test_any_archived_receipt_change_is_rejected(tmp_path, change):
    _, root, _, _, _ = fixture(tmp_path)
    inventory, meta = monitor.Inventory(), monitor.Metadata()
    inventory.scan(root, meta, "v" * 64)
    archive_window(root, 0)
    inventory.scan(root, meta, "v" * 64)
    p = root / "windows/000000/data/generation/shard.bin.COMPLETE.json"
    row = json.loads(p.read_text())
    if change == "remove":
        p.unlink()
    elif change == "append_receipt":
        write(p.parent / "extra.bin.COMPLETE.json", row)
    else:
        row["sha256"] = "f" * 64
        if change == "atomic_replace":
            other = p.with_name("replacement.json"); write(other, row); other.replace(p)
        else:
            write(p, row)
    meta.cache.clear(); meta.reset()
    with pytest.raises(ValueError, match="IMMUTABLE_RECEIPT|ARCHIVED_GENERATION_RECEIPT"):
        inventory.scan(root, meta, "v" * 64)


def test_rebuild_uses_new_generation_and_counts_new_compute_once(tmp_path):
    import shutil
    _, root, _, _, _ = fixture(tmp_path)
    inventory, meta = monitor.Inventory(), monitor.Metadata()
    old = root / "windows/000000/data/generation"
    before = inventory.scan(root, meta, "v" * 64)
    saved = old.joinpath("shard.bin").read_bytes()
    archive_window(root, 0); inventory.scan(root, meta, "v" * 64)
    new = old.with_name("rebuilt_generation")
    shutil.copytree(old, new); (new / "shard.bin").write_bytes(saved)
    meta.reset(); after = inventory.scan(root, meta, "v" * 64)
    assert after["committed_records_all_generations"] == before["committed_records_all_generations"] + 4
    meta.reset(); again = inventory.scan(root, meta, "v" * 64)
    assert again["committed_records_all_generations"] == after["committed_records_all_generations"]


def test_cold_history_initialization_is_bounded_and_resumable(tmp_path):
    _, root, _, _, _ = fixture(tmp_path)
    archive_window(root, 0); archive_window(root, 1)
    max_receipt = max(p.stat().st_size for p in root.glob("windows/*/data/*/*COMPLETE.json"))
    meta = monitor.Metadata(max_bytes=max_receipt + 10)
    inventory = monitor.Inventory()
    incomplete = 0
    for _ in range(8):
        meta.reset()
        result = inventory.scan(root, meta, "v" * 64)
        assert meta.bytes_read <= meta.max_bytes
        if result["scan_complete"]:
            break
        incomplete += 1
    assert incomplete and result["scan_complete"]
    assert result["committed_records_all_generations"] == 8


def test_full_run_history_projection_with_small_metadata_budget(tmp_path, monkeypatch):
    # Exercise the actual scan over 3342 archived ledgers. Filesystem metadata
    # is simulated; the mutation/GC tests above exercise actual temporary files.
    inventory, meta = monitor.Inventory(), monitor.Metadata(max_entries=500000)
    generations = 3342
    stamp = (1, 2, 3, 4, 5)
    for n in range(generations):
        p = tmp_path / str(n)
        inventory.generations[str(p)] = {"stamp": stamp, "archived": True,
            # Worst case: all 512 scenes miss, ceil(512/12)=43 shard pairs.
            "checks": {str(p / f"receipt_{k}.json"): (stamp, "sha") for k in range(87)},
            "files": {}, "blocks": 0, "fast": [],
            "shards": {str(p / "shard.bin"): {"bytes": 240, "record_count": 12, "sha256": "sha"}}}
    monkeypatch.setattr(inventory, "discover", lambda *_: None)
    monkeypatch.setattr(monitor, "file_stamp", lambda _: stamp)
    monkeypatch.setattr(Path, "exists", lambda _: True)
    monkeypatch.setattr(Path, "is_dir", lambda _: True)
    monkeypatch.setattr(Path, "is_symlink", lambda _: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Archived receipts must not be re-read/walked after LRU eviction")
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(meta, "read", forbidden)
    result = inventory.scan(tmp_path, meta, "identity")
    assert result["committed_records_all_generations"] == generations * 12
    assert result["archived_generations"] == generations
    assert meta.entries == 294096 < meta.max_entries and meta.bytes_read == 0
