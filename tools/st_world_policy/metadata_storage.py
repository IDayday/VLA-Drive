"""Bounded metadata-only storage admission using the existing campaign ledger.

No model/data reader, scheduler, resource dispatch or automatic policy migration.
The explicit migration requires Lead-owned live receipts. All filesystem state,
including failed partials, stays charged. The thin runner invokes only the two metadata APIs in this same process.
"""
from contextlib import contextmanager
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import importlib
import threading
import stat
import time

from tools.st_world_policy import cache_storage_budget as b

MIB = 1 << 20
PACKAGE_MAX = 4 * b.GIB
CONTROL_LIMIT = 16 * MIB
REG_SCHEMA = "st_metadata_build_registration_v1"
PACKAGE_SCHEMA = "st_metadata_storage_package_v1"
MIGRATION_SCHEMA = "st_metadata_storage_migration_v1"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                b._fail("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: b._fail("nonfinite JSON constant"))


def _version(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def pinned_bytes(path, expected_sha):
    """Hash and parse exactly the same bounded bytes; reject replacement races."""
    path = b._path(path)
    b._hash(expected_sha, "receipt SHA")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            b._fail("receipt must be a private regular file")
        raw = stream.read(CONTROL_LIMIT + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > CONTROL_LIMIT or _version(before) != _version(after) or _version(after) != _version(path.stat()):
        b._fail("receipt size/version changed")
    if _sha(raw) != expected_sha:
        b._fail("receipt SHA mismatch: " + str(path))
    return raw


def pinned_json(path, expected_sha):
    raw = pinned_bytes(path, expected_sha)
    return _json(raw), raw


def tree_inventory(root, *, max_depth=4, max_entries=10000, patterns=None, directory_patterns=None):
    """Bounded stat-only metadata census, including directory allocation.

    No symlinks, shared hardlinks, mount escapes or unknown names. Logical size
    is also charged so sparse output cannot hide the future allocation bound.
    """
    root = b._path(root)
    if not root.exists():
        return {"root": str(root), "entries": [], "charged_bytes": 0}
    device = root.stat().st_dev
    rows, total = [], 0
    compiled = [re.compile(p) for p in patterns] if patterns is not None else None
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        info = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        if info.st_dev != device or stat.S_ISLNK(info.st_mode):
            b._fail("metadata link or mount escape")
        if stat.S_ISDIR(info.st_mode):
            kind = "directory"
            if path != root and directory_patterns is not None and not any(re.fullmatch(p, relative) for p in directory_patterns):
                b._fail("unregistered metadata directory: " + relative)
            if depth > max_depth:
                b._fail("metadata tree depth exceeds registration")
            with os.scandir(path) as children:
                for child in children:
                    stack.append((Path(child.path), depth + 1))
                    if len(stack) + len(rows) > max_entries:
                        b._fail("metadata entry count exceeds registration")
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            kind = "file"
            if compiled is not None and not any(p.fullmatch(relative) for p in compiled):
                b._fail("unregistered metadata filename: " + relative)
        else:
            b._fail("metadata contains nonregular/shared hardlink")
        amount = max(info.st_size, info.st_blocks * 512)
        total += amount
        rows.append({"relative": relative, "kind": kind, "device": info.st_dev,
                     "inode": info.st_ino, "size": info.st_size,
                     "allocated_bytes": info.st_blocks * 512,
                     "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns})
        if len(rows) > max_entries:
            b._fail("metadata entry count exceeds registration")
    return {"root": str(root), "entries": sorted(rows, key=lambda r: r["relative"]),
            "charged_bytes": total}


def _registration(row):
    if row.get("identity_kind") != REG_SCHEMA or row.get("registration_sha256") != row["cache_identity"]:
        b._fail("metadata claim must bind a typed build registration, not a final index identity")
    reg, _ = pinned_json(row["registration_path"], row["registration_sha256"])
    if reg.get("schema") != REG_SCHEMA or reg.get("output_root") != row["root"]:
        b._fail("registration/output root mismatch")
    if reg.get("storage_requirements") != {k: row[k] for k in ("output_bytes", "staging_bytes", "metadata_bytes")}:
        b._fail("registration storage amounts mismatch")
    return reg


def validate_package(core, declarations):
    """Called on every original StorageBudget lock acquisition after migration."""
    package = core["metadata_package"]
    if package.get("schema") != PACKAGE_SCHEMA:
        b._fail("unknown metadata package")
    maximum = b._integer(package.get("maximum_bytes"), "metadata package maximum", positive=True)
    if maximum > PACKAGE_MAX:
        b._fail("metadata package exceeds four GiB")
    old_claims = package["baseline_claims"]
    old_legacy = package["baseline_legacy_reservations"]
    b._hash(package["baseline_policy_sha256"], "baseline policy")
    b._hash(package["baseline_ledger_sha256"], "baseline ledger")
    if any(declarations.get(root) != row for root, row in old_claims.items()):
        b._fail("metadata package changed a baseline claim")
    legacy = {r["claim_id"]: r for r in core["legacy_reservations"]}
    if any(legacy.get(row["claim_id"]) != row for row in old_legacy):
        b._fail("metadata package changed a baseline reservation")
    old_ids = {r["claim_id"] for r in old_legacy}
    added = sum(r["reserved_bytes"] for r in legacy.values() if r["claim_id"] not in old_ids)
    for root, row in declarations.items():
        if root not in old_claims:
            _registration(row)
            added += b.StorageBudget._total(row)
    if added > maximum:
        b._fail("aggregate metadata package exceeds its single allowance")
    return added


def _immutable(path, value):
    if path.exists():
        if _json(path.read_bytes()) != value:
            b._fail("immutable migration artifact conflict")
    else:
        b._atomic(path, value)


def _checkpoint_reserve(path, sha, *, now=None):
    receipt, _ = pinned_json(path, sha)
    now = time.time() if now is None else now
    if receipt.get("schema") != "st_checkpoint_storage_commitments_v1" or receipt.get("status") != "CURRENT":
        b._fail("checkpoint commitments not verified")
    if not receipt["checked_at_unix"] <= now <= receipt["valid_until_unix"]:
        b._fail("checkpoint commitment receipt expired")
    return b._integer(receipt["remaining_bytes"], "remaining checkpoint commitments")


def migrate_policy(*, old_policy, old_policy_sha256, new_policy, new_policy_sha256,
                   expected_old_ledger_sha256, quiescence_receipt, quiescence_sha256,
                   pilot_inventory, pilot_inventory_sha256,
                   checkpoint_receipt, checkpoint_sha256, renew_quiescence_receipt=None,
                   renew_quiescence_sha256=None, fault=None):
    """Only explicit additive migration; all mutation is under CLAIMS.lock.

    `fault` is a test callback, never a CLI switch. Pending records fix the exact
    transition before snapshot/state/receipt, enabling safe same-transition retry.
    """
    old_doc, _ = pinned_json(old_policy, old_policy_sha256)
    new_doc, _ = pinned_json(new_policy, new_policy_sha256)
    old = b.StorageBudget(old_policy)
    new = b.StorageBudget(new_policy)
    if old.policy != old_doc or new.policy != new_doc:
        b._fail("policy changed while opening migration")
    for key in ("schema", "campaign", "state_dir", "cache_cap_bytes", "free_reserve_bytes", "ledger_reserve_bytes"):
        if old.core[key] != new.core[key]:
            b._fail("migration changes immutable campaign envelope: " + key)
    if not set(old.core["allowed_roots"]).issubset(new.core["allowed_roots"]):
        b._fail("migration removed an allowed root")
    if new.core["legacy_reservations"][:len(old.core["legacy_reservations"])] != old.core["legacy_reservations"]:
        b._fail("migration removed/reordered an old reservation")
    if any(new.declarations.get(k) != v for k, v in old.declarations.items()):
        b._fail("migration changed/removed an old claim")
    package = new.core.get("metadata_package")
    if package is None:
        b._fail("migration requires one bounded metadata package")
    if "metadata_package" in old.core:
        if package != old.core["metadata_package"]:
            b._fail("migration changed original metadata package baseline")
    elif (package["baseline_claims"] != old.declarations or
          package["baseline_legacy_reservations"] != old.core["legacy_reservations"] or
          package["baseline_policy_sha256"] != old_policy_sha256 or
          package["baseline_ledger_sha256"] != expected_old_ledger_sha256):
        b._fail("metadata package baseline not exact")
    q, _ = pinned_json(quiescence_receipt, quiescence_sha256)
    inventory, _ = pinned_json(pilot_inventory, pilot_inventory_sha256)
    bindings = {"old_policy_sha256": old_policy_sha256, "new_policy_sha256": new_policy_sha256,
                "old_ledger_sha256": expected_old_ledger_sha256,
                "pilot_inventory_sha256": pilot_inventory_sha256,
                "checkpoint_sha256": checkpoint_sha256}
    if (q.get("schema") != "st_metadata_writer_quiescence_v1" or q.get("status") != "QUIESCENT" or
        q.get("owner_role") != "Lead" or q.get("bindings") != bindings or
        q.get("active_or_unknown_writers") != [] or q.get("queued_writers_disabled") is not True or
        q.get("future_admission_owner") != "Lead"):
        b._fail("missing exact Lead quiescence/queue evidence")
    required_roots = set(old.declarations) | {r["root"] for r in old.core["legacy_reservations"]}
    if inventory.get("schema") != "st_metadata_pilot_inventory_v1":
        b._fail("pilot inventory schema mismatch")
    pilots = {r["root"]: r for r in inventory["roots"]}
    if len(pilots) != len(inventory["roots"]):
        b._fail("duplicate pilot inventory root")
    required_roots |= set(pilots)
    for evidence in q.get("process_evidence", []):
        if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
            b._fail("quiescence process evidence must have exact path/SHA")
        pinned_json(evidence["path"], evidence["sha256"])
    if set(q.get("checked_writer_roots", [])) != required_roots or not q.get("process_evidence"):
        b._fail("quiescence does not cover every writer/pilot root")
    additions = new.core["legacy_reservations"][len(old.core["legacy_reservations"]):]
    if {r["root"] for r in additions} != set(pilots):
        b._fail("new legacy reservations must be exactly inventoried pilots")
    extra_allowed = set(new.core["allowed_roots"]) - set(old.core["allowed_roots"])
    if extra_allowed != {p for p in pilots if not any(b._inside(Path(p), Path(a)) for a in old.core["allowed_roots"])}:
        b._fail("allowed-root extension must be exact pilot roots")
    for row in additions:
        if row["evidence_path"] != str(b._path(pilot_inventory)) or row["evidence_sha256"] != pilot_inventory_sha256:
            b._fail("pilot reservation not bound to inventory")
        amount = ((pilots[row["root"]]["charged_bytes"] + MIB - 1) // MIB) * MIB
        if row["reserved_bytes"] != max(MIB, amount):
            b._fail("pilot reservation must be exact MiB-rounded inventory")
    for root in set(new.declarations) - set(old.declarations):
        if not any(b._inside(Path(root), Path(a)) for a in old.core["allowed_roots"]):
            b._fail("new writer root must remain inside original allowed roots")
    transition = {"schema": MIGRATION_SCHEMA, **bindings,
                  "quiescence_sha256": quiescence_sha256}
    identity = b._digest(transition)
    active_q, active_q_sha = q, quiescence_sha256
    if bool(renew_quiescence_receipt) != bool(renew_quiescence_sha256):
        b._fail("renewal requires both explicit quiescence receipt and SHA")
    if renew_quiescence_receipt:
        active_q, _ = pinned_json(renew_quiescence_receipt, renew_quiescence_sha256)
        active_q_sha = renew_quiescence_sha256
        if (active_q.get("schema") != "st_metadata_writer_quiescence_v1" or
            active_q.get("status") != "QUIESCENT" or active_q.get("owner_role") != "Lead" or
            active_q.get("bindings") != bindings or active_q.get("resumes_migration_id") != identity or
            active_q.get("active_or_unknown_writers") != [] or active_q.get("queued_writers_disabled") is not True or
            active_q.get("future_admission_owner") != "Lead" or
            set(active_q.get("checked_writer_roots", [])) != required_roots or not active_q.get("process_evidence")):
            b._fail("renewed quiescence does not bind the same pending transition")
        for evidence in active_q["process_evidence"]:
            if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
                b._fail("renewed quiescence evidence requires exact pins")
            pinned_json(evidence["path"], evidence["sha256"])
    def require_fresh_quiescence():
        path = renew_quiescence_receipt or quiescence_receipt
        pinned_json(path, active_q_sha)
        now = time.time()
        if not active_q["checked_at_unix"] <= now <= active_q["valid_until_unix"]:
            b._fail("Lead quiescence expired; keep pending evidence and supply explicit renewed receipt/SHA for " + identity)
    pending_path = new.state_dir / ("MIGRATION_" + identity + ".pending.json")
    snapshot_path = new.state_dir / ("MIGRATION_" + identity + ".snapshot.json")
    complete_path = new.state_dir / ("MIGRATION_" + identity + ".complete.json")
    def inject(name):
        if fault is not None:
            fault(name)
    with new.lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            # Recheck exact bytes inside the one global lock.
            pinned_json(old_policy, old_policy_sha256); pinned_json(new_policy, new_policy_sha256)
            for path in (new.state_dir, new.state_path, new.lock_path, pending_path, snapshot_path, complete_path):
                b._path(path)
            require_fresh_quiescence()  # Pending does not waive live quiescence.
            if renew_quiescence_receipt and not pending_path.exists():
                b._fail("renewal requires its original pending migration")
            for other in new.state_dir.glob("MIGRATION_*.pending.json"):
                if other == pending_path:
                    continue
                candidate = _json(other.read_bytes())
                other_complete = other.with_name(other.name.replace(".pending.json", ".complete.json"))
                if (candidate.get("transition", {}).get("old_ledger_sha256") == expected_old_ledger_sha256
                    and not other_complete.exists()):
                    b._fail("conflicting pending migration; renew its exact quiescence instead")
            if pending_path.exists():
                pending = _json(pending_path.read_bytes())
                if pending["transition"] != transition:
                    b._fail("conflicting pending migration")
                old_raw = bytes.fromhex(pending["old_ledger_hex"])
                old_state = _json(old_raw)
            else:
                old_state, old_raw = pinned_json(old.state_path, expected_old_ledger_sha256)
                if old_state.get("core") != old.core:
                    b._fail("old ledger core mismatch")
                for root, record in old_state["claims"].items():
                    if old.declarations.get(root) != record["claim"]:
                        b._fail("old ledger claim mismatch")
                    if not 0 <= record["observed_bytes"] <= record["charged_bytes"] <= old._total(record["claim"]):
                        b._fail("old ledger accounting invalid")
                for root, expected in pilots.items():
                    if tree_inventory(root) != expected:
                        b._fail("pilot inventory changed before adoption")
                pending = {"transition": transition, "old_ledger_hex": old_raw.hex()}
            if _sha(old_raw) != expected_old_ledger_sha256:
                b._fail("pending old snapshot checksum mismatch")
            new_state = copy.deepcopy(old_state)
            new_state["core"] = new.core
            history = new_state.setdefault("metadata_migrations", [])
            if identity not in history:
                history.append(identity)
            current_raw = new.state_path.read_bytes()
            current = _json(current_raw)
            if current != old_state and current != new_state:
                b._fail("ledger progressed or conflicting migration; refresh explicit receipts")
            # Pending retries may not adopt files changed in the fault window.
            for root, expected in pilots.items():
                if tree_inventory(root) != expected:
                    b._fail("pilot inventory changed during migration")
            new._check_cap(new_state)
            reserve = _checkpoint_reserve(checkpoint_receipt, checkpoint_sha256)
            new._check_disk(new_state, new.state_dir,
                            free_reserve_bytes=new.core["free_reserve_bytes"] + reserve)
            # Include snapshot, pending, receipt, state publication and interrupted
            # atomic files in the existing small ledger metadata allowance.
            snapshot = {"old_sha256": expected_old_ledger_sha256, "ledger_hex": old_raw.hex()}
            result = {"schema": MIGRATION_SCHEMA, "status": "MIGRATED", "migration_id": identity,
                      **transition, "completion_quiescence_sha256": active_q_sha,
                      "completion_quiescence_receipt": str(b._path(renew_quiescence_receipt or quiescence_receipt)),
                      "snapshot_path": str(snapshot_path),
                      "new_core_sha256": b._digest(new.core),
                      "new_ledger_sha256": _sha((json.dumps(new_state, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()),
                      "metadata_package_reserved_bytes": validate_package(new.core, new.declarations)}
            if complete_path.exists():
                completed = _json(complete_path.read_bytes())
                comparable = dict(completed, completion_quiescence_sha256=active_q_sha,
                                  completion_quiescence_receipt=result["completion_quiescence_receipt"])
                if current != new_state or comparable != result:
                    b._fail("completed migration no longer matches exact state/transition")
                return completed  # Read-only retry does not rewrite its old receipt.
            used, _ = b._inventory(new.state_dir)
            required = sum(b.json_write_bound(v) for v in (pending, snapshot, result, new_state))
            if used + required > new.core["ledger_reserve_bytes"]:
                b._fail("migration metadata/staging exceeds ledger reserve")
            _immutable(pending_path, pending); inject("after_pending")
            _immutable(snapshot_path, snapshot); inject("after_snapshot")
            require_fresh_quiescence()
            if current != new_state:
                new._save(new_state)
            inject("after_state")
            require_fresh_quiescence()
            _immutable(complete_path, result); inject("after_receipt")
            return result
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _resources(reg):
    limits = reg["limits"]
    for name in ("max_disk_bytes", "max_rss_bytes", "max_seconds", "max_entries"):
        b._integer(limits[name], name, positive=True)
    b._integer(limits["max_depth"], "max_depth")
    if (limits["max_disk_bytes"] > PACKAGE_MAX or limits["max_rss_bytes"] > 4 * b.GIB or
        limits["max_seconds"] > 4 * 3600 or limits["max_entries"] > 20000 or limits["max_depth"] > 4):
        b._fail("metadata resource limit exceeds bounded implementation")
    if not isinstance(reg.get("directory_patterns", []), list):
        b._fail("metadata directory patterns must be explicit")
    if not isinstance(reg.get("file_patterns"), list) or not reg["file_patterns"]:
        b._fail("explicit metadata filename patterns required")
    for pattern in reg["file_patterns"] + reg.get("directory_patterns", []):
        if not isinstance(pattern, str) or len(pattern) > 256:
            b._fail("invalid metadata filename pattern")
        re.compile(pattern)
    if reg.get("writer_kind") not in ("current_index", "future_preparation", "future_index"):
        b._fail("unsupported metadata writer")
    if limits["max_depth"] != 0 and reg["writer_kind"] != "future_preparation":
        b._fail("index output must stay flat")
    return limits


def process_tree_rss(pid):
    """Current Linux RSS of this process and every descendant (not ru_maxrss).

    Fail closed on an inaccessible extant /proc entry. No process signaling.
    """
    seen, pending, total = set(), [pid], 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        base = Path("/proc") / str(current)
        try:
            status = (base / "status").read_text()
            value = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.M)
            if value:
                total += int(value.group(1)) * 1024
            tasks = list((base / "task").iterdir())
            for task in tasks:
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            if base.exists():
                b._fail("cannot inspect live metadata process tree")
    return total


class MetadataWriteContext:
    """Only valid while metadata_writing owns the original campaign flock."""
    def __init__(self, budget, state, root, claim, reg, checkpoint_path, checkpoint_sha):
        self.budget, self.state, self.root, self.claim, self.registration = budget, state, root, claim, reg
        self.limits = _resources(reg)
        self.checkpoint_path, self.checkpoint_sha = checkpoint_path, checkpoint_sha
        self.started = time.monotonic()
        self.last_check = self.started
        self.failed = None
        self.peak_rss_bytes = 0
        self.peak_disk_bytes = 0
        self.checks = 0
        self.active = True
        self.monitoring = False
        self.commit_checks = 0
        self.callback_lock = threading.RLock()
        self.writer_callbacks = 0
        self.writer_commit_callbacks = 0

    def check(self, *, before_commit=False, path=None):
        """Call during long reads and immediately before *every* publication.

        The caller must call at least every five seconds and before publication.
        Lead's independent finite process-tree monitor covers blocked native IO;
        this cooperative context is not a cgroup or asynchronous kill mechanism.
        Sticky failure prevents later publication through this context. This is
        not an implicit check of image/tensor legality; the writer keeps its own
        existing source/contract checks.
        """
        if not self.active:
            b._fail("metadata guard is closed")
        if self.failed:
            b._fail("metadata guard already failed: " + self.failed)
        try:
            now = time.monotonic()
            if not self.monitoring and now - self.last_check > 5:
                b._fail("metadata check interval exceeded five seconds; keep reads bounded and use the outer finite monitor")
            if now - self.started > self.limits["max_seconds"]:
                b._fail("metadata deadline exceeded")
            # No recursive flock: state is the exact precharged state under the
            # still-held original lock. Other cooperative writers are blocked.
            b._path(self.root)
            if path is not None:
                target = b._path(path)
                if target == self.root or not b._inside(target, self.root):
                    b._fail("metadata write destination escapes registered root")
                relative = target.relative_to(self.root).as_posix()
                if not any(re.fullmatch(p, relative) for p in self.registration["file_patterns"]):
                    b._fail("unregistered metadata write destination")
                # Validate every planned ancestor before mkdir, even when the
                # whole parent chain is not on disk yet. Census alone is too late.
                directories = target.relative_to(self.root).parts[:-1]
                if len(directories) > self.limits["max_depth"]:
                    b._fail("planned metadata directory depth exceeds registration")
                for depth in range(1, len(directories) + 1):
                    planned = Path(*directories[:depth]).as_posix()
                    if not any(re.fullmatch(p, planned) for p in self.registration.get("directory_patterns", [])):
                        b._fail("unregistered planned metadata directory: " + planned)
                parent = b._existing_parent(target.parent)
                if parent.stat().st_dev != self.root.stat().st_dev:
                    b._fail("metadata write parent is a different filesystem")
            observed = tree_inventory(self.root, max_depth=self.limits["max_depth"],
                                      max_entries=self.limits["max_entries"],
                                      patterns=self.registration["file_patterns"],
                                      directory_patterns=self.registration.get("directory_patterns", []))
            disk = observed["charged_bytes"]
            rss = process_tree_rss(os.getpid())
            self.peak_disk_bytes = max(self.peak_disk_bytes, disk)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            if disk > self.limits["max_disk_bytes"] or disk > self.budget._total(self.claim):
                b._fail("metadata physical/logical allocation exceeds reservation")
            if rss > self.limits["max_rss_bytes"]:
                b._fail("metadata process-tree RSS exceeds reservation")
            checkpoints = _checkpoint_reserve(self.checkpoint_path, self.checkpoint_sha)
            self.budget._check_disk(self.state, self.root,
                                   free_reserve_bytes=self.budget.core["free_reserve_bytes"] + checkpoints)
            self.last_check = time.monotonic()
            self.checks += 1
            if before_commit:
                self.commit_checks += 1
            return {"checks": self.checks, "charged_disk_bytes": disk, "process_tree_rss_bytes": rss,
                    "elapsed_seconds": self.last_check - self.started, "before_commit": before_commit}
        except Exception as error:
            self.failed = str(error)
            raise


@contextmanager
def metadata_writing(*, policy_path, policy_sha256, registration_path, registration_sha256,
                     checkpoint_receipt, checkpoint_sha256):
    """Admit an exact metadata registration and precharge the complete attempt.

    Holds CLAIMS.lock across the finite build. Root must be empty on its first
    admission; resume requires its own immutable METADATA_ADMISSION.json. No
    tensor `identity.json` is fabricated and the final index identity is unused.
    """
    global _ACTIVE_GUARD
    if _ACTIVE_GUARD is not None:
        b._fail("nested metadata writers are not supported")
    policy, _ = pinned_json(policy_path, policy_sha256)
    reg, _ = pinned_json(registration_path, registration_sha256)
    if reg.get("schema") != REG_SCHEMA:
        b._fail("metadata registration schema mismatch")
    limits = _resources(reg)
    budget = b.StorageBudget(policy_path)
    if budget.policy != policy or "metadata_package" not in budget.core:
        b._fail("metadata writes require explicit migrated package")
    root = b._path(reg["output_root"])
    with budget._locked() as state:
        pinned_json(policy_path, policy_sha256)
        claim = budget.declarations.get(str(root))
        if claim is None or claim["cache_identity"] != registration_sha256 or _registration(claim) != reg:
            b._fail("metadata registration is not the declared claim")
        if str(b._path(registration_path)) != claim["registration_path"]:
            b._fail("metadata registration path differs from declaration")
        total = budget._total(claim)
        if limits["max_disk_bytes"] != total:
            b._fail("writer bound must include all output/staging/control bytes")
        observed = tree_inventory(root, max_depth=limits["max_depth"], max_entries=limits["max_entries"],
                                  patterns=reg["file_patterns"], directory_patterns=reg.get("directory_patterns", []))
        known = state["claims"].get(str(root))
        admission = {"schema": "st_metadata_write_admission_v1", "registration_sha256": registration_sha256,
                     "registration_path": str(b._path(registration_path)), "claim": claim,
                     "policy_sha256_at_admission": policy_sha256}
        admission_path = root / "METADATA_ADMISSION.json"
        files = [r for r in observed["entries"] if r["kind"] == "file"]
        if known is None and files:
            pins = reg.get("preexisting_pins", [])
            if (not isinstance(pins, list) or len(pins) > 8 or
                len({p["relative_path"] for p in pins}) != len(pins) or
                {p["relative_path"] for p in pins} != {r["relative"] for r in files}):
                b._fail("unclaimed metadata root contains foreign files")
            for pin in pins:
                target = b._path(root / pin["relative_path"])
                if not b._inside(target, root):
                    b._fail("preexisting metadata pin escapes root")
                pinned_bytes(target, pin["sha256"])
        if known is not None:
            if known["claim"] != claim:
                b._fail("metadata resume claim changed")
            if admission_path.exists():
                existing = _json(admission_path.read_bytes())
                if (existing.get("registration_sha256") != registration_sha256 or
                    existing.get("registration_path") != str(b._path(registration_path)) or existing.get("claim") != claim):
                    b._fail("metadata admission differs from resume registration")
                admission = existing  # Keep the original policy provenance.
            elif files:
                pins = reg.get("preexisting_pins", [])
                if {p["relative_path"] for p in pins} != {r["relative"] for r in files}:
                    b._fail("claimed metadata files lack original admission receipt")
                for pin in pins:
                    pinned_bytes(root / pin["relative_path"], pin["sha256"])
        if observed["charged_bytes"] > total:
            b._fail("retained metadata partials exceed claim")
        checkpoints = _checkpoint_reserve(checkpoint_receipt, checkpoint_sha256)
        state["claims"][str(root)] = {"claim": claim, "observed_bytes": observed["charged_bytes"],
                                     "charged_bytes": total, "device": b._existing_parent(root).stat().st_dev}
        budget._check_cap(state)
        budget._check_disk(state, root, free_reserve_bytes=budget.core["free_reserve_bytes"] + checkpoints)
        budget._save(state)  # Full durable precharge precedes mkdir/admission/SQLite.
        root.mkdir(parents=True, exist_ok=True)
        if observed["charged_bytes"] + b.json_write_bound(admission) > total:
            b._fail("metadata admission metadata has no remaining allowance")
        _immutable(admission_path, admission)
        guard = MetadataWriteContext(budget, state, root, claim, reg, checkpoint_receipt, checkpoint_sha256)
        if _ACTIVE_GUARD is not None:
            b._fail("nested metadata writers are not supported")
        _ACTIVE_GUARD = guard
        try:
            guard.check()
            yield guard
            guard.check(before_commit=True)
        finally:
            guard.active = False
            _ACTIVE_GUARD = None
            # Never release the claim or forget charged partial files on error.


_ACTIVE_GUARD = None


def check_active_metadata_storage(*, before_commit=False, path=None):
    """Thin process-local hook shared with the builder's bounded worker threads.

    No-op outside an admitted invocation. Registered runner checks hook counts,
    so an old unwired CLI cannot be reported as resource-qualified. It neither
    opens another ledger nor reacquires the existing flock.
    """
    guard = _ACTIVE_GUARD
    if guard is None:
        return None
    with guard.callback_lock:
        guard.writer_callbacks += 1
        if before_commit:
            guard.writer_commit_callbacks += 1
        return guard.check(before_commit=before_commit, path=path)


def run_registered_metadata(**kwargs):
    """Thin same-process dispatcher for the two metadata builders only.

    Exact command is pinned by the registration. The owner must wire the hook
    at bounded hash/read chunks and immediately before writes/publications.
    Lead supplies the outer finite process-tree monitor/timeout, CPU affinity and
    authorization; this function does not launch or signal a process or job.
    """
    reg, _ = pinned_json(kwargs["registration_path"], kwargs["registration_sha256"])
    command = reg.get("command")
    if not isinstance(command, list) or not all(isinstance(a, str) for a in command) or len(command) < 4:
        b._fail("metadata command must be exact argv")
    expected_module = {"current_index": "tools.st_world_policy.index_teacher_metadata",
                       "future_index": "tools.st_world_policy.index_teacher_metadata",
                       "future_preparation": "tools.st_world_policy.prepare_future_metadata"}[reg["writer_kind"]]
    if command[1:3] != ["-m", expected_module] or command[3] not in ("build", "verify"):
        b._fail("runner only accepts registered metadata CLI")
    output_flag = "--plan" if reg["writer_kind"] == "future_preparation" else ("--index" if command[3] == "verify" else "--output")
    if output_flag not in command or command.index(output_flag) + 1 >= len(command):
        b._fail("metadata command lacks explicit output binding")
    command_output = b._path(command[command.index(output_flag) + 1])
    if output_flag == "--plan": command_output = command_output.parent
    if str(command_output) != reg["output_root"]:
        b._fail("command output differs from registered metadata root")
    if reg.get("callback_protocol") != "st_metadata_storage_check_v1":
        b._fail("metadata child must use the reviewed resource callback")
    if b._path(reg["cwd"]) != Path.cwd().resolve():
        b._fail("run from the exact registered source cwd")
    # Validate registered source bytes before importing the entry point. Imports
    # do not call parser/build; capabilities and every declared dependency must
    # pass before even writer admission or output directory creation.
    pins = reg.get("source_pins")
    if not isinstance(pins, list) or not 1 <= len(pins) <= 16:
        b._fail("metadata source pins are required before dispatch")
    sources = {}
    cwd = Path.cwd().resolve()
    for pin in pins:
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            b._fail("metadata source pin requires exact path/SHA")
        path = b._path(pin["path"])
        if path in sources or not b._inside(path, cwd) or path.suffix != ".py":
            b._fail("metadata source pin is duplicate or outside registered source")
        pinned_bytes(path, pin["sha256"])
        sources[path] = pin["sha256"]
    entry = cwd.joinpath(*expected_module.split(".")).with_suffix(".py")
    if entry not in sources:
        b._fail("metadata entry point source pin missing")
    module = importlib.import_module(expected_module)
    if (getattr(module, "METADATA_STORAGE_CALLBACK_PROTOCOL", None) != "st_metadata_storage_check_v1" or
        b._path(getattr(module, "__file__", "")) != entry):
        b._fail("metadata module is not a source-bound callback-capable implementation")
    dependencies = getattr(module, "METADATA_STORAGE_CALLBACK_DEPENDENCIES", None)
    if not isinstance(dependencies, tuple) or not all(isinstance(name, str) for name in dependencies):
        b._fail("metadata callback dependency declaration missing")
    expected_sources = {entry}
    for name in dependencies:
        if not re.fullmatch(r"(?:starVLA|tools)(?:\.[A-Za-z_][A-Za-z_0-9]*)+", name):
            b._fail("invalid metadata callback dependency module")
        expected_sources.add(cwd.joinpath(*name.split(".")).with_suffix(".py"))
    if set(sources) != expected_sources:
        b._fail("metadata callback dependency source pins incomplete or extra")
    for path, sha in sources.items():
        pinned_bytes(path, sha)
    # No subprocess or arbitrary command execution. The two real parsers still
    # validate every option, keeping all input/schema checks in their owners.
    with metadata_writing(**kwargs) as guard:
        args = module.parser().parse_args(command[3:])
        result = module.build(args) if reg["writer_kind"] == "future_preparation" else module.run(args)
        guard.check(before_commit=True)
        if guard.writer_callbacks == 0 or (command[3] == "build" and guard.writer_commit_callbacks == 0):
            b._fail("metadata writer never called required resource/publication hook")
        if isinstance(result, dict) and result.get("status") in ("PARTIAL", "FAILED", "BLOCKED"):
            b._fail("metadata writer returned incomplete status")
        return {"status": "GUARDED_METADATA_RETURN", "artifact": result,
                "registration_sha256": kwargs["registration_sha256"],
                "checks": guard.checks, "writer_callbacks": guard.writer_callbacks,
                "writer_commit_callbacks": guard.writer_commit_callbacks,
                "peak_disk_bytes": guard.peak_disk_bytes,
                "peak_process_tree_rss_bytes": guard.peak_rss_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one finite, pinned metadata API; no automatic migration")
    for name in ("policy-path", "policy-sha256", "registration-path", "registration-sha256",
                 "checkpoint-receipt", "checkpoint-sha256"):
        run.add_argument("--" + name, required=True)
    migration = commands.add_parser("migrate", help="explicit Lead-receipted same-ledger additive migration")
    for name in ("old-policy", "old-policy-sha256", "new-policy", "new-policy-sha256",
                 "expected-old-ledger-sha256", "quiescence-receipt", "quiescence-sha256",
                 "pilot-inventory", "pilot-inventory-sha256", "checkpoint-receipt", "checkpoint-sha256"):
        migration.add_argument("--" + name, required=True)
    migration.add_argument("--renew-quiescence-receipt")
    migration.add_argument("--renew-quiescence-sha256")
    args = vars(parser.parse_args())
    name = args.pop("command")
    try:
        result = migrate_policy(**args) if name == "migrate" else run_registered_metadata(**args)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "BLOCKED_STORAGE", "error": str(error)}, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
