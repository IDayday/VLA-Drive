"""Durable, campaign-wide byte reservations for cooperative teacher writers.

This module has no torch/model dependency. Reservations are never evicted or
automatically released: interrupted output and orphan files still occupy space.
Only the new cache_teacher entry point is guarded; immutable old runtimes must
be represented by explicit, conservative legacy reservations.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


GIB = 1 << 30
DEFAULT_CACHE_CAP_BYTES = 768 * GIB
DEFAULT_FREE_RESERVE_BYTES = 1024 * GIB
DEFAULT_LEDGER_RESERVE_BYTES = 16 << 20
SCHEMA = "st_cache_storage_budget_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def json_bytes(value):
    """The same serialization used by the existing teacher atomic_json writer."""
    return len((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def json_write_bound(value):
    # Also covers filesystem block rounding and the old/new atomic-file window.
    return json_bytes(value) + 8192


def teacher_storage_requirements(identity, manifest, bytes_per_image, world_size):
    """Conservative byte bounds, independent of resource policy and tensor math.

    Metadata includes one receipt/progress allowance per shard, rank receipts,
    plan/config files, and publication. Staging covers concurrent full shards,
    initial immutable metadata, and bounded interrupted-file retention.
    """
    images = len(identity["records"])
    shards = identity["shards"]
    largest = min(images, identity["images_per_shard"])
    shard_bytes = largest * bytes_per_image + 8192
    initial_metadata = json_write_bound(identity) + json_write_bound(manifest)
    shard_metadata = 32768 + largest * 512
    return {
        "output_bytes": images * bytes_per_image + shards * 8192,
        "staging_bytes": max(world_size * (shard_bytes + shard_metadata), initial_metadata, (1 << 20) + shards * 256),
        "metadata_bytes": world_size * initial_metadata + images * 512 + shards * 65536 + (world_size + 4) * (1 << 20),
    }


class StorageBudgetError(ValueError):
    pass


def _fail(message):
    raise StorageBudgetError("BLOCKED_STORAGE: " + message)


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        _fail(name + " must be an exact lowercase SHA256")
    return value


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        _fail(name + " must be an explicit nonnegative integer")
    return value


def _path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != str(path.resolve()):
        _fail("noncanonical or symlink path: " + str(path))
    return path


def _inside(path, parent):
    return path == parent or parent in path.parents


def _existing_parent(path):
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        _fail("storage parent is not a directory: " + str(path))
    return path


def _atomic(path, value):
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _inventory(root):
    """Stat only once at admission/resume; never rehash tensor payloads."""
    if not root.exists():
        return 0, []
    if not root.is_dir():
        _fail("cache root is not a directory")
    total, names = 0, []
    for entry in os.scandir(root):
        info = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("cache contains a symlink, directory, or shared hardlink: " + entry.path)
        total += max(info.st_size, info.st_blocks * 512)
        names.append(entry.name)
    return total, names


class StorageBudget:
    """One immutable campaign envelope; append-only declarations and claims.

    A policy may add declarations. The cap, reserve, legacy reservations and
    previously admitted claims cannot change. All processes must use the same
    canonical policy state_dir on a filesystem supporting cross-host flock.
    """
    def __init__(self, policy_path):
        self.policy_path = _path(policy_path)
        self.policy, self.core, self.declarations = self._read_policy()
        self.state_dir = Path(self.core["state_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _path(self.state_dir)
        self.state_path = self.state_dir / "CLAIMS.json"
        self.lock_path = self.state_dir / "CLAIMS.lock"

    def _read_policy(self):
        policy = json.loads(self.policy_path.read_text())
        if policy.get("schema") != SCHEMA or not isinstance(policy.get("campaign"), str) or not policy["campaign"]:
            _fail("unrecognized storage policy/campaign")
        allowed = [_path(p) for p in policy["allowed_roots"]]
        if not allowed:
            _fail("storage policy needs explicit allowed roots")
        core = {"schema": SCHEMA, "campaign": policy["campaign"],
                "state_dir": str(_path(policy["state_dir"])),
                "allowed_roots": [str(p) for p in allowed],
                "cache_cap_bytes": _integer(policy.get("cache_cap_bytes", DEFAULT_CACHE_CAP_BYTES), "cache cap", positive=True),
                "free_reserve_bytes": _integer(policy.get("free_reserve_bytes", DEFAULT_FREE_RESERVE_BYTES), "free reserve"),
                "ledger_reserve_bytes": _integer(policy.get("ledger_reserve_bytes", DEFAULT_LEDGER_RESERVE_BYTES), "ledger reserve", positive=True),
                "legacy_reservations": policy.get("legacy_reservations", [])}
        if "metadata_package" in policy:
            core["metadata_package"] = policy["metadata_package"]
        if "frozen_w_package" in policy:
            core["frozen_w_package"] = policy["frozen_w_package"]
        roots, ids, identities, declarations = [], set(), set(), {}
        for legacy, rows in ((True, core["legacy_reservations"]), (False, policy.get("claims", []))):
            for row in rows:
                root = _path(row["root"])
                if not any(_inside(root, p) for p in allowed):
                    _fail("claim path escapes allowed roots")
                if any(_inside(root, old) or _inside(old, root) for old in roots):
                    _fail("duplicate or overlapping cache roots")
                if _inside(root, Path(core["state_dir"])) or _inside(Path(core["state_dir"]), root):
                    _fail("claim overlaps its storage ledger")
                claim_id = row["claim_id"]
                if not isinstance(claim_id, str) or not claim_id or claim_id in ids:
                    _fail("duplicate or invalid claim_id")
                ids.add(claim_id)
                roots.append(root)
                if legacy:
                    _integer(row["reserved_bytes"], "legacy reserved bytes", positive=True)
                    _hash(row["evidence_sha256"], "legacy evidence identity")
                    evidence = _path(row["evidence_path"])
                    if evidence.stat().st_size > 16 * (1 << 20):
                        _fail("legacy evidence must be a small immutable registration, not a cache payload")
                    if hashlib.sha256(evidence.read_bytes()).hexdigest() != row["evidence_sha256"]:
                        _fail("legacy reservation evidence changed")
                else:
                    identity = _hash(row["cache_identity"], "cache identity")
                    if identity in identities:
                        _fail("duplicate cache identity at a different root")
                    identities.add(identity)
                    for name in ("output_bytes", "staging_bytes", "metadata_bytes"):
                        _integer(row[name], name, positive=name != "output_bytes")
                    declarations[str(root)] = row
        if core["ledger_reserve_bytes"] + sum(r["reserved_bytes"] for r in core["legacy_reservations"]) > core["cache_cap_bytes"]:
            _fail("legacy reservations exceed campaign cap")
        frozen_w_rows = [r for r in declarations.values() if r.get("identity_kind") == "st_frozen_spatial_w_v1"]
        if "frozen_w_package" in core:
            from tools.st_world_policy.spatial_w_storage import validate_package as validate_frozen_w
            validate_frozen_w(core, declarations)
            # A W package reserves every declared byte once in this same
            # campaign envelope, including claims not yet admitted by writers.
            declared = (core["ledger_reserve_bytes"] + sum(r["reserved_bytes"] for r in core["legacy_reservations"])
                        + sum(self._total(r) for r in declarations.values()))
            if declared > core["cache_cap_bytes"]:
                _fail("all declared claims exceed the original campaign cap")
        elif frozen_w_rows:
            _fail("typed frozen W claims require explicit additive package migration")
        if "metadata_package" in core:
            from tools.st_world_policy.metadata_storage import validate_package
            validate_package(core, declarations)
        return policy, core, declarations

    @contextmanager
    def _locked(self):
        # Refuse retargeting an existing instance or its lock/state via symlinks.
        for path in (self.policy_path, self.state_dir, self.state_path, self.lock_path):
            _path(path)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                policy, core, declarations = self._read_policy()
                if core != self.core:
                    _fail("campaign storage envelope changed")
                if "metadata_package" in core and not self.state_path.exists():
                    _fail("metadata package requires explicit migration of an existing ledger")
                state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
                    "schema": SCHEMA, "core": core, "claims": {}}
                if state.get("schema") != SCHEMA or state.get("core") != core:
                    _fail("campaign ledger belongs to a different policy")
                for root, record in state["claims"].items():
                    if declarations.get(root) != record["claim"]:
                        _fail("an admitted claim changed or disappeared")
                    if not 0 <= record["observed_bytes"] <= record["charged_bytes"] <= self._total(record["claim"]):
                        _fail("invalid persisted cache accounting")
                self.policy, self.declarations = policy, declarations
                self._check_cap(state)
                yield state
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _total(claim):
        return sum(claim[k] for k in ("output_bytes", "staging_bytes", "metadata_bytes"))

    def _check_cap(self, state):
        total = self.core["ledger_reserve_bytes"] + sum(r["reserved_bytes"] for r in self.core["legacy_reservations"])
        total += sum(self._total(r["claim"]) for r in state["claims"].values())
        if total > self.core["cache_cap_bytes"]:
            _fail("aggregate cache reservations exceed campaign cap")
        return total

    def _check_disk(self, state, root, extra_bytes=0, free_reserve_bytes=0):
        parent = _existing_parent(root)
        device = parent.stat().st_dev
        outstanding = self.core["ledger_reserve_bytes"] if self.state_dir.stat().st_dev == device else 0
        for row in self.core["legacy_reservations"]:
            if _existing_parent(Path(row["root"])).stat().st_dev == device:
                # Legacy writers are not instrumented; reserve their full bound.
                outstanding += row["reserved_bytes"]
        for record in state["claims"].values():
            if record["device"] == device:
                # observed_bytes is a stat census, not the precharged upper bound.
                # It is intentionally conservative between resumes.
                outstanding += self._total(record["claim"]) - record["observed_bytes"]
        floor = max(self.core["free_reserve_bytes"], free_reserve_bytes)
        if shutil.disk_usage(parent).free < floor + max(extra_bytes, outstanding):
            _fail("remaining reservations plus separate free-space reserve do not fit")

    def _save(self, state):
        # Interrupted atomic ledger files are retained and bounded too.
        used, _ = _inventory(self.state_dir)
        growth = json_write_bound(state)
        if used + growth > self.core["ledger_reserve_bytes"]:
            _fail("ledger metadata/staging allowance exhausted")
        self._check_disk(state, self.state_dir, growth)
        _atomic(self.state_path, state)

    @staticmethod
    def migrate_metadata_policy(**kwargs):
        """Explicit receipt-bound additive migration; never an implicit initialize.

        This API keeps the original CLAIMS.lock and ledger. It does not inspect
        or stop jobs; Lead must supply pinned live quiescence evidence.
        """
        from tools.st_world_policy.metadata_storage import migrate_policy
        return migrate_policy(**kwargs)

    def initialize(self):
        """Persist the legacy envelope without admitting or launching a build."""
        with self._locked() as state:
            self._save(state)
            return {"status": "STORAGE_BUDGET_READY", "policy_sha256": _digest(self.policy),
                    "cache_cap_bytes": self.core["cache_cap_bytes"],
                    "free_reserve_bytes": self.core["free_reserve_bytes"],
                    "legacy_reserved_bytes": sum(r["reserved_bytes"] for r in self.core["legacy_reservations"]),
                    "ledger_reserved_bytes": self.core["ledger_reserve_bytes"],
                    "campaign_reserved_bytes": self._check_cap(state),
                    "guarded_claims": len(state["claims"])}

    def reserve(self, root, cache_identity, *, output_bytes, staging_bytes, metadata_bytes, free_reserve_bytes=0):
        root = _path(root)
        _hash(cache_identity, "cache identity")
        required = {"output_bytes": output_bytes, "staging_bytes": staging_bytes, "metadata_bytes": metadata_bytes}
        for key, value in required.items():
            _integer(value, key)
        _integer(free_reserve_bytes, "free reserve")
        with self._locked() as state:
            claim = self.declarations.get(str(root))
            if claim is None or claim["cache_identity"] != cache_identity:
                _fail("unknown cache root or identity; declare it before building")
            if any(required[k] > claim[k] for k in required):
                _fail("declared output, staging, or metadata allowance is too small")
            observed, names = _inventory(root)
            known = str(root) in state["claims"]
            identity_path = root / "identity.json"
            if identity_path.exists():
                identity = json.loads(identity_path.read_text())
                if identity.get("identity") != cache_identity or _digest({k:v for k,v in identity.items() if k != "identity"}) != cache_identity:
                    _fail("existing cache identity differs from claim")
            elif names and not known:
                # The plan command can precede admission but must bind the identity.
                for name in names:
                    if not re.fullmatch(r"PLAN_rank\d+\.json|resolved_config_plan_rank\d+\.json", name):
                        _fail("unclaimed existing output has no cache identity")
                    if json.loads((root / name).read_text()).get("cache_identity") != cache_identity:
                        _fail("preexisting plan identity differs from claim")
            if observed > self._total(claim):
                _fail("existing output and interrupted files exceed the claim")
            state["claims"][str(root)] = {"claim": claim, "observed_bytes": observed,
                                          "charged_bytes": observed,
                                          "device": _existing_parent(root).stat().st_dev}
            total = self._check_cap(state)
            self._check_disk(state, root, free_reserve_bytes=free_reserve_bytes)
            self._save(state)
            return CacheReservation(self, root, claim, free_reserve_bytes,
                                    {"policy_sha256": _digest(self.policy), "claim_id": claim["claim_id"],
                                     "guard_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     "cache_identity": cache_identity, "reserved_bytes": self._total(claim),
                                     "campaign_reserved_bytes": total, "cache_cap_bytes": self.core["cache_cap_bytes"],
                                     "free_reserve_bytes": max(self.core["free_reserve_bytes"], free_reserve_bytes)})


class CacheReservation:
    def __init__(self, budget, root, claim, free_reserve_bytes, receipt):
        self.budget, self.root, self.claim = budget, root, claim
        self.free_reserve_bytes, self.receipt = free_reserve_bytes, receipt

    @contextmanager
    def writing(self, maximum_new_bytes):
        """Precharge a bounded write before IO, keeping flock through commit.

        Callers supply a serialization upper bound. The teacher hooks bound the
        known safetensors format and every JSON serialization. Charges include
        overwritten metadata and remain conservative after exceptions; reserve()
        reconciles them by stat on a subsequent resume. No per-shard tree scan.
        """
        _integer(maximum_new_bytes, "write bound")
        with self.budget._locked() as state:
            _path(self.root)
            record = state["claims"].get(str(self.root))
            if record is None or record["claim"] != self.claim:
                _fail("missing or changed reservation")
            if maximum_new_bytes > self.claim["staging_bytes"]:
                _fail("write exceeds bounded staging allowance")
            if record["charged_bytes"] + maximum_new_bytes > self.budget._total(self.claim):
                _fail("cache write exceeds reserved output/staging/metadata cap")
            self.budget._check_disk(state, self.root, maximum_new_bytes, self.free_reserve_bytes)
            record["charged_bytes"] += maximum_new_bytes
            # Durable precharge survives SIGKILL in the writer's commit window.
            self.budget._save(state)
            yield
