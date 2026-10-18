#!/usr/bin/env python3
"""Independent, tiny CPU fixtures; never invoke fixed-child or GPU admission."""
import argparse
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

sys.dont_write_bytecode = True


def run(wrapper):
    spec = importlib.util.spec_from_file_location("reviewed_wrapper", wrapper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = []

    def check(name, fn):
        started = time.monotonic()
        try:
            detail = fn()
            results.append({"name": name, "status": "PASS", "detail": detail,
                            "wall_seconds": time.monotonic() - started})
        except Exception as exc:
            results.append({"name": name, "status": "FAIL", "error": repr(exc),
                            "wall_seconds": time.monotonic() - started})

    with tempfile.TemporaryDirectory(prefix="independent_gradient_resources_") as tmp:
        root = Path(tmp)

        def admission():
            payload = root / "tiny.bin"
            payload.write_bytes(b"12345678")
            ledger = mod.ReadAdmission(root / "tiny_counter.json", limit=16)
            calls = []

            def read(path, **kwargs):
                calls.append(kwargs)
                with open(path, "rb") as handle:
                    with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
                        return bytes(view)

            wrapped = ledger.wrap(read, "torch_load_nominal_bytes")
            assert wrapped(payload, mmap=True) == b"12345678"
            assert ledger.total() == 8
            assert wrapped(payload, mmap=True) == b"12345678"
            try:
                wrapped(payload, mmap=True)
            except RuntimeError:
                pass
            else:
                raise AssertionError("over-budget mmap operation was admitted")
            assert len(calls) == 2 and ledger.total() == 16
            assert ledger.values["rejected_admission"]["bytes"] == 8
            return {"mmap_calls": len(calls), "admitted_bytes": ledger.total(),
                    "rejection_before_loader": True}

        check("mmap_and_repeated_load_precharge", admission)

        def unaccountable():
            ledger = mod.ReadAdmission(root / "unknown_counter.json", limit=16)
            calls = []
            wrapped = ledger.wrap(lambda x: calls.append(x), "torch_load_nominal_bytes")
            try:
                wrapped(object())
            except ValueError:
                pass
            else:
                raise AssertionError("unaccountable input was admitted")
            assert calls == []
            return {"loader_called": False}

        check("unknown_input_rejected_before_loader", unaccountable)

        def child_case(name, script, expected_status, expected_reason=None, **limits):
            attempt = root / name
            attempt.mkdir()
            output = root / (name + "_output")
            result = mod.supervise([sys.executable, "-c", script], attempt=attempt,
                output=output, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                wall=44, **limits)
            assert result["status"] == expected_status, result
            assert result.get("reason") == expected_reason, result
            assert result.get("live_owned_remaining") == 0, result
            assert result["no_RLIMIT_AS"] is True, result
            return result

        check("success_and_owned_cleanup", lambda: child_case("success", "print('tiny')", "EXITED"))
        check("native_failure_no_retry", lambda: child_case("failure", "raise SystemExit(23)", "STOPPED_NO_AUTO_RETRY"))
        check("rss_stop_and_cleanup", lambda: child_case("rss", "import time; time.sleep(10)",
            "STOPPED_NO_AUTO_RETRY", "RSS_LIMIT", rss_limit=1))
        check("wall_stop_and_cleanup", lambda: child_case("wall", "import time; time.sleep(10)",
            "STOPPED_NO_AUTO_RETRY", "WALL_LIMIT"))
        check("stdout_stop_and_cleanup", lambda: child_case("stdout", "import sys; sys.stdout.write('x' * (3 << 20)); sys.stdout.flush()",
            "STOPPED_NO_AUTO_RETRY", "STDOUT_LIMIT"))

        def fast_read():
            payload = root / "read.bin"
            payload.write_bytes(b"z" * (256 << 10))
            def after_exit_sample(owner, observed):
                time.sleep(.1)
                return mod.process_sample(owner, observed)
            result = child_case("fast_read", "from pathlib import Path; Path(" + repr(str(payload)) + ").read_bytes()",
                "STOPPED_NO_AUTO_RETRY", "READ_LIMIT", read_limit=128 << 10, sample=after_exit_sample)
            assert result["observed_rchar_bytes"] >= 256 << 10, result
            return result

        check("exited_leader_rchar_accounted_before_reap", fast_read)

        def orphan():
            code = "import os,time\nif os.fork() == 0: time.sleep(10)\nelse: time.sleep(.3)\n"
            return child_case("orphan", code, "STOPPED_NO_AUTO_RETRY", "OWNED_DESCENDANTS_AFTER_LEADER_EXIT")

        check("owned_descendant_cleanup", orphan)

    return {"schema": "gradient_balance_independent_resources_fixture_v1",
            "wrapper_path": str(wrapper), "wrapper_sha256": hashlib.sha256(Path(wrapper).read_bytes()).hexdigest(),
            "reviewer": "/root/consequence_runtime_resource_review", "cpu_only": True,
            "real_model_GT_images_manifest_metric_cache_read": False,
            "gpu_or_scoring_calls": 0, "fixtures": results,
            "status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL",
            "self_rusage": {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                            "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wrapper", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.wrapper)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "fixtures": [{"name": x["name"], "status": x["status"]} for x in result["fixtures"]]}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
