"""Independent tiny CPU cases; never calls plan, supervisor, or live storage."""
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest


HERE = Path(__file__).parent
SOURCE = Path(os.environ.get("ST_PLAN_REVIEW_SOURCE", "/mnt/project/VLA-Drive-st-full-future-plan-control-20261017/reports/st_world_policy/20261017/full_future_plan_control_v1/plan_once.py"))


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("independent_plan_review", SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def controls(module, role):
    value = json.loads((HERE / "DEV_PLAN_ADMISSION.json").read_text())
    value["scope"]["role"] = role
    value["scope"]["exact_scenes"] = 1696 if role == "dev" else 101592
    value["status"] = "PASS_BOUNDED_" + role.upper() + "_PLAN_ONLY"
    value["output_root"] = str(module.BASE / "caches" / ("full_future_metadata_" + role + "_v1"))
    now = time.time()
    value.update(issued_at_unix=now-1, latest_start_unix=now+900, valid_until_unix=now+1200)
    argv = json.loads((HERE / ("DEV_PLAN_ARGV_V2.json" if role == "dev" else "TRAIN_PLAN_ARGV.json")).read_text())
    return value, argv, now


@pytest.mark.parametrize("role", ["dev", "train"])
def test_exact_role_argv(module, role):
    admission, argv, now = controls(module, role)
    assert module.validate(admission, argv, now) == Path(admission["output_root"])


def test_train_does_not_accept_dev_build_limits(module):
    admission, argv, now = controls(module, "train")
    argv[argv.index("--max-seconds")+1] = "1800"
    with pytest.raises(ValueError, match="limits changed"):
        module.validate(admission, argv, now)


def test_root_directory_and_allocated_bytes(module, tmp_path):
    child = tmp_path / "control"
    child.mkdir()
    (child / "tiny.json").write_bytes(b"0")
    paths = [tmp_path, child, child / "tiny.json"]
    expected = sum(max(p.stat().st_size, p.stat().st_blocks*512) for p in paths)
    assert module.root_bytes(tmp_path) == expected


def test_allocated_bytes_trigger_cap(module, tmp_path, monkeypatch):
    p = tmp_path / "tiny.json"
    p.write_bytes(b"0")
    actual = sum(max(x.stat().st_size, x.stat().st_blocks*512) for x in [tmp_path, p])
    monkeypatch.setitem(module.LIMITS, "root_inclusive_bytes_including_control_temp_logs", actual-1)
    with pytest.raises(ValueError, match="output limit"):
        module.root_bytes(tmp_path)


def test_symlink_root_child_rejected(module, tmp_path):
    p = tmp_path / "tiny.json"
    p.write_bytes(b"0")
    (tmp_path / "alias.json").symlink_to(p)
    with pytest.raises(ValueError):
        module.root_bytes(tmp_path)


def test_short_cleanup_window_rejected(module):
    admission, argv, now = controls(module, "dev")
    admission["valid_until_unix"] = now+219
    with pytest.raises(ValueError, match="window"):
        module.validate(admission, argv, now)
