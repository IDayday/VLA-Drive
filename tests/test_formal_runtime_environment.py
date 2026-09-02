import json
from pathlib import Path

from scripts.audit_formal_runtime_environment import compare_fingerprints


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_runtime_fingerprint_compare_accepts_identical_nodes(tmp_path):
    left = tmp_path / "node0.json"
    right = tmp_path / "node1.json"
    payload = {
        "python": {"version": "3.9.25", "executable": "/shared/python"},
        "packages": {"torch": {"version": "2.5.1"}},
        "core_file_sha256": {"action_decoder.py": "abc"},
    }
    _write(left, payload)
    _write(right, payload)

    report = compare_fingerprints(left, right)

    assert report["equal"] is True
    assert report["left_sha256"] == report["right_sha256"]


def test_runtime_fingerprint_compare_rejects_host_local_package_drift(tmp_path):
    left = tmp_path / "node0.json"
    right = tmp_path / "node1.json"
    _write(
        left,
        {
            "python": {"version": "3.9.25"},
            "packages": {"transformers": {"version": "4.57.6"}},
        },
    )
    _write(
        right,
        {
            "python": {"version": "3.10.20"},
            "packages": {"transformers": {"version": "4.57.0"}},
        },
    )

    report = compare_fingerprints(left, right)

    assert report["equal"] is False
    assert report["left_sha256"] != report["right_sha256"]
    assert report["differing_top_level_fields"] == ["packages", "python"]


def test_formal_launchers_require_shared_runtime_audit():
    repo_root = Path(__file__).resolve().parents[1]
    for relative in (
        "local_planreg_wm_v1/benchmark_formal_common.sh",
        "local_planreg_wm_v1/formal_launch_common.sh",
    ):
        source = (repo_root / relative).read_text(encoding="utf-8")
        assert "source \"${PLANREG_SCRIPT_DIR}/formal_runtime.sh\"" in source
        assert "planreg_formal_runtime_setup" in source
        assert "planreg_formal_runtime_audit_local" in source
        assert "planreg_formal_runtime_audit_remote" in source
        assert "planreg_formal_runtime_compare" in source
        assert '"HF_HOME=${HF_HOME}"' in source
        assert '"TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"' in source

    for relative in (
        "local_planreg_wm_v1/smoke_formal_real_data.sh",
        "local_planreg_wm_v1/evaluate_formal_checkpoint.sh",
    ):
        single_node = (repo_root / relative).read_text(encoding="utf-8")
        assert 'source "${script_dir}/formal_runtime.sh"' in single_node
        assert "planreg_formal_runtime_setup" in single_node
        assert "planreg_formal_runtime_audit_local" in single_node
