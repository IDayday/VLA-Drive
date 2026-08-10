import csv
from pathlib import Path

from tools.field2plan.record_pdms_result import parse_pdms_csv, record_result


def _write_score_csv(path: Path, average: float = 0.9) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["token", "valid", "score"]
        )
        writer.writeheader()
        writer.writerow({"token": "a", "valid": "True", "score": "1.0"})
        writer.writerow({"token": "b", "valid": "False", "score": "0.0"})
        writer.writerow(
            {"token": "average", "valid": "True", "score": str(average)}
        )


def test_parse_pdms_csv_excludes_average_row(tmp_path: Path) -> None:
    score_csv = tmp_path / "score.csv"
    _write_score_csv(score_csv, average=0.75)
    assert parse_pdms_csv(score_csv) == (0.75, 2, 1, 1)


def test_record_result_atomically_rebuilds_live_summaries(tmp_path: Path) -> None:
    score_csv = tmp_path / "score.csv"
    checkpoint = tmp_path / "checkpoint.pt"
    log = tmp_path / "eval.log"
    prediction_dir = tmp_path / "predictions"
    summary_root = tmp_path / "summary"
    _write_score_csv(score_csv, average=0.893)
    checkpoint.write_bytes(b"weights")
    log.write_text("complete", encoding="utf-8")
    prediction_dir.mkdir()

    record = record_result(
        summary_root=summary_root,
        status="complete",
        protocol="navsim_v1_1_pdms_ws2_seed20260808",
        experiment="p2_test",
        checkpoint_step=10000,
        inference_world_size=2,
        inference_seed=20260808,
        checkpoint=checkpoint,
        prediction_dir=prediction_dir,
        evaluator_log=log,
        result_csv=score_csv,
    )

    assert record.pdms == 0.893
    csv_text = (summary_root / "summary.csv").read_text(encoding="utf-8")
    markdown = (summary_root / "summary.md").read_text(encoding="utf-8")
    assert "0.893000000000" in csv_text
    assert "89.300000" in markdown
    assert "p2_test" in markdown
    assert (summary_root / "records" / "p2_test-step10000.json").is_file()


def test_failed_record_is_visible_without_fake_zero_score(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing.pt"
    prediction_dir = tmp_path / "predictions"
    log = tmp_path / "eval.log"
    prediction_dir.mkdir()
    log.write_text("failure", encoding="utf-8")
    summary_root = tmp_path / "summary"

    record_result(
        summary_root=summary_root,
        status="failed",
        protocol="navsim_v1_1_pdms_ws2_seed20260808",
        experiment="p2_failed",
        checkpoint_step=20000,
        inference_world_size=2,
        inference_seed=20260808,
        checkpoint=checkpoint,
        prediction_dir=prediction_dir,
        evaluator_log=log,
        error="evaluator failed",
    )

    rows = list(
        csv.DictReader(
            (summary_root / "summary.csv").open("r", encoding="utf-8")
        )
    )
    assert rows[0]["status"] == "failed"
    assert rows[0]["pdms"] == ""
    assert "FAILED" in (summary_root / "summary.md").read_text(encoding="utf-8")
