from pathlib import Path

from starVLA.dataloader.navsim_dataset import resolve_navsim_data_path


def test_relocates_trainval_all_sensor_layout(monkeypatch):
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", "/runtime/navsim")
    monkeypatch.setenv(
        "NAVSIM_TRAINVAL_SENSOR_ROOT", "/runtime/trainval-sensors"
    )
    embedded = (
        "/preprocess/checkout/navsim_dataset_raw/trainval_all/"
        "trainval_sensor_blobs/trainval/log-token/CAM_F0/frame.jpg"
    )

    assert resolve_navsim_data_path(embedded) == str(
        Path("/runtime/trainval-sensors/log-token/CAM_F0/frame.jpg")
    )


def test_relocates_legacy_sensor_layout(monkeypatch):
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", "/runtime/navsim")
    monkeypatch.setenv(
        "NAVSIM_TRAINVAL_SENSOR_ROOT", "/runtime/trainval-sensors"
    )
    embedded = (
        "/preprocess/checkout/navsim_dataset_raw/sensor_blobs/trainval/"
        "log-token/CAM_L0/frame.jpg"
    )

    assert resolve_navsim_data_path(embedded) == str(
        Path("/runtime/trainval-sensors/log-token/CAM_L0/frame.jpg")
    )


def test_unrelated_absolute_path_is_unchanged(monkeypatch):
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", "/runtime/navsim")
    monkeypatch.setenv(
        "NAVSIM_TRAINVAL_SENSOR_ROOT", "/runtime/trainval-sensors"
    )
    embedded = "/another/dataset/CAM_R0/frame.jpg"

    assert resolve_navsim_data_path(embedded) == embedded
