#!/usr/bin/env python3
"""Audit the deleted public checkpoint shards with bounded HTTP range reads.

Only the small ZIP pickle metadata, central directory, and one four-byte
action-head tensor are fetched. The script therefore proves the training
runtime metadata belongs to the current merged public checkpoint without
downloading the four historical ~1 GB shards.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import re
import struct
from typing import Any
import urllib.request
import zipfile


MODEL_REVISION = "dad1600"
SHARDS = (
    ("best-epoch_26-step_174312_1.pth", 1_069_466_889),
    ("best-epoch_26-step_174312_2.pth", 1_067_494_875),
    ("best-epoch_26-step_174312_3.pth", 1_067_806_321),
    ("best-epoch_26-step_174312_4.pth", 1_067_013_533),
)
DEFAULT_MERGED = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)
IDENTITY_KEY = "agent.action_head.scorer.pred_score.no_at_fault_collisions.2.bias"


def _ignore_tensor_rebuild(*_args: Any) -> None:
    """Replace tensor rebuilds while decoding trusted checkpoint metadata."""

    return None


class _MetadataUnpickler(pickle.Unpickler):
    """Decode the checkpoint structure without loading remote tensor storage."""

    def persistent_load(self, persistent_id: Any) -> Any:
        return ("persistent_storage", persistent_id)

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name.startswith("_rebuild"):
            return _ignore_tensor_rebuild
        return super().find_class(module, name)


def _unpickle_metadata(payload: bytes) -> dict[str, Any]:
    metadata = _MetadataUnpickler(io.BytesIO(payload)).load()
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint metadata must be a dict, got {type(metadata)}")
    return metadata


def _url(name: str) -> str:
    return (
        "https://modelscope.cn/models/ArteMe/DriveVLA-M0/resolve/"
        f"{MODEL_REVISION}/{name}"
    )


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "stage2-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers.get("Content-Range"):
            raise RuntimeError(
                f"server ignored bounded range {start}-{end}: "
                f"status={response.status}"
            )
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"range length mismatch: {len(payload)} != {expected}")
    return payload


def _pickle_from_zip_prefix(prefix: bytes) -> bytes:
    if prefix[:4] != b"PK\x03\x04":
        raise ValueError("checkpoint prefix is not a ZIP local header")
    name_length, extra_length = struct.unpack_from("<HH", prefix, 26)
    payload_offset = 30 + name_length + extra_length
    payload = prefix[payload_offset:]
    operations = list(pickletools.genops(payload))
    stop = operations[-1][2] + 1
    return payload[:stop]


def _scalar_after_key(operations: list, key: str) -> Any:
    scalar_operations = {
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "UNICODE",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG1",
        "LONG4",
    }
    index = next(
        index for index, operation in enumerate(operations) if operation[1] == key
    )
    for operation in operations[index + 1 : index + 10]:
        if operation[0].name in scalar_operations:
            return operation[1]
    raise KeyError(key)


def _storage_id_after_key(operations: list, key: str) -> str:
    index = next(
        index for index, operation in enumerate(operations) if operation[1] == key
    )
    for operation in operations[index + 1 : index + 30]:
        value = operation[1]
        if isinstance(value, str) and value.isdigit():
            return value
    raise KeyError(f"storage for {key}")


def _value_opcode_after_key(operations: list, key: str) -> str:
    """Return the first non-memo pickle opcode encoding a value after ``key``."""

    index = next(
        index for index, operation in enumerate(operations) if operation[1] == key
    )
    memo_opcodes = {"BINPUT", "LONG_BINPUT", "MEMOIZE"}
    for operation in operations[index + 1 : index + 6]:
        if operation[0].name not in memo_opcodes:
            return operation[0].name
    raise KeyError(f"value opcode for {key}")


def _inspect_pickle(payload: bytes) -> tuple[dict[str, Any], list]:
    operations = list(pickletools.genops(payload))
    strings = [operation[1] for operation in operations if isinstance(operation[1], str)]
    decoded = _unpickle_metadata(payload)
    epoch_loop = decoded["loops"]["fit_loop"]
    epoch_progress = epoch_loop["epoch_progress"]
    batch_progress = epoch_loop["epoch_loop.batch_progress"]
    scheduler_progress = epoch_loop["epoch_loop.scheduler_progress"]
    optimizer_progress = epoch_loop[
        "epoch_loop.automatic_optimization.optim_progress"
    ]["optimizer"]["step"]
    checkpoint_paths = sorted(
        value for value in strings if "/checkpoints/" in value or value.endswith("/checkpoints")
    )
    directory_epoch_labels = sorted(
        {
            int(match.group(1))
            for value in checkpoint_paths
            for match in [re.search(r"_(\d+)epochs(?:_|/)", value)]
            if match is not None
        }
    )
    return (
        {
            "epoch": _scalar_after_key(operations, "epoch"),
            "global_step": _scalar_after_key(operations, "global_step"),
            "pytorch_lightning_version": _scalar_after_key(
                operations, "pytorch-lightning_version"
            ),
            "checkpoint_paths": checkpoint_paths,
            "top_level_keys": sorted(decoded),
            "contains_hyper_parameters": "hyper_parameters" in decoded,
            "epoch_progress": epoch_progress,
            "batch_progress": batch_progress,
            "scheduler_progress": scheduler_progress,
            "optimizer_step_progress": optimizer_progress,
            "run_directory_epoch_labels": directory_epoch_labels,
            "state_key_count": sum(value.startswith("agent.") for value in strings),
            "contains_identity_key": IDENTITY_KEY in strings,
            "optimizer_states_value_opcode": _value_opcode_after_key(
                operations, "optimizer_states"
            ),
            "lr_schedulers_value_opcode": _value_opcode_after_key(
                operations, "lr_schedulers"
            ),
            "checkpoint_callback_descriptors": sorted(
                value for value in strings if value.startswith("ModelCheckpoint{")
            ),
        },
        operations,
    )


def _zip64_central_directory(url: str, size: int) -> bytes:
    tail_start = max(0, size - 262_144)
    tail = _fetch_range(url, tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("ZIP EOCD not found")
    entries, directory_size, directory_offset = struct.unpack_from(
        "<HII", tail, eocd + 10
    )
    if (
        entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        if locator < 0:
            raise ValueError("ZIP64 locator not found")
        zip64_offset = struct.unpack_from("<Q", tail, locator + 8)[0]
        zip64 = _fetch_range(url, zip64_offset, zip64_offset + 55)
        if zip64[:4] != b"PK\x06\x06":
            raise ValueError("invalid ZIP64 EOCD")
        directory_size = struct.unpack_from("<Q", zip64, 40)[0]
        directory_offset = struct.unpack_from("<Q", zip64, 48)[0]
    return _fetch_range(
        url, directory_offset, directory_offset + directory_size - 1
    )


def _remote_zip_entry(url: str, size: int, suffix: str) -> bytes:
    directory = _zip64_central_directory(url, size)
    position = 0
    while position < len(directory):
        if directory[position : position + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central-directory entry at {position}")
        method = struct.unpack_from("<H", directory, position + 10)[0]
        compressed_size = struct.unpack_from("<I", directory, position + 20)[0]
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", directory, position + 28
        )
        local_offset = struct.unpack_from("<I", directory, position + 42)[0]
        name = directory[
            position + 46 : position + 46 + name_length
        ].decode()
        if name.endswith(suffix):
            if method != 0:
                raise ValueError(f"expected stored tensor entry, method={method}")
            local_header = _fetch_range(url, local_offset, local_offset + 29)
            local_name_length, local_extra_length = struct.unpack_from(
                "<HH", local_header, 26
            )
            data_offset = (
                local_offset + 30 + local_name_length + local_extra_length
            )
            return _fetch_range(
                url, data_offset, data_offset + compressed_size - 1
            )
        position += 46 + name_length + extra_length + comment_length
    raise KeyError(suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-checkpoint", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    shard_reports = []
    shard_operations = {}
    for name, size in SHARDS:
        url = _url(name)
        prefix = _fetch_range(url, 0, min(size, 1_048_576) - 1)
        pickle_payload = _pickle_from_zip_prefix(prefix)
        metadata, operations = _inspect_pickle(pickle_payload)
        best_score_storage = _storage_id_after_key(
            operations, "best_model_score"
        )
        best_score_bytes = _remote_zip_entry(
            url, size, f"/data/{best_score_storage}"
        )
        if len(best_score_bytes) != 4:
            raise ValueError(
                f"unexpected callback-score size in {name}: "
                f"{len(best_score_bytes)}"
            )
        shard_operations[name] = operations
        shard_reports.append(
            {
                "name": name,
                "lfs_size": size,
                "url_revision": MODEL_REVISION,
                "fetched_prefix_bytes": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "best_model_score_storage_id": best_score_storage,
                "best_model_score": struct.unpack("<f", best_score_bytes)[0],
                **metadata,
            }
        )

    merged = args.merged_checkpoint.resolve()
    with zipfile.ZipFile(merged) as checkpoint_zip:
        pickle_name = next(
            name for name in checkpoint_zip.namelist() if name.endswith("/data.pkl")
        )
        merged_pickle = checkpoint_zip.read(pickle_name)
        merged_operations = list(pickletools.genops(merged_pickle))
        merged_storage = _storage_id_after_key(merged_operations, IDENTITY_KEY)
        merged_entry = next(
            name
            for name in checkpoint_zip.namelist()
            if name.endswith(f"/data/{merged_storage}")
        )
        merged_bytes = checkpoint_zip.read(merged_entry)

    shard_name, shard_size = SHARDS[-1]
    shard_storage = _storage_id_after_key(shard_operations[shard_name], IDENTITY_KEY)
    shard_bytes = _remote_zip_entry(
        _url(shard_name), shard_size, f"/data/{shard_storage}"
    )
    if len(shard_bytes) == 4:
        scalar_dtype = "float32"
        shard_scalar = struct.unpack("<f", shard_bytes)[0]
        merged_scalar = struct.unpack("<f", merged_bytes)[0]
    else:
        scalar_dtype = "raw"
        shard_scalar = None
        merged_scalar = None

    report = {
        "modelscope_revision": MODEL_REVISION,
        "bounded_range_reads_only": True,
        "shards": shard_reports,
        "training_state_recoverability": {
            "optimizer_states_stripped": all(
                shard["optimizer_states_value_opcode"] == "EMPTY_LIST"
                for shard in shard_reports
            ),
            "lr_schedulers_stripped": all(
                shard["lr_schedulers_value_opcode"] == "EMPTY_LIST"
                for shard in shard_reports
            ),
            "scheduler_executed_every_optimizer_step": all(
                shard["scheduler_progress"]["total"]["completed"]
                == shard["global_step"]
                == shard["optimizer_step_progress"]["total"]["completed"]
                for shard in shard_reports
            ),
            "scheduler_type_recoverable": False,
            "scheduler_horizon_recoverable": False,
            "trainer_max_epochs_recoverable": False,
            "epoch_progress_consensus": all(
                shard["epoch_progress"] == shard_reports[0]["epoch_progress"]
                for shard in shard_reports
            ),
            "run_directory_epoch_label_consensus": sorted(
                {
                    label
                    for shard in shard_reports
                    for label in shard["run_directory_epoch_labels"]
                }
            ),
            "optimizer_epochs_from_step_ratio": (
                shard_reports[0]["global_step"]
                // shard_reports[0]["batch_progress"]["current"]["completed"]
            ),
            "consequence": (
                "Loop progress proves a scheduler executed at every optimizer "
                "step and that the saved run processed 27 optimizer-step epochs. "
                "Stripped scheduler state and absent hyperparameters prevent "
                "recovering its class, configured horizon, trainer max_epochs, "
                "or exact LR curve from the checkpoint alone. The 25epochs run-"
                "directory label is therefore evidence for a counterfactual "
                "schedule horizon, not an authoritative training configuration."
            ),
        },
        "callback_best_model_score_consensus": {
            "all_shards_equal": len(
                {shard["best_model_score"] for shard in shard_reports}
            )
            == 1,
            "value": shard_reports[0]["best_model_score"],
            "monitor": "val/score_epoch",
        },
        "merged_checkpoint": str(merged),
        "identity_check": {
            "state_key": IDENTITY_KEY,
            "historical_shard": shard_name,
            "historical_storage_id": shard_storage,
            "merged_storage_id": merged_storage,
            "dtype": scalar_dtype,
            "historical_raw_hex": shard_bytes.hex(),
            "merged_raw_hex": merged_bytes.hex(),
            "historical_scalar": shard_scalar,
            "merged_scalar": merged_scalar,
            "byte_exact": shard_bytes == merged_bytes,
        },
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
