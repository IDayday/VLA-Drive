"""Pinned current/history observations without any target or teacher-cache reads."""
import json
from pathlib import Path

from torch.utils.data import Dataset

from .foresight_dataset import current_observation
from .st_world_schema import ModelInputs, identity_hash
from .st_world_history import HistoryReader, file_sha256


class STObservationDataset(Dataset):
    def __init__(self, root, manifest, *, history_root=None):
        self.root = Path(root)
        self.source = json.loads((self.root / "identity.json").read_text())
        index = json.loads((self.root / "index.json").read_text())
        selection = json.loads(Path(manifest).read_text())
        if (self.source.get("schema") != "ddpolicy_current_cameras_v1"
                or self.source["split"] not in ("train", "dev")
                or identity_hash(index) != self.source["index_sha256"]
                or identity_hash({k: v for k, v in self.source.items() if k != "identity"}) != self.source["identity"]):
            raise ValueError("Only the registered train/dev current observation source is permitted")
        if (selection.get("schema") != "st_fixed_scene_list_v1"
                or selection["source_index_hash"] != self.source["index_sha256"]
                or selection["split"] != self.source["split"]):
            raise ValueError("Observation selection differs from the original data role")
        self.rows = selection["scenes"]
        lookup = {r["token"]: r for r in index}
        if not self.rows or len({r["token"] for r in self.rows}) != len(self.rows):
            raise ValueError("Empty or duplicated observation population")
        self.images = {}
        for row in self.rows:
            if row["token"] not in lookup or lookup[row["token"]]["log"] != row["log"]:
                raise ValueError("Observation token/log is outside the declared source")
            path = self.root / "current" / (row["token"] + ".json")
            if file_sha256(path) != row["current_record_sha256"]:
                raise ValueError("Current observation changed after fixed population registration")
            record = json.loads(path.read_text())
            if len(record["image_paths"]) != 3:
                raise ValueError("Original three-view input contract changed")
            self.images[row["token"]] = [(p, file_sha256(Path(p))) for p in record["image_paths"]]
        self.history = None
        if history_root:
            history_identity = json.loads((Path(history_root) / "HISTORY_MANIFEST.json").read_text())
            if history_identity["source_student_identity"] != self.source["identity"]:
                raise ValueError("History/current data sources differ")
            self.history = HistoryReader(history_root, history_identity["identity"])
            if any(r["token"] not in self.history.index for r in self.rows):
                raise ValueError("Fixed observation population lacks registered history records")
        self.identity = {"schema": "st_observation_only_v1", "current_source": self.source["identity"],
            "manifest_sha256": file_sha256(Path(manifest)), "split": self.source["split"],
            "image_pins": self.images, "history": self.history.manifest["identity"] if self.history else None,
            "targets_read": False, "teacher_cache_read": False}
        self.identity["identity"] = identity_hash(self.identity)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / "current" / (row["token"] + ".json")
        if file_sha256(path) != row["current_record_sha256"]:
            raise ValueError("Current observation content changed during inference")
        record = json.loads(path.read_text())
        if (record["identity"] != self.source["identity"] or record["token"] != row["token"]
                or record["log"] != row["log"]):
            raise ValueError("Foreign current observation")
        for path, expected in self.images[row["token"]]:
            if file_sha256(Path(path)) != expected:
                raise ValueError("Image content changed after observation registration")
        return ModelInputs(current_observation(record),
            self.history.read(row["token"], current_record=record) if self.history else None)
