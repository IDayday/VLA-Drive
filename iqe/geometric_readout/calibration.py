"""Read-only calibration sidecar; no labels or metric caches are model inputs."""
from collections import OrderedDict
import torch
from iqe.contracts import require
from iqe.io import digest, file_hash, read_json


class CalibrationReader:
    def __init__(self, capacity=128):
        require(1 <= capacity <= 1024, "bounded calibration cache")
        self.capacity, self.loaded = capacity, OrderedDict()

    def read(self, scene):
        path = scene.observation_ref
        checksum = file_hash(path)
        if scene.scene_id in self.loaded:
            previous, value = self.loaded[scene.scene_id]
            require(previous == (checksum, scene.observation_hash), "calibration observation mutated")
            self.loaded.move_to_end(scene.scene_id)
            return value
        record = read_json(path)
        require(record['token'] == scene.scene_id and record['log'] == scene.source_log_id,
                "calibration scene/log mismatch")
        require(digest({'record': record, 'images': {p: file_hash(p) for p in record['image_paths']}})
                == scene.observation_hash, "calibration/image observation hash mismatch")
        c = record['current_calibration']
        require(set(c) == {'intrinsics', 'extrinsics', 'distortion'}, "current calibration whitelist")
        require(len(record['image_paths']) == 3, "locked three-view calibration")
        result = {'camera_intrinsics': torch.tensor(c['intrinsics'], dtype=torch.float32),
                  'camera_to_ego': torch.tensor(c['extrinsics'], dtype=torch.float32),
                  'camera_image_sizes': torch.tensor([[1024., 576.]]*3)}
        require(result['camera_intrinsics'].shape == (3, 3, 3)
                and result['camera_to_ego'].shape == (3, 4, 4)
                and all(torch.isfinite(t).all().item() for t in result.values()), "invalid calibration")
        self.loaded[scene.scene_id] = ((checksum, scene.observation_hash), result)
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return result

    def batch(self, scenes, device):
        rows = [self.read(s) for s in scenes]
        return {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}
