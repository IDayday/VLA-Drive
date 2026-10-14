"""Bounded spatial reader: checksum each immutable version once, not each epoch."""
from collections import OrderedDict
from pathlib import Path
import os
import time
import torch
from iqe.contracts import require
from . import context


def version(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class VerifiedSpatialReader(context.SpatialCache):
    def __init__(self, config, capacity=64, verification_capacity=32768):
        require(0 < capacity <= 128 and capacity <= verification_capacity <= 65536, 'bounded spatial reader memory/metadata')
        self.execution_mode = os.environ.get('IQE_SPATIAL_READER_MODE', 'legacy')
        require(self.execution_mode in ('legacy', 'mmap_v1'), 'unsupported spatial reader execution mode')
        if self.execution_mode == 'mmap_v1':
            capacity = int(os.environ.get('IQE_SPATIAL_MMAP_RECORDS', '64'))
            require(1 <= capacity <= verification_capacity, 'bounded spatial mmap record count')
        super().__init__(config, capacity)
        self.verification_capacity = verification_capacity
        self.verified = OrderedDict()
        self.binding_file = config.root/'BASE_BINDING.json'
        self.binding_version = version(self.binding_file)
        self.code_version = version(context.__file__)
        self.statistics = {'checksum_verifications': 0, 'version_reuses': 0, 'tensor_bytes_hashed': 0,
                           'execution_mode': self.execution_mode, 'record_capacity': capacity,
                           'read_seconds': 0., 'batch_seconds': 0., 'record_loads': 0, 'record_hits': 0}

    def valid(self, scene):
        require(version(self.binding_file) == self.binding_version and version(context.__file__) == self.code_version,
                'frozen context binding/code changed during training')
        path = self.path(scene)
        receipt = Path(str(path)+'.COMPLETE.json')
        if not receipt.exists():
            return False
        current = (version(path), version(receipt), scene.observation_hash)
        previous = self.verified.get(scene.scene_id)
        if previous is not None:
            require(current == previous, 'immutable spatial record changed during training')
            self.verified.move_to_end(scene.scene_id)
            self.statistics['version_reuses'] += 1
            return True
        valid = super().valid(scene)  # actual key, receipt and full tensor SHA256
        require(current == (version(path), version(receipt), scene.observation_hash), 'spatial record changed during verification')
        if valid:
            self.verified[scene.scene_id] = current
            self.statistics['checksum_verifications'] += 1
            self.statistics['tensor_bytes_hashed'] += path.stat().st_size
            while len(self.verified) > self.verification_capacity:
                self.verified.popitem(last=False)
        return valid

    def read(self, scene):
        # Even a RAM hit rejects mutation of the immutable backing identity.
        start = time.perf_counter()
        require(self.valid(scene), 'missing spatial context '+scene.scene_id)
        if self.execution_mode == 'legacy':
            result = super().read(scene)
        elif scene.scene_id in self.loaded:
            self.loaded.move_to_end(scene.scene_id)
            self.statistics['record_hits'] += 1
            result = self.loaded[scene.scene_id]
        else:
            # The immutable torch archive remains the backing storage. Mapped
            # pages can be shared by S10/S11 through the OS page cache; no second
            # full private 288GB tensor copy is retained by each process.
            result = torch.load(self.path(scene), map_location='cpu', weights_only=True, mmap=True)
            self.loaded[scene.scene_id] = result
            self.statistics['record_loads'] += 1
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.statistics['read_seconds'] += time.perf_counter()-start
        return result

    def batch(self, scenes, device):
        if self.execution_mode == 'legacy':
            return super().batch(scenes, device)
        start = time.perf_counter()
        rows = [self.read(scene) for scene in scenes]
        keys = ('visual', 'visual_valid', 'visual_xy')
        maxn = max(row['visual'].shape[1] for row in rows)
        if all(row['visual'].shape[1] == maxn for row in rows):
            # Identical stack and transfer as reference, without an unnecessary
            # full-size zero allocation/copy for each already complete record.
            result = {key: torch.stack([row[key] for row in rows]).to(device) for key in keys}
        else:
            padded = []
            for row in rows:
                n = row['visual'].shape[1]
                values = {}
                for key in keys:
                    shape = (row[key].shape[0], maxn, *row[key].shape[2:])
                    value = row[key].new_zeros(shape)
                    value[:, :n] = row[key]
                    values[key] = value
                padded.append(values)
            result = {key: torch.stack([row[key] for row in padded]).to(device) for key in keys}
        self.statistics['batch_seconds'] += time.perf_counter()-start
        return result
