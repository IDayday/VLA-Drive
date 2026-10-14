"""Bounded spatial reader: checksum each immutable version once, not each epoch."""
from collections import OrderedDict
from pathlib import Path
import os
import socket
import time
import torch
from iqe.contracts import require
from iqe.io import read_json, file_hash, digest
from . import context


def version(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class VerifiedSpatialReader(context.SpatialCache):
    def __init__(self, config, capacity=64, verification_capacity=32768):
        require(0 < capacity <= 128 and capacity <= verification_capacity <= 65536, 'bounded spatial reader memory/metadata')
        self.execution_mode = os.environ.get('IQE_SPATIAL_READER_MODE', 'legacy')
        require(self.execution_mode in ('legacy', 'mmap_v1', 'ram_mmap_v1'), 'unsupported spatial reader execution mode')
        if self.execution_mode != 'legacy':
            capacity = int(os.environ.get('IQE_SPATIAL_MMAP_RECORDS', '64'))
            require(1 <= capacity <= verification_capacity, 'bounded spatial mmap record count')
        super().__init__(config, capacity)
        self.verification_capacity = verification_capacity
        self.verified = OrderedDict()
        self.binding_file = config.root/'BASE_BINDING.json'
        self.binding_version = version(self.binding_file)
        self.code_version = version(context.__file__)
        self.ram_records = None
        if self.execution_mode == 'ram_mmap_v1':
            self._bind_ram_stage()
        self.statistics = {'checksum_verifications': 0, 'version_reuses': 0, 'tensor_bytes_hashed': 0,
                           'execution_mode': self.execution_mode, 'record_capacity': capacity,
                           'read_seconds': 0., 'batch_seconds': 0., 'record_loads': 0, 'record_hits': 0}

    def _bind_ram_stage(self):
        # Import at construction, after this module has defined version(): the
        # standalone copier uses the same immutable-file version check.
        from .ram_stage import stage_root
        from .binding import scenes_by_role
        self.ram_root = stage_root(self.config)
        self.ram_manifest = self.ram_root/'MANIFEST.json'
        self.ram_manifest_version = version(self.ram_manifest)
        manifest = read_json(self.ram_manifest)
        scenes = scenes_by_role(self.config)['scorer_fit']
        ids = [scene.scene_id for scene in scenes]
        require(manifest['schema'] == 'readonly_spatial_tmpfs_v1' and manifest['status'] == 'COMPLETE'
                and manifest['host'] == socket.gethostname()
                and manifest['root'] == str(self.ram_root)
                and self.ram_root.resolve() == self.ram_root
                and manifest['config_identity'] == self.config.identity
                and manifest['Base_binding_hash'] == file_hash(self.binding_file)
                and manifest['source_role'] == 'scorer_fit'
                and manifest['source_scene_hash'] == digest(ids)
                and manifest['requested'] == len(ids)
                and set(manifest['records']) == set(ids),
                'RAM spatial manifest identity or complete coverage mismatch')
        self.ram_records = manifest['records']
        self.ram_manifest_hash = file_hash(self.ram_manifest)
        require(version(self.ram_manifest) == self.ram_manifest_version, 'RAM manifest changed while binding')

    def path(self, scene):
        if self.ram_records is not None and scene.scene_id in self.ram_records:
            return self.ram_root/(scene.scene_id+'.pt')
        # Validation/calibration/development are explicitly outside this D_S
        # RAM stage, and keep using their original immutable project records.
        return super().path(scene)

    def valid(self, scene):
        require(version(self.binding_file) == self.binding_version and version(context.__file__) == self.code_version,
                'frozen context binding/code changed during training')
        if self.ram_records is not None:
            require(version(self.ram_manifest) == self.ram_manifest_version, 'RAM spatial manifest changed during training')
            if scene.scene_id in self.ram_records:
                source = super().path(scene)
                record = self.ram_records[scene.scene_id]
                require(list(version(source)) == record['source_version']
                        and list(version(str(source)+'.COMPLETE.json')) == record['source_receipt_version'],
                        'canonical spatial source changed after RAM staging')
                require(self.path(scene).is_file(), 'missing staged spatial context '+scene.scene_id)
                if scene.scene_id not in self.verified:
                    staged_receipt = read_json(str(self.path(scene))+'.COMPLETE.json')
                    require(staged_receipt['checksum'] == record['checksum']
                            and staged_receipt['scene_id'] == scene.scene_id,
                            'staged spatial checksum differs from verified canonical source')
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
