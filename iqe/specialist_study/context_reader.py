"""Bounded spatial reader: checksum each immutable version once, not each epoch."""
from collections import OrderedDict
from pathlib import Path
from iqe.contracts import require
from . import context


def version(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class VerifiedSpatialReader(context.SpatialCache):
    def __init__(self, config, capacity=64, verification_capacity=32768):
        require(0 < capacity <= 128 and capacity <= verification_capacity <= 65536, 'bounded spatial reader memory/metadata')
        super().__init__(config, capacity)
        self.verification_capacity = verification_capacity
        self.verified = OrderedDict()
        self.binding_file = config.root/'BASE_BINDING.json'
        self.binding_version = version(self.binding_file)
        self.code_version = version(context.__file__)
        self.statistics = {'checksum_verifications': 0, 'version_reuses': 0, 'tensor_bytes_hashed': 0}

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
        require(self.valid(scene), 'missing spatial context '+scene.scene_id)
        return super().read(scene)
