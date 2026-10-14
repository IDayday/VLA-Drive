"""Small immutable current-camera sidecars, verified once instead of per epoch.

No image decoding, VLM, score or target access. Partial log shards are resumable;
only complete coverage produces the final manifest. This is not a visual cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import torch
from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.config import StudyConfig
from .calibration import CalibrationReader

ROLES = ('scorer_fit', 'stage_val', 'selector_cal', 'dev_report')


def version(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def scene_identity(scene):
    return [scene.scene_id, scene.source_log_id, scene.observation_hash, scene.observation_ref]


def prepare(source, output, workers=4):
    require(1 <= workers <= 4, 'bounded shared IO: at most four calibration workers')
    root = Path(output)
    roles = scenes_by_role(source)
    scenes = [s for role in ROLES for s in roles[role]]
    require(len({s.scene_id for s in scenes}) == len(scenes), 'calibration role overlap')
    identity = dict(schema='verified_current_calibration_bank_v1',
        Base_binding_hash=file_hash(source.root/'BASE_BINDING.json'),
        source_config_identity=source.identity,
        roles={r:digest([scene_identity(s) for s in roles[r]]) for r in ROLES},
        code=file_hash(__file__), reader_code=file_hash(Path(__file__).with_name('calibration.py')),
        labels_read=False, image_decode_calls=0, current_images_hashed=True)
    with lock(root/'BUILD.lock'):
        atomic_json(root/'IDENTITY.json', identity, immutable=True)
        groups = {}
        for s in scenes: groups.setdefault(s.source_log_id, []).append(s)
        def one(item):
            log, rows = item
            key = digest([log, [scene_identity(s) for s in rows]])
            path = root/'logs'/(key+'.json')
            if path.exists():
                data = read_json(path)
                require(data['key'] == key and set(data['records']) == {s.scene_id for s in rows},
                        'partial calibration shard conflict')
            else:
                reader = CalibrationReader(min(1024, len(rows)))
                records = {}
                for s in rows:
                    tensors = reader.read(s)  # actual observation plus image content hashes
                    observation = read_json(s.observation_ref)
                    records[s.scene_id] = dict(identity=scene_identity(s),
                        observation_file_hash=file_hash(s.observation_ref),
                        images={p:dict(checksum=file_hash(p), version=version(p))
                                for p in observation['image_paths']},
                        tensors={k:v.tolist() for k,v in tensors.items()})
                atomic_json(path, dict(key=key, records=records), immutable=True)
                data = read_json(path)
            return dict(path=str(path.resolve()), checksum=file_hash(path), scenes=list(data['records']))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            shards = list(pool.map(one, groups.items()))
        covered = [s for row in shards for s in row['scenes']]
        require(len(covered) == len(scenes) and set(covered) == {s.scene_id for s in scenes},
                'calibration coverage incomplete')
        manifest = dict(identity=identity, status='COMPLETE', records=len(scenes),
            role_counts={r:len(roles[r]) for r in ROLES}, shards=shards,
            total_bytes=sum(Path(s['path']).stat().st_size for s in shards))
        atomic_json(root/'MANIFEST.json', manifest, immutable=True)
        return {k:v for k,v in manifest.items() if k not in ('identity', 'shards')}


class PreparedCalibrationReader:
    def __init__(self, source, root):
        self.path = Path(root)/'MANIFEST.json'
        manifest = read_json(self.path)
        require(manifest['status'] == 'COMPLETE' and
                manifest['identity']['Base_binding_hash'] == file_hash(source.root/'BASE_BINDING.json') and
                manifest['identity']['source_config_identity'] == source.identity and
                manifest['identity']['code'] == file_hash(__file__), 'calibration bank/source mismatch')
        self.manifest_version = version(self.path)
        self.records, self.tensors, self.verified = {}, {}, {}
        for shard in manifest['shards']:
            require(file_hash(shard['path']) == shard['checksum'], 'corrupt calibration shard')
            records = read_json(shard['path'])['records']
            require(not self.records.keys() & records.keys(), 'duplicate calibration record')
            self.records.update(records)
        require(len(self.records) == manifest['records'], 'incomplete calibration manifest')

    def read(self, scene):
        require(version(self.path) == self.manifest_version, 'calibration manifest changed')
        record = self.records[scene.scene_id]
        require(record['identity'] == scene_identity(scene), 'calibration scene/observation mismatch')
        current = [version(scene.observation_ref)] + [version(p) for p in record['images']]
        if self.verified.get(scene.scene_id) != current:
            require(file_hash(scene.observation_ref) == record['observation_file_hash'],
                    'current calibration observation content changed')
            for p, item in record['images'].items():
                # Builder already hashed original immutable versions. A changed
                # mount/version must prove the same bytes, not merely existence.
                if version(p) != item['version']:
                    require(file_hash(p) == item['checksum'], 'calibration image changed')
            self.verified[scene.scene_id] = current
        if scene.scene_id not in self.tensors:
            self.tensors[scene.scene_id] = {k:torch.tensor(v, dtype=torch.float32)
                                           for k,v in record['tensors'].items()}
        return self.tensors[scene.scene_id]

    def batch(self, scenes, device):
        rows = [self.read(s) for s in scenes]
        return {k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-campaign', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    source = StudyConfig(**read_json(Path(a.source_campaign)/'resolved_config.json'))
    print(json.dumps(prepare(source, a.output, a.workers), allow_nan=False))


if __name__ == '__main__': main()
