"""Four independent target fields; deployment continues to use CurrentDataset."""
import json
from pathlib import Path
from .foresight_dataset import ForesightTrainingDataset
from .dino_foresight_dataset import DINOTrainingDataset
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from starVLA.model.modules.foresight.tradeoff import CANDIDATES


class FullForesightDataset(DINOTrainingDataset):
    def __init__(self, root, *, candidate, current, future, dino_root=None,
                 dino_index=None, expected_dino=None, allow_partial=False,
                 image_root=None, **kwargs):
        ForesightTrainingDataset.__init__(self, root, **kwargs)
        self.local_image_root = image_root
        if image_root:
            stage = json.loads((Path(root)/'local_stage.json').read_text())
            if stage['source_identity'] != self.identity['identity'] or Path(stage['image_root']) != Path(image_root):
                raise ValueError('Foreign local image staging')
            if not allow_partial and stage['scenes'] != len(self.index):
                raise ValueError('Formal training requires the whole locally staged scene population')
        self.dino_identity = None
        self.dino_current, self.dino_future = current, future
        if not (current or future):
            if dino_root:
                raise ValueError('Unexpected visual targets for a disabled ablation')
            return
        self.dino_root = Path(dino_root)
        idx = Path(dino_index)
        cache = json.loads((self.dino_root/'identity.json').read_text())
        index = json.loads((idx/'identity.json').read_text())
        if cache['schema'] != 'ddp_full_dino_cache_v1' or cache['identity'] != expected_dino or identity_hash({k:v for k,v in cache.items() if k != 'identity'}) != expected_dino:
            raise ValueError('Wrong full-method cache schema or identity')
        c = CANDIDATES[candidate]
        if cache['candidate'] != c.record() or cache['index'] != index['identity'] or cache['grid_hw'] != list(c.grid_hw):
            raise ValueError('Wrong teacher resolution, pool or temporal index')
        if index['schema'] != 'ddp_full_dino_index_v1' or index['horizons_s'] != [0., 1., 2., 4.] or index['views'] != ['CAM_F0', 'CAM_L0', 'CAM_R0']:
            raise ValueError('Wrong physical horizon/view contract')
        split = self.identity['split']; path = idx/(split+'_scenes.json')
        if index['index_hashes'][split] != self.identity['index_sha256'] or index['partition_sha256'] != self.identity['partition_sha256'] or file_sha256(path) != index['files'][split]:
            raise ValueError('Scene, log split or time index changed')
        if not allow_partial:
            done = json.loads((self.dino_root/'COMPLETE.json').read_text())
            if done['identity'] != expected_dino or done['images'] != cache['image_count']:
                raise ValueError('Incomplete full target cache')
        if self.interaction_identity:
            ident = self.interaction_identity
            if identity_hash({k:v for k,v in ident.items() if k != 'identity'}) != ident['identity'] or ident['teacher_completed_updates'] <= 0:
                raise ValueError('Invalid frozen, trained MAE identity')
            if ident['query_order_s'] != [.5*(i+1) for i in range(8)]:
                raise ValueError('MAE time queries mismatch')
        self.dino_scenes = json.loads(path.read_text())
        if [r['token'] for r in self.dino_scenes] != [r['token'] for r in self.index]:
            raise ValueError('Auxiliary labels cannot change the scene population')
        self.dino_identity = cache
        self._checked_chunks = set()

    # The parent fetches CURRENT observations first, then four explicit target
    # fields. Its time selection is [0] and [1,2,3], never a mixed random draw.
