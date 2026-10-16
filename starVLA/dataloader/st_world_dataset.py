"""Thin ST dataset over the original current observation and ego IL data.

No model/teacher initialization and no change to the original current whitelist.
Scene joining uses immutable token/log/content identities, never CSV position.
"""
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset

from .foresight_dataset import current_observation
from .st_world_history import HistoryReader, file_sha256
from .st_world_schema import ModelInputs, TeacherFeatureContract, identity_hash
from .st_world_targets import TeacherCacheReader


def projected_teacher_contract(raw_contract, *, pca_identity, dimension):
    """New target coordinate identity; never relabel raw1024 as compressed256."""
    if type(dimension) is not int or not 1<=dimension<=raw_contract.channels:
        raise ValueError('Invalid explicitly selected PCA dimension')
    value=raw_contract.to_dict()
    value.update(channels=dimension,storage_dtype='float32',
        transform_identity=identity_hash({'schema':'st_projected_teacher_transform_v1',
            'raw_teacher_contract_identity':raw_contract.identity,'pca_identity':pca_identity,
            'dimension':dimension,'standardize_channels':True,'whiten':False}))
    return TeacherFeatureContract.from_dict(value)


class STWorldTrainingDataset(Dataset):
    def __init__(self, data_root, manifest_path, teacher_root, expected_teacher_identity,
                 expected_contract, history_root=None, pca_path=None, require_split='train', need_future=False):
        if require_split not in ('train','dev'):
            raise ValueError('ST baseline may not open test/navtest training data')
        if type(need_future) is not bool:
            raise ValueError('need_future must be a resolved bool')
        self.root=Path(data_root)
        self.manifest_path=Path(manifest_path)
        self.source=json.loads((self.root/'identity.json').read_text())
        source_index=json.loads((self.root/'index.json').read_text())
        if (self.source['schema']!='ddpolicy_current_cameras_v1' or self.source['split']!=require_split or
                self.source['identity']!=identity_hash({k:v for k,v in self.source.items() if k!='identity'}) or
                self.source['index_sha256']!=identity_hash(source_index)):
            raise ValueError('Original current population/role identity mismatch')
        manifest=json.loads(self.manifest_path.read_text())
        if (manifest.get('schema')!='st_fixed_scene_list_v1' or manifest['split']!=require_split or
                manifest['source_index_hash']!=self.source['index_sha256']):
            raise ValueError('Selected scenes must retain original split/index identity')
        lookup={r['token']:r for r in source_index}
        self.rows=manifest['scenes']
        if not self.rows or len({r['token'] for r in self.rows})!=len(self.rows):
            raise ValueError('Empty/duplicate fixed training manifest')
        for row in self.rows:
            if row['token'] not in lookup or lookup[row['token']]['log']!=row['log']:
                raise ValueError('Selected scene/log outside original role')
        self.teacher=TeacherCacheReader(teacher_root,expected_identity=expected_teacher_identity)
        if (self.teacher.manifest['split']!=require_split or
                self.teacher.manifest.get('source_student_identity')!=self.source['identity'] or
                self.teacher.manifest.get('source_index_sha256')!=self.source['index_sha256'] or
                self.teacher.manifest.get('partition_sha256')!=self.source['partition_sha256']):
            raise ValueError('Teacher data source/partition/role differs from current data')
        for row in self.rows:
            if row['token'] not in self.teacher.scenes or self.teacher.scenes[row['token']]['log']!=row['log']:
                raise ValueError('WAITING_INPUTS: full fixed manifest lacks complete teacher records')
        # The preparation receipt pins each source current JSON even with history
        # disabled. A raw teacher cache alone does not pin mutable ego metadata.
        history_manifest_path=self.manifest_path.parent/'HISTORY_MANIFEST.json'
        self.current_pins={}
        if history_manifest_path.exists():
            hm=json.loads(history_manifest_path.read_text())
            hd=json.loads((history_manifest_path.parent/'COMPLETE.json').read_text())
            if (identity_hash({k:v for k,v in hm.items() if k!='identity'})!=hm['identity'] or
                    hd['history_manifest_sha256']!=file_sha256(history_manifest_path) or
                    hm['source_student_identity']!=self.source['identity']):
                raise ValueError('Source-current content pins changed')
            self.current_pins={r['token']:r['current_record_sha256'] for r in hm['scenes']}
        for row in self.rows:
            if 'current_record_sha256' in row:
                self.current_pins[row['token']]=row['current_record_sha256']
            if row['token'] not in self.current_pins:
                raise ValueError('Fixed selection lacks immutable per-current-record checksum')
            if file_sha256(self.root/'current'/(row['token']+'.json'))!=self.current_pins[row['token']]:
                raise ValueError('Current record changed after data registration')
        self.ego_identity=json.loads((self.root/'ego_identity.json').read_text())
        if (self.ego_identity['schema']!='foresight_ego_labels_v1' or
                self.ego_identity['index_sha256']!=self.source['index_sha256'] or
                self.ego_identity['identity']!=identity_hash({k:v for k,v in self.ego_identity.items() if k!='identity'})):
            raise ValueError('Original ego target contract changed')
        self.ego_pins={r['token']:file_sha256(self.root/'ego'/(r['token']+'.pt')) for r in self.rows}
        self.history=None
        if history_root is not None:
            history_identity=json.loads((Path(history_root)/'HISTORY_MANIFEST.json').read_text())['identity']
            self.history=HistoryReader(history_root,history_identity)
            if self.history.manifest['split']!=require_split:
                raise ValueError('History role mismatch')
            if any(r['token'] not in self.history.index for r in self.rows):
                raise ValueError('History population incomplete')
        if isinstance(expected_contract,dict):expected_contract=TeacherFeatureContract.from_dict(expected_contract)
        self.pca=None
        if pca_path is not None:
            from tools.st_world_policy.teacher_pca import TeacherPCA
            receipt=json.loads((Path(pca_path)/'PCA.json').read_text())
            self.pca=TeacherPCA.load(pca_path,dimension=expected_contract.channels,
                expected_identity=receipt['identity'],expected_teacher_contract=self.teacher.contract.identity)
            effective=projected_teacher_contract(self.teacher.contract,pca_identity=receipt['identity'],dimension=expected_contract.channels)
        else:
            effective=self.teacher.contract
        if effective.identity!=expected_contract.identity:
            raise ValueError('Raw/PCA target contract does not match registered training configuration')
        self.contract=effective
        self.need_future=need_future
        self.require_split=require_split
        self.identity={'schema':'st_training_dataset_v1','data_manifest_sha256':file_sha256(self.manifest_path),
            'current_source_identity':self.source['identity'],'current_record_pins':self.current_pins,
            'ego_identity':self.ego_identity['identity'],'ego_record_pins':self.ego_pins,
            'teacher_cache_identity':expected_teacher_identity,'raw_teacher_contract_identity':self.teacher.contract.identity,
            'effective_target_contract':effective.to_dict(),'pca_identity':self.pca.identity if self.pca else None,
            'history_identity':self.history.manifest['identity'] if self.history else None,
            'split':require_split,'need_future':need_future,'scenes':len(self.rows)}
        self.identity['identity']=identity_hash(self.identity)

    def __len__(self):return len(self.rows)

    def __getitem__(self, index):
        row=self.rows[index];token=row['token']
        path=self.root/'current'/(token+'.json')
        if file_sha256(path)!=self.current_pins[token]:raise ValueError('Current JSON changed during training')
        record=json.loads(path.read_text())
        if record['identity']!=self.source['identity'] or record['token']!=token or record['log']!=row['log']:
            raise ValueError('Foreign current scene')
        scene=self.teacher.scenes[token]
        if scene['timestamp_us']!=record['timestamp']:
            raise ValueError('Teacher/current decision time mismatch')
        for ref,source_path in zip(scene['frames'][0]['views'],record['image_paths']):
            if ref is None or Path(ref['path']).resolve()!=Path(source_path).resolve() or file_sha256(source_path)!=ref['sha256']:
                raise ValueError('Teacher source pixels differ from current model input')
        current=current_observation(record)
        history=self.history.read(token,current_record=record) if self.history else None
        inputs=ModelInputs(current,history)
        ego_path=self.root/'ego'/(token+'.pt')
        if file_sha256(ego_path)!=self.ego_pins[token]:raise ValueError('Ego target changed during training')
        label=torch.load(ego_path,map_location='cpu',weights_only=True)
        if label['identity']!=self.ego_identity['identity'] or label['token']!=token:
            raise ValueError('Original ego label identity mismatch')
        ego=label['ego']
        if ego.ndim!=2 or ego.shape[-1]!=4 or not torch.isfinite(ego).all():raise ValueError('Illegal original normalized ego GT')
        item=self.teacher.read_scene(token,expected_split=self.require_split,need_future=self.need_future)
        current_z,future_z=item['current_dense_features'],item['future_dense_features']
        if self.pca is not None:
            current_z=self.pca.transform(current_z).detach()
            future_z=self.pca.transform(future_z).detach()
            # Invalid labels remain exact zeros after the affine PCA transform.
            current_z=torch.where(item['current_valid'][...,None],current_z,torch.zeros_like(current_z))
            future_z=torch.where(item['future_valid'][...,None],future_z,torch.zeros_like(future_z))
        targets={'ego':ego,'current_dense_features':current_z,'current_valid':item['current_valid'],
                 'future_dense_features':future_z,'future_valid':item['future_valid'],
                 'future_times_s':torch.tensor(item['future_times_s'],dtype=torch.float32)}
        return inputs,targets


def collate_st_world(rows):
    if not rows:raise ValueError('Empty ST optimizer batch')
    inputs,targets=zip(*rows)
    keys=set(targets[0])
    if any(set(t)!=keys for t in targets):raise ValueError('Mixed target schemas in batch')
    return list(inputs),{key:torch.stack([t[key] for t in targets]) for key in targets[0]}
