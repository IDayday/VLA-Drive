"""Explicit A-only index view over the pinned S0 deterministic input readers.

Instantiate inside the verified historical runtime. Generic DINO/clip files
retain their original identity and physical indexing; only the new A-only
teacher supplies fitted interaction labels. No B item is exposed by this view.
"""
import json
from pathlib import Path
import torch


class AOnlyBaseView:
    def __init__(self, original_class, *args, population_contract, **kwargs):
        from starVLA.model.modules.vehicle_joint.initialization import identity_hash
        contract=json.loads(Path(population_contract).read_text())
        if contract['schema']!='a_only_s0_training_population_v1' or contract['role']!='A_train':
            raise ValueError('Explicit A-only base training population required')
        audit=json.loads(Path(contract['source_audit']).read_text())
        if audit['identity']!=contract['source_audit_identity']:raise ValueError('Population source audit changed')
        index=json.loads(Path(contract['index']).read_text())
        if identity_hash(index)!=contract['index_identity']:raise ValueError('A-only index changed')
        roles=audit['roles']
        if any(roles.get(row['log'].rsplit('_',2)[0])!='A_train' for row in index):
            raise ValueError('Correction/calibration recording in base gradients')
        interaction_root=kwargs.pop('interaction_root',None);expected=kwargs.pop('expected_interaction',None)
        # The delegated dataset checks every original generic cache identity.
        self.parent=original_class(*args,interaction_root=None,expected_interaction=None,**kwargs)
        full={row['token']:(i,row) for i,row in enumerate(self.parent.index)}
        if any(row['token'] not in full or full[row['token']][1]!=row for row in index):
            raise ValueError('A-only scene is not a legal original training scene')
        self.index=index;self.positions=[full[row['token']][0] for row in index]
        self.root=self.parent.root
        self.identity={**self.parent.identity,'schema':'a_only_current_input_view_v1',
            'parent_current_identity':self.parent.identity['identity'],
            'index_sha256':identity_hash(index),'population_contract':contract,
            'scope':'Only A_train indices visible to optimizer; generic source files may contain other read-only rows'}
        self.identity.pop('identity',None);self.identity['identity']=identity_hash(self.identity)
        self.ego_identity={**self.parent.ego_identity,'parent_ego_identity':self.parent.ego_identity['identity'],
                           'index_sha256':identity_hash(index)}
        self.ego_identity.pop('identity',None);self.ego_identity['identity']=identity_hash(self.ego_identity)
        for name in ('future_identity','dino_identity','clip_identity'):setattr(self,name,getattr(self.parent,name,None))
        self.interaction_root=Path(interaction_root) if interaction_root else None
        self.interaction_identity=None
        self.action_codec=None
        if contract.get('action_codec'):
            self.action_codec=json.loads(Path(contract['action_codec']).read_text())
            if self.action_codec['identity']!=contract['action_codec_identity'] or self.action_codec['index_sha256']!=identity_hash(index):
                raise ValueError('Action statistics are not fitted on this A population')
            import importlib.util
            spec=importlib.util.spec_from_file_location('_a_only_codec_dataset',Path(__file__).with_name('a_only_codec.py'))
            codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
            self.recode_targets=codec.recode_historical_targets
            from starVLA.dataloader import foresight_dataset as fd
            if [fd.X_MEAN,fd.Y_MEAN]!=self.action_codec['mean_xy_m'] or [fd.X_STD,fd.Y_STD]!=self.action_codec['std_xy_m']:
                raise ValueError('Current-input codec was not installed before reading A data')
        elif contract.get('require_A_teacher',True):
            raise ValueError('Formal A-only base needs identified A-only normalization')
        if self.interaction_root:
            ident=json.loads((self.interaction_root/'identity.json').read_text())
            complete=json.loads((self.interaction_root/'COMPLETE.json').read_text())
            if ident['identity']!=expected or ident['index_hash']!=identity_hash(index) or ident['split']!='train':
                raise ValueError('Interaction labels do not cover exactly the A population')
            if complete['identity']!=expected or complete['scenes']!=len(index):raise ValueError('Incomplete A-only teacher export')
            teacher=json.loads(Path(contract['teacher_training_identity']).read_text())
            if ident['teacher_identity']!=teacher['identity'] or teacher['train_index_hash']!=identity_hash(index):
                raise ValueError('Teacher has not been fitted on the registered A-only population')
            if teacher['config']['data']!=contract['teacher_input_view']:raise ValueError('Foreign fitted teacher input role')
            self.interaction_identity=ident
        elif contract.get('require_A_teacher',True):
            raise ValueError('A-only S0 retains MAE; a fitted A-only teacher cache is mandatory')

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        observation,targets=self.parent[self.positions[index]]
        if observation['token']!=self.index[index]['token']:raise ValueError('Delegated index mapping changed')
        if self.action_codec:
            targets={**targets,'ego':self.recode_targets(targets['ego'],self.action_codec)}
        if self.interaction_root:
            label=torch.load(self.interaction_root/'targets'/(observation['token']+'.pt'),map_location='cpu',weights_only=True)
            if label['identity']!=self.interaction_identity['identity'] or label['token']!=observation['token']:
                raise ValueError('A-only fitted target changed')
            targets={**targets,'interaction_latent':label['latent'],
                     'interaction_valid':torch.tensor(label['interaction_target_valid'],dtype=torch.bool)}
        return observation,targets
