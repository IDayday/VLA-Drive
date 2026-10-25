"""Immutable, fixed-boundary candidate-bank adoption for joint IL.

Registry publication never changes an already locked boundary. Checkpoints
retain the actual bank and all samplers, independent of a newer registry.
"""
import fcntl
import hashlib
import json
from pathlib import Path

import torch.distributed as dist

from .checkpoint import _rank_zero_call, atomic_json
from .contracts import file_sha, fingerprint

ENTRY_FIELDS={'source_updates','source_checkpoint','source_model_sha256','parent_bank_hash',
              'bank_root','bank_hash','data_manifest_sha256','score_protocol_sha256'}


def validate_refresh_config(value):
    if set(value)!={'enabled','produce_every_updates','adopt_every_updates','new_candidates_per_scene','max_candidates','registry'}:
        raise ValueError('Unknown/missing candidate refresh configuration')
    if (type(value['enabled']) is not bool or value['produce_every_updates']!=10000
            or value['adopt_every_updates']!=5000 or value['new_candidates_per_scene']!=2
            or value['max_candidates']!=16 or not isinstance(value['registry'],str)):
        raise ValueError('Unsupported candidate refresh schedule/budget')
    return value


class BankVersions:
    def __init__(self,run_root,config,builder,run_identity):
        self.root=Path(run_root);self.config=validate_refresh_config(config)
        bank=builder.bank;self.initial=dict(source_updates=0,source_checkpoint=None,source_model_sha256=None,
            parent_bank_hash=None,bank_root=str(Path(bank.manifest_path).parent.resolve()),bank_hash=bank.bank_hash,
            data_manifest_sha256=builder.data.sha,
            score_protocol_sha256=bank.manifest['identity']['score_protocol_sha256'])
        self.run_identity=fingerprint(run_identity)

    def _registry(self):
        path=Path(self.config['registry'])
        if not path.exists():return [],None
        payload=path.read_bytes();data=json.loads(payload);body=dict(data);seal=body.pop('sha256',None)
        if set(body)!={'schema','entries'} or body['schema']!='world_postcot_bank_registry_v1':
            raise ValueError('Wrong candidate-bank registry schema')
        if seal is not None and fingerprint(body)!=seal:raise ValueError('Changed bank registry seal')
        seen={};by_update={}
        for row in body['entries']:
            if set(row)!=ENTRY_FIELDS:raise ValueError('Wrong candidate-bank registry entry fields')
            step=row['source_updates']
            if type(step) is not int or not 10000<=step<=90000 or step%self.config['produce_every_updates']:
                raise ValueError('Refresh source must be a declared formal10k..90k snapshot')
            if (row['data_manifest_sha256']!=self.initial['data_manifest_sha256']
                    or row['score_protocol_sha256']!=self.initial['score_protocol_sha256']):
                raise ValueError('Refresh data/score identity differs from the initial bank')
            if row['bank_hash'] in seen or step in by_update:raise ValueError('Duplicate/ambiguous refresh version')
            seen[row['bank_hash']]=row;by_update[step]=row
        for row in body['entries']:
            current=row;visited=set()
            while current['parent_bank_hash']!=self.initial['bank_hash']:
                parent=current['parent_bank_hash']
                if parent in visited or parent not in seen:raise ValueError('Unresolved/cyclic bank parent lineage')
                visited.add(parent);previous=seen[parent]
                if previous['source_updates']>=current['source_updates']:raise ValueError('Bank parent is not an earlier source snapshot')
                current=previous
        return body['entries'],hashlib.sha256(payload).hexdigest()

    def _validate_source(self,row):
        if row['source_updates']==0:return
        root=Path(row['source_checkpoint']);seal=json.loads((root/'COMPLETE').read_text())
        files=json.loads((root/'files.json').read_text());identity=json.loads((root/'identity.json').read_text())
        # The publisher verifies source model bytes. Adoption consumes scored
        # records, so it checks the immutable full checkpoint receipt without
        # rereading a10GB model on the training hot path.
        if (seal.get('status')!='COMPLETE' or seal['updates']!=row['source_updates']
                or seal['files_sha256']!=file_sha(root/'files.json')
                or seal['identity_sha256']!=fingerprint(identity)
                or fingerprint(identity)!=self.run_identity or identity.get('qualification') is not False
                or files[seal['model_file']]!=row['source_model_sha256']):
            raise ValueError('Refresh source is not this formal IL lineage/snapshot')

    def _decide(self,builder,updates):
        folder=self.root/'bank_bindings';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'update{updates:06d}.json'
        with (folder/f'update{updates:06d}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():
                decision=json.loads(path.read_text());body=dict(decision);seal=body.pop('sha256',None)
                if seal!=fingerprint(body):raise ValueError('Changed locked bank boundary')
                if builder.bank_binding is not None and builder.bank_binding['updates']==updates and decision!=builder.bank_binding:
                    raise ValueError('Locked boundary differs from the saved checkpoint decision')
            elif builder.bank_binding is not None and builder.bank_binding['updates']==updates:
                # Migration to a new output directory preserves the decision
                # stored by its parent checkpoint, not today's registry.
                decision=dict(builder.bank_binding);atomic_json(path,decision)
            else:
                selected=(builder.bank_binding or {}).get('selected',self.initial)
                entries,registry_sha=([],None) if updates==0 else self._registry()
                eligible=[entry for entry in entries if entry['source_updates']<=updates]
                if eligible:
                    latest=max(eligible,key=lambda entry:entry['source_updates'])
                    if latest['source_updates']>selected['source_updates']:selected=latest
                    elif latest['source_updates']==selected['source_updates'] and latest['bank_hash']!=selected['bank_hash']:
                        raise ValueError('Conflicting bank at the already adopted source step')
                self._validate_source(selected)
                decision=dict(schema='world_postcot_bank_binding_v1',updates=updates,
                    initial_bank_hash=self.initial['bank_hash'],selected=dict(selected),registry_sha256=registry_sha)
                decision['sha256']=fingerprint(decision);atomic_json(path,decision)
            body=dict(decision);seal=body.pop('sha256',None)
            if (seal!=fingerprint(body) or decision['schema']!='world_postcot_bank_binding_v1'
                    or decision['updates']!=updates or decision['initial_bank_hash']!=self.initial['bank_hash']):
                raise ValueError('Locked boundary belongs to another run/bank')
            return decision

    def adopt(self,builder,updates):
        if not self.config['enabled'] or updates%self.config['adopt_every_updates']:return None
        decision=_rank_zero_call(lambda:self._decide(builder,updates))
        error=None
        try:builder.adopt_bank(decision['selected'],decision)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        errors=[error]
        if dist.is_initialized():
            errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
        if any(value is not None for value in errors):raise RuntimeError('Candidate-bank admission failed consistently: '+repr(errors))
        return decision
