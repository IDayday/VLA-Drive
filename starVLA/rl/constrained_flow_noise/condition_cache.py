"""Read-only observation conditions. No GT, action labels, reward or noise output."""
import fcntl
import json
import os
from pathlib import Path
import uuid

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import CONDITION_BOUNDARY, file_sha, fingerprint


class ConditionStore:
    def __init__(self, root, identity=None, *, readonly=True):
        self.root = Path(root)
        self.readonly = readonly
        if readonly:
            self.identity = json.loads((self.root / 'identity.json').read_text())
            complete = json.loads((self.root / 'COMPLETE.json').read_text())
            if complete['identity_sha256'] != fingerprint(self.identity):
                raise ValueError('Condition publication identity changed')
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / 'identity.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = self.root / 'identity.json'
                if path.exists() and json.loads(path.read_text()) != identity:
                    raise ValueError('Cache execution contract changed; use a new cache identity')
                if not path.exists():
                    atomic_json(path, identity)
            self.identity = identity
        if self.identity['boundary'] != CONDITION_BOUNDARY or self.identity['schema'] != 'original_deployment_condition_v1':
            raise ValueError('Old BF16 feature cache is not valid for this research')
        self.identity_sha = fingerprint(self.identity)

    def path(self, token):
        if not token or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('Invalid scene token')
        return self.root / token[:2] / (token + '.pt')

    def get(self, token):
        path = self.path(token)
        if not path.with_suffix('.json').exists():
            if self.readonly:
                raise FileNotFoundError(path)
            return None
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != self.identity_sha or seal['sha256'] != file_sha(path):
            raise ValueError('Condition seal mismatch')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if set(row) != {'token', 'raw'} or row['token'] != token:
            raise ValueError('Condition cache may contain only observation features and scene identity')
        raw = row['raw']
        if raw.ndim != 3 or raw.shape[:2] != (1, 8) or raw.requires_grad or not torch.isfinite(raw).all():
            raise ValueError('Invalid normalized observation condition')
        return raw

    def put(self, token, raw):
        if self.readonly:
            raise ValueError('Read-only cache')
        path = self.path(token); path.parent.mkdir(parents=True, exist_ok=True)
        raw = raw.detach().cpu().contiguous()
        if raw.ndim != 3 or raw.shape[:2] != (1, 8) or not torch.isfinite(raw).all():
            raise ValueError('Invalid observation feature')
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = self.get(token)
            if existing is not None:
                if not torch.equal(existing, raw):
                    raise ValueError('Repeated cache computation differs')
                return
            tmp = path.with_suffix('.partial-' + uuid.uuid4().hex)
            torch.save({'token': token, 'raw': raw}, tmp)
            os.replace(tmp, path)
            atomic_json(path.with_suffix('.json'), {'identity_sha256': self.identity_sha,
                        'sha256': file_sha(path), 'token': token})

    def finalize(self, tokens):
        entries = []
        for token in tokens:
            self.get(token)
            entries.append(json.loads(self.path(token).with_suffix('.json').read_text()))
        if len(tokens) != len(set(tokens)):
            raise ValueError('Duplicate cache population')
        atomic_json(self.root / 'manifest.json', {'entries': entries, 'identity_sha256': self.identity_sha})
        atomic_json(self.root / 'COMPLETE.json', {'count': len(tokens), 'identity_sha256': self.identity_sha,
                    'manifest_sha256': file_sha(self.root / 'manifest.json')})
