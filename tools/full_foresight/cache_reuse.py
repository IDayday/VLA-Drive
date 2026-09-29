"""Byte-verified reuse of compatible per-image targets, never schema renaming."""
import json
from pathlib import Path
from safetensors import safe_open
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256


class CurrentTargetReuse:
    def __init__(self, root, candidates, recipe):
        self.root = Path(root);self.candidates=candidates;self.identities={};self.maps={};self.checked=set()
        for c in candidates:
            folder=self.root/c.name
            ident=json.loads((folder/'identity.json').read_text())
            if ident['schema']!='dino_tradeoff_cache_v1' or ident['candidate']!=c.record() or ident['recipe']!=recipe:
                raise ValueError('Current-cache teacher/preprocess/pooling is not identical')
            if identity_hash({k:v for k,v in ident.items() if k!='identity'})!=ident['identity']:
                raise ValueError('Changed current-cache identity')
            self.identities[c.name]=ident;lookup={}
            for path in sorted(folder.glob('chunk_*.json')):
                meta=json.loads(path.read_text())
                if meta['identity']!=ident['identity']:raise ValueError('Foreign reusable chunk')
                for at,key in enumerate(meta['keys']):
                    if key in lookup:raise ValueError('Duplicate reusable image key')
                    lookup[key]=(path,at,meta)
            self.maps[c.name]=lookup

    def get(self, row):
        if any(row['key'] not in self.maps[c.name] for c in self.candidates):return None
        values={};transform=image_hash=None;origins={}
        for c in self.candidates:
            path,at,meta=self.maps[c.name][row['key']];tensor_path=path.with_suffix('.safetensors')
            if str(path) not in self.checked:
                if file_sha256(tensor_path)!=meta['sha256']:raise ValueError('Corrupt reusable image target')
                self.checked.add(str(path))
            with safe_open(str(tensor_path),framework='pt',device='cpu') as handle:
                if handle.metadata()['identity']!=self.identities[c.name]['identity']:
                    raise ValueError('Reusable tensor identity mismatch')
                z=handle.get_slice('features')[at];valid=handle.get_slice('valid')[at]
            if not valid.all() or z.shape!=(1024,*c.grid_hw):raise ValueError('Reusable spatial contract mismatch')
            if image_hash is not None and image_hash!=meta['image_sha256'][at]:raise ValueError('Pool variants came from different RGB')
            image_hash=meta['image_sha256'][at];transform=meta['transforms'][at];values[c.name]=z
            origins[c.name]={'identity':meta['identity'],'chunk_sha256':meta['sha256'],'row':at,
                'original_quantization_mse':meta['quantization_mse'],'original_quantization_max_abs':meta['quantization_max_abs']}
        return values,transform,image_hash,origins
