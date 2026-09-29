"""Download the exact registered VAE from Diffusers' public standalone distribution."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .flux_targets import (FLUX_REPOSITORY,FLUX_REVISION,FLUX_CONFIG,FLUX_WEIGHT,
                          FLUX_CONFIG_GIT_BLOB,FLUX_WEIGHT_SHA256,verify_flux_source)
from .score_pdms import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256

PUBLIC_REPOSITORY='diffusers/FLUX.1-vae'
PUBLIC_REVISION='da548cfb003bdeebaff6da0211fc8fbc67cb563a'
WEIGHT_BYTES=167666902
HTTP=requests.Session()
HTTP.mount('https://',HTTPAdapter(max_retries=Retry(total=4,connect=4,read=4,backoff_factor=.5,
    status_forcelist=(429,500,502,503,504),allowed_methods=('GET',))))


def download(url,path,size,expected_sha=None):
    if path.exists():
        if path.stat().st_size!=size or expected_sha and file_sha256(path)!=expected_sha:
            raise ValueError(f'Existing destination differs; refusing to overwrite {path}')
        return
    partial=path.with_suffix(path.suffix+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise ValueError('Partial file exceeds expected public asset size')
    if offset<size:
        headers={'Range':f'bytes={offset}-'} if offset else {}
        with HTTP.get(url,headers=headers,stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            if offset and response.status_code==206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Server resumed the wrong byte range')
                mode='ab'
            elif response.status_code==200:mode='wb';offset=0
            else:raise ValueError('Unexpected download response')
            with partial.open(mode) as stream:
                for block in response.iter_content(2**20):
                    if block:stream.write(block)
    if partial.stat().st_size!=size:raise ValueError('Truncated download retained for resume')
    if expected_sha and file_sha256(partial)!=expected_sha:raise ValueError('Downloaded weight hash differs from canonical BFL VAE')
    partial.replace(path)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--identity-output',required=True)
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'DOWNLOAD.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=Path(a.identity_output);started=time.time()
    url=f'https://huggingface.co/api/models/{PUBLIC_REPOSITORY}/revision/{PUBLIC_REVISION}?blobs=true'
    response=HTTP.get(url,timeout=30);response.raise_for_status();metadata=response.json()
    if metadata['id']!=PUBLIC_REPOSITORY or metadata['sha']!=PUBLIC_REVISION or metadata.get('private') or metadata.get('gated'):
        raise ValueError('Expected pinned publicly accessible Diffusers distribution')
    entries={row['rfilename']:row for row in metadata['siblings']}
    if entries['config.json']['blobId']!=FLUX_CONFIG_GIT_BLOB:
        raise ValueError('Standalone config differs from registered BFL source')
    weight=entries['diffusion_pytorch_model.safetensors']
    if weight['lfs']['sha256']!=FLUX_WEIGHT_SHA256 or weight['size']!=WEIGHT_BYTES:
        raise ValueError('Standalone weights differ from registered BFL source')
    (root/'vae').mkdir(exist_ok=True)
    for relative in (FLUX_CONFIG,FLUX_WEIGHT):
        name=Path(relative).name;entry=entries[name]
        address=f'https://huggingface.co/{PUBLIC_REPOSITORY}/resolve/{PUBLIC_REVISION}/{name}'
        print(f'Downloading/verifying {name} ({entry["size"]} bytes)',flush=True)
        download(address,root/relative,entry['size'],FLUX_WEIGHT_SHA256 if relative==FLUX_WEIGHT else None)
    identity={'repository':FLUX_REPOSITORY,'revision':FLUX_REVISION,
              'files':{name:file_sha256(root/name) for name in (FLUX_CONFIG,FLUX_WEIGHT)},
              'canonical_identity':'Exact registered BFL config Git blob and VAE weight SHA256, byte-identical public distribution',
              'download_source':{'repository':PUBLIC_REPOSITORY,'revision':PUBLIC_REVISION,'gated':False,'metadata_url':url},
              'generic_pretraining_data_fully_auditable':False,'driving_adaptation':False}
    verify_flux_source(root,identity)
    if output.exists():
        if json.loads(output.read_text())!=identity:raise ValueError('Existing VAE identity differs; not overwritten')
    else:atomic_json(output,identity)
    report={'status':'COMPLETE','identity':identity,'weight_bytes':WEIGHT_BYTES,'config_bytes':entries['config.json']['size'],
            'elapsed_seconds':time.time()-started,'GPU_used':False}
    atomic_json(root/'DOWNLOAD_COMPLETE.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
