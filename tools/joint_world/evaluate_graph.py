"""Evaluate a saved joint graph on current-only caches with independent geometry matching."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow
from tools.joint_world.train_graph import load_samples, evaluate
from tools.structured_world_v1p1.budget import start, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','cache','targets','data-root','output','ledger','run-id']: p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['identity']['config']
    samples,manifest,fingerprint=load_samples(a.cache,a.targets,a.data_root,8)
    train_manifest=json.loads((Path(saved['identity']['arguments']['cache'])/'manifest.json').read_text())
    for key in ['world_checkpoint_sha256','baseline_checkpoint_sha256','world_config','sensor_contract']:
        if manifest['identity'][key]!=train_manifest['identity'][key]:raise ValueError('Upstream condition identity changed: '+key)
    identity={'arguments':vars(a),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'cache_identity':manifest['identity_sha256'],'labels_fingerprint':fingerprint,'checkpoint_step':saved['step']}
    start(a.ledger,a.run_id,0,identity)
    try:
        model=JointTrajectoryFlow(samples[0]['cache']['context'].shape[-1],**{k:v for k,v in cfg.items() if k in ['dim','heads','layers','scale_m']}).cuda().eval()
        model.load_state_dict(saved['model'],strict=True)
        evaluate(model,samples,out,saved['step'])
        (out/'manifest.json').write_text(json.dumps(identity,indent=2));record(a.ledger,a.run_id,0,'complete')
    except BaseException:record(a.ledger,a.run_id,0,'failed');raise


if __name__=='__main__':main()
