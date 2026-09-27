"""Repack a frozen public current cache with another registered current-only selector."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.model.modules.joint_world.local_cache import cache_identity,signature,load_payload
from starVLA.model.modules.joint_world.local_graph import build_local_graph
from starVLA.model.modules.joint_world.public_baseline import sha256
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,observation_from_files
from tools.local_interaction_mask_v2.train_foundation import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','selector','dataset','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();source=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text());old=manifest['identity']
    if not manifest['complete'] or manifest['failed']:raise ValueError('Source cache incomplete')
    selector=json.loads(Path(a.selector).read_text());spec=json.loads(Path(a.dataset).read_text())
    identity=cache_identity(old['foundation_sha256'],old['public_origin'],selector,old['graph_config'])
    for key in ('foundation_training','foundation_step'):identity[key]=old[key]
    records=[]
    for record in manifest['records']:
        token=record['token'];payload=load_payload(source,record,manifest)
        metadata=(json.loads((Path(spec['current_records'])/(token+'.json')).read_text()) if 'current_records' in spec else
                  current_metadata_from_training_pickle(Path(spec['meta_root'])/(token+'.pkl'),token))
        observation,_=observation_from_files(Path(spec['observations'])/(token+'.npz'),metadata,verify_transform=True)
        if observation.fingerprint()!=payload['observation_identity']:raise ValueError('Current input changed while rebuilding graph')
        prediction=payload['current_prediction']
        graph=build_local_graph(prediction['boxes'][0].float().numpy(),prediction['logits'][0].float().numpy(),observation,selector)
        payload=dict(payload,local_graph=graph.tensor_state(),identity_sha256=signature(identity))
        path=out/(token+'.pt');torch.save(payload,path);records.append({'token':token,'sha256':sha256(path)})
    atomic_json(out/'manifest.json',{'identity':identity,'identity_sha256':signature(identity),'records':records,
        'expected':len(records),'failed':0,'complete':True,'regraph_source_manifest_sha256':sha256(source/'manifest.json')})


if __name__=='__main__':main()
