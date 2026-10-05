"""Diagnostic current-image repetition through the same genuine video encoder.

These artificial static clips are references, never formal future labels or
missing-frame replacements. They have a separate schema and cannot enter the
student clip Dataset.
"""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image
import torch

from starVLA.model.modules.foresight.video_target_encoder import VideoTargetEncoder
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    parser=argparse.ArgumentParser(__doc__)
    for key in ('current-data','future-targets','source-root','checkpoint','source-sha',
                'weight-sha256','output','campaign-root','run-id'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--queries',help='Explicit fixed training diagnostic scene list; never all-training fallback')
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:
        raise ValueError('Invalid reference shard')
    if subprocess.check_output(['git','status','--porcelain']).strip():
        raise ValueError('Freeze diagnostic source')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    data=Path(args.current_data);out=Path(args.output)
    current=json.loads((data/'identity.json').read_text())
    targets=json.loads((Path(args.future_targets)/'identity.json').read_text())
    rows=json.loads((data/'index.json').read_text())
    if targets['future_target_type']!='video_clip' or targets['scene_index_hash']!=current['index_sha256']:
        raise ValueError('Only matching split/population video references allowed')
    selection=None
    if args.queries:
        requested=json.loads(Path(args.queries).read_text());lookup={r['token']:r for r in rows}
        if len({r['token'] for r in requested})!=len(requested):raise ValueError('Duplicate diagnostic query')
        for r in requested:
            if r['token'] not in lookup or lookup[r['token']]['log']!=r['log']:raise ValueError('Foreign diagnostic query')
        rows=[lookup[r['token']] for r in requested];selection=identity_hash(requested)
    elif current['split']!='dev':
        raise ValueError('Training static diagnostic requires an explicit fixed subset')
    out.mkdir(parents=True,exist_ok=True);(out/'targets').mkdir(exist_ok=True)
    with metered_run(args.campaign_root,args.run_id,1,
                     {'kind':'artificial_static_video_diagnostic_reference','real_optimizer_updates':0}) as (meter,_,save):
        encoder=VideoTargetEncoder(args.source_root,args.checkpoint,
            source_sha=args.source_sha,weight_sha256=args.weight_sha256).cuda()
        if encoder.identity!=targets['recipe']:
            raise ValueError('Static reference must use the identical video encoder/recipe')
        identity={'schema':'action_video_static_reference_v1','current':current['identity'],
            'future_target_identity':targets['identity'],'recipe':encoder.identity,
            'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'purpose':'artificial current-frame repetition; never a real future label',
            'current_future_population_hash':current['index_sha256']}
        if selection:identity['diagnostic_query_hash']=selection
        identity['identity']=identity_hash(identity)
        if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:
            raise ValueError('Different diagnostic reference identity')
        atomic_json(out/'identity.json',identity)
        for row in rows[args.shard::args.shards]:
            target=out/'targets'/(row['token']+'.pt')
            if target.exists():
                saved=torch.load(target,weights_only=True)
                if saved['identity']!=identity['identity']:
                    raise ValueError('Foreign existing static reference')
                continue
            record=json.loads((data/'current'/(row['token']+'.json')).read_text())
            clips=[];hashes=[]
            for path in record['image_paths']:
                with Image.open(path) as image:
                    frame=image.convert('RGB')
                    clips.append(encoder.preprocess([frame]*8))
                hashes.append(file_sha256(path))
            features=encoder(torch.stack(clips).cuda()).half().cpu()
            if not torch.isfinite(features).all():
                raise FloatingPointError('Invalid static-reference features')
            temporary=target.with_suffix('.tmp')
            torch.save({'identity':identity['identity'],'token':row['token'],
                        'features':features,'current_image_sha256':hashes},temporary)
            temporary.replace(target)
            meter['inference_scenes']=meter.get('inference_scenes',0)+1;save()
        atomic_json(out/f'shard_{args.shard}.json',{'identity':identity['identity'],
            'status':'COMPLETE','scenes':len(rows[args.shard::args.shards])})
        if all((out/f'shard_{i}.json').exists() for i in range(args.shards)):
            states=[json.loads((out/f'shard_{i}.json').read_text()) for i in range(args.shards)]
            if any(s['identity']!=identity['identity'] or s['status']!='COMPLETE' for s in states) or sum(s['scenes'] for s in states)!=len(rows):
                raise ValueError('Static-reference population incomplete')
            atomic_json(out/'COMPLETE.json',{'identity':identity['identity'],'scenes':len(rows)})


if __name__=='__main__':
    main()
