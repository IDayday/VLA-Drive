"""Read-only matched G checkpoints, own normal forward, fixed S0 queries."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('dl_audit_helpers',Path(__file__).with_name('common.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('lock','inputs','queries','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--group',choices=('G1_FULL_UNIFORM','G2_EVENT_UNIFORM','G3_EVENT_LOCAL'),required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resources=helpers.check_host();lock=helpers.read(a.lock)
    if not a.limit and not (a.queries/'COMPLETE.json').exists():raise ValueError('Full immutable query bank required')
    frozen=Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
    if helpers.source_sha(frozen)!=lock['training_source_sha']:raise ValueError('G model source changed')
    sys.path.insert(0,str(frozen))
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from tools.structured_world.infer_checkpoint import load_model,scene_noise
    from starVLA.dataloader.structured_world.current_inputs import CurrentInputs
    import starVLA.model.modules as modules
    modules.__path__.append(str(ROOT/'starVLA/model/modules'))
    from starVLA.model.modules.decision_local.consequence_geometry import read_consequences
    from starVLA.model.modules.structured_world.grid import GridSpec
    from starVLA.model.modules.structured_world.queries import body_relations
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=CurrentInputs(a.inputs);lookup={r['token']:i for i,r in enumerate(data.index)}
    group=lock['groups'][a.group];started=time.monotonic()
    model,source,checkpoint=load_model(Path(group['run']),group['tag'],'cuda')
    model.eval().requires_grad_(False)
    if checkpoint['completed']!=group['completed'] or source['identity']!=group['identity']:raise ValueError('Changed matched checkpoint')
    labels=helpers.read(a.queries/'identity.json');a.output.mkdir(parents=True,exist_ok=True)
    identity={'group':a.group,'updates':checkpoint['completed'],'training_source_sha':lock['training_source_sha'],
        'checkpoint_files':checkpoint['file_sha256'],'query_identity':labels['identity'],'input_identity':data.identity['identity'],
        'source_sha':helpers.source_sha(ROOT),'reader_sha256':helpers.digest(__file__),
        'geometric_operator_sha256':helpers.digest(ROOT/'starVLA/model/modules/decision_local/consequence_geometry.py'),
        'sampling_seed':42,'clearance_proxy':'threshold current/future predicted occupancy at fixed0.5, signed EDT clipped10m; not an original G distance head',
        'overlap_proxy':'max predicted occupancy over complete body; uncalibrated, not official collision probability',
        'forward':'normal own H_A/q0/B0/Bt; S0 supplies only fixed independent query coordinates'}
    ident=helpers.identity(identity);helpers.atomic_json(a.output/'identity.json',{'identity':ident,**identity})
    for k in ('chunks','records'): (a.output/k).mkdir(exist_ok=True)
    done=0;grid=GridSpec();body={'length':5.176,'width':2.297,'rear_axle_to_center':1.461}
    for receipt in sorted((a.queries/'records').glob('*.json')):
        qmeta=helpers.read(receipt);query_path=a.queries/'chunks'/(receipt.stem+'.safetensors')
        if qmeta['identity']!=labels['identity'] or helpers.digest(query_path)!=qmeta['sha256']:raise ValueError('Fixed query bank changed')
        q=load_file(str(query_path))
        results=[];tokens=[]
        for i,token in enumerate(qmeta['tokens']):
            if (a.limit and done>=a.limit) or time.monotonic()-started>=a.max_seconds:break
            if token not in lookup:raise ValueError('Query not in current input population')
            observation=data[lookup[token]]
            with torch.no_grad():
                plan=model.plan_from_current([observation],initial_noise=scene_noise(token,42,model.steps,'cuda'))
                fields=model.scene_semantics(plan['current']['B0'],plan['Bt'])
            road=fields['road_distance_m'][0,0].float().cpu().numpy()
            p0=fields['p0'][0,0].float().cpu().numpy();pt=fields['pt'][0,:,0].float().cpu().numpy()
            poses=q['poses'][i,0].numpy();K,T,_=poses.shape
            full=np.ones((T,grid.height,grid.width),bool);road_valid=full[0]
            current=np.repeat((p0>=.5)[None],T,axis=0)
            current_values,_,_=read_consequences(road,road_valid,current,full,poses,body)
            future_values,_,_=read_consequences(road,road_valid,pt>=.5,full,poses,body)
            pos=torch.from_numpy(poses).cuda().reshape(1,K*T,3)
            time_index=torch.arange(T,device='cuda').repeat(K)[None]
            now_read=body_relations(fields['road_distance_m'],fields['p0'][:,None].expand(-1,T,-1,-1,-1),pos,time_index,[body],grid=grid)
            fut_read=body_relations(fields['road_distance_m'],fields['pt'],pos,time_index,[body],grid=grid)
            current_values[...,2]=now_read['occupancy_overlap_proxy'].reshape(K,T).cpu().numpy()
            future_values[...,2]=fut_read['occupancy_overlap_proxy'].reshape(K,T).cpu().numpy()
            results.append(torch.from_numpy(np.concatenate((current_values,future_values[...,1:]),axis=-1)))
            tokens.append(token);done+=1
        if tokens:
            dest=a.output/'chunks'/(receipt.stem+'.safetensors');tmp=dest.with_suffix('.tmp')
            save_file({'relations':torch.stack(results)},str(tmp),metadata={'identity':ident});tmp.replace(dest)
            helpers.atomic_json(a.output/'records'/receipt.name,{'identity':ident,'tokens':tokens,'sha256':helpers.digest(dest)})
            print(json.dumps({'group':a.group,'scenes':done,'seconds':time.monotonic()-started}),flush=True)
        if (a.limit and done>=a.limit) or time.monotonic()-started>=a.max_seconds:break
    expected=min(len(data.index),a.limit) if a.limit else len(data.index)
    helpers.atomic_json(a.output/('COMPLETE.json' if done==expected else 'PARTIAL.json'),{'identity':ident,'scenes':done,
        'expected':expected,'full_development_population':done==1696 and not a.limit,'GPU_hours':(time.monotonic()-started)/3600.,'resources':resources})


if __name__=='__main__':main()
