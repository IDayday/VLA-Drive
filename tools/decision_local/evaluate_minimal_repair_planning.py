"""Matched natural/recovery trajectory export; camera input is target-free.

Uses the existing canonical q0/q_final file transport. For RECOVERY the q0
transport field is explicitly plan_in, not the untouched S0 natural proposal.
"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,read,digest,identity,source_sha,check_host
from tools.decision_local.train_small_model import build_model,load_predictor
from starVLA.model.modules.decision_local.minimal_repair_contract import CurrentFeatures,PlanInputs,execution_view
from starVLA.model.modules.decision_local.consequence_contract import PredictionQueries,OracleRelations,predicted_relations
from starVLA.model.modules.decision_local.recovery_plans import RecoveryConfig,build_plan_pool
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig,trajectory_queries
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points
from starVLA.dataloader.foresight_dataset import decode_ego,encode_ego
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--predictor',type=Path);p.add_argument('--oracle-labels',type=Path);p.add_argument('--index',type=Path)
    p.add_argument('--distribution',choices=('NATURAL','RECOVERY'),required=True)
    p.add_argument('--mode',choices=('camera_only','oracle_probe'),required=True)
    p.add_argument('--relations-ablation',choices=('none','zero','reverse_time'),default='none',
        help='Development-only dependence diagnostic; never the selected execution policy')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resources=check_host();start=time.monotonic()
    if not 0<=a.rank<a.shards:raise ValueError('Invalid scene shard')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=read(a.features/'identity.json');saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    info=saved['identity'];kind=info['kind']
    natural=info.get('protocol')=='natural_local_v1'
    if info.get('protocol') not in ('minimal_repair_v1','natural_local_v1') or info['anchor_identity']!=meta['anchor_identity']:raise ValueError('Wrong repair protocol/base')
    if natural and meta.get('action_codec_identity'):
        anchor=read(info['natural_anchor_record'])
        from starVLA.dataloader.a_only_codec import install_legacy_codec
        if anchor['identity']!=meta['anchor_identity'] or anchor['action_codec_identity']!=meta['action_codec_identity']:raise ValueError('Wrong action codec')
        install_legacy_codec(read(anchor['action_codec']))
    camera=kind in (('N_NONE','R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE') if natural else ('LOG_NONE','MIN_NONE','R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF'))
    oracle=kind in (('N_CV','N_FUT') if natural else ('LOG_CV','LOG_FUT','MIN_CV','MIN_FUT'))
    if not (camera or oracle) or (a.mode=='camera_only')!=camera:raise ValueError('Incorrect execution mode')
    if (camera and a.oracle_labels is not None) or (oracle and a.oracle_labels is None):raise ValueError('GT data only allowed in explicit oracle probe')
    if meta['split']=='navtest' and (not camera or a.distribution!='NATURAL' or a.relations_ablation!='none' or not (a.output.parent/'FINAL_METHOD_LOCK.json').exists()):raise ValueError('Only locked natural camera-only final Navtest')
    if a.relations_ablation!='none' and (meta['split']!='dev' or kind.endswith('_NONE')):raise ValueError('Relation intervention requires development relation model')
    index=read(a.index) if a.index else meta['index'];lookup={r['token']:r for r in meta['index']}
    if len({r['token'] for r in index})!=len(index) or any(lookup.get(r['token'])!=r for r in index):raise ValueError('Foreign/duplicate export rows')
    model=build_model(kind,info['seed'],info['dim']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    predictor=None
    if kind.startswith('R_'):
        if a.predictor is None or digest(a.predictor)!=info['predictor_sha256']:raise ValueError('Predictor differs from R training')
        predictor,_=load_predictor(a.predictor,'P_DIFF' if kind in ('R_MIN_DIFF','R_DIFF_FUTURE') else 'P_ABS')
    elif a.predictor is not None:raise ValueError('No predictor in this control')
    qcfg=QueryConfig(**info['query_config'])
    if 'recovery_config' in info:rcfg=RecoveryConfig(**info['recovery_config'])
    else:
        # The first integration-smoke source predated the redundant config
        # snapshot. Accept only its exact executed generator hash, never an
        # arbitrary old checkpoint or silent scientific default.
        if info['scope']!='smoke' or info['source_files'].get('starVLA/model/modules/decision_local/recovery_plans.py')!=digest(ROOT/'starVLA/model/modules/decision_local/recovery_plans.py'):
            raise ValueError('Missing registered plan pool configuration')
        rcfg=RecoveryConfig()
    if oracle:
        labels=read(a.oracle_labels/'identity.json')
        if labels['feature_identity']!=meta['identity'] or labels['query_config']!=qcfg.to_dict() or labels['recovery']!=rcfg.to_dict():
            # JSON tuples serialize as lists; compare canonical values.
            if labels['feature_identity']!=meta['identity'] or identity(labels['query_config'])!=identity(qcfg.to_dict()) or identity(labels['recovery'])!=identity(rcfg.to_dict()):raise ValueError('Oracle query contract differs')
    declaration={'schema':'minimal_repair_planning_export_v1','features':meta['identity'],'index':index,'checkpoint_sha256':digest(a.checkpoint),
        'kind':kind,'mode':a.mode,'distribution':a.distribution,'sampling_seed':42,'head_inference_batch':1,'head_precision':'FP32','TF32':False,
        'source_sha':source_sha(ROOT),'writer_sha256':digest(__file__),'recovery':rcfg.to_dict(),'query_config':qcfg.to_dict(),
        'oracle_labels':read(a.oracle_labels/'identity.json')['identity'] if oracle else None,
        'relations_ablation':a.relations_ablation,
        'predictor_sha256':digest(a.predictor) if a.predictor else None,
        'q0_transport_meaning':'plan_in for recovery; original S0 q0 for natural','main_trajectory':'q_final only; no GT selection'}
    if natural:declaration.update(schema='natural_local_planning_export_v1',action_codec_identity=meta.get('action_codec_identity'))
    ident=identity(declaration);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Foreign previous export')
    atomic_json(a.output/'identity.json',{'identity':ident,**declaration})
    for k in ('predictions','records','workers'):(a.output/k).mkdir(exist_ok=True)
    wanted={r['token'] for r in index};done=0;expected=0;body={'length':5.176,'width':2.297,'rear_axle_to_center':1.461};grid=GridSpec()
    with torch.no_grad():
        for record_path in sorted((a.features/'records').glob('*.json'))[a.rank::a.shards]:
            record=read(record_path);chunk=a.features/'chunks'/(record_path.stem+'.safetensors')
            if not any(t in wanted for t in record['tokens']):continue
            if digest(chunk)!=record['sha256']:raise ValueError('Changed frozen feature bytes')
            with safe_open(str(chunk),framework='pt',device='cpu') as f:
                for offset,token in enumerate(record['tokens']):
                    if token not in wanted:continue
                    expected+=1
                    if time.monotonic()-start>a.max_seconds:raise TimeoutError('Finite inference budget')
                    dest=a.output/'predictions'/(token+'.npz');receipt=a.output/'records'/(token+'.json')
                    if receipt.exists():
                        r=read(receipt)
                        if r['identity']!=ident or digest(dest)!=r['sha256']:raise ValueError('Changed saved prediction')
                        done+=1;continue
                    q0=f.get_slice('q0')[offset,0].numpy();qenc=f.get_slice('q0_encoded')[offset,0]
                    state=f.get_slice('state')[offset];velocity=decode_ego(state[None])[0,:2].numpy()/.5
                    plan_index=0 if a.distribution=='NATURAL' else 1+int(hashlib.sha256((token+':fixed-recovery-v1').encode()).hexdigest()[:8],16)%4
                    plan=q0 if plan_index==0 else build_plan_pool(q0,velocity,token,rcfg)['plans'][plan_index]
                    encoded=qenc if plan_index==0 else torch.from_numpy(encode_ego(plan))
                    current=CurrentFeatures(f.get_slice('H_A')[offset:offset+1].cuda(),f.get_slice('W')[offset:offset+1].cuda(),state[None].cuda(),torch.tensor([[body[k] for k in ('length','width','rear_axle_to_center')]],device='cuda'))
                    generated=trajectory_queries(plan,velocity,qcfg);poses=torch.from_numpy(generated['poses'])[None].cuda()
                    _,outside=grid.normalized_reference(body_points(poses,body,execution_fast=True))
                    times=torch.from_numpy(generated['times']).cuda();plans=PlanInputs(torch.from_numpy(plan)[None].cuda(),encoded[None].cuda(),times)
                    queries=PredictionQueries(poses,times,~outside.any(-1),torch.from_numpy(generated['kinematic_valid'])[None].cuda())
                    if oracle:
                        rec=read(a.oracle_labels/'records'/(token+'.json'));file=a.oracle_labels/'scenes'/(token+'.npz')
                        if rec['identity']!=labels['identity'] or digest(file)!=rec['sha256']:raise ValueError('Changed oracle labels')
                        with np.load(file,allow_pickle=False) as l:
                            if not np.array_equal(l['poses'][0,plan_index],generated['poses']):raise ValueError('Oracle queries came from another input plan')
                            source=1 if kind.endswith('CV') else 2
                            values=np.concatenate((l['relations'][0,plan_index,0],l['relations'][0,plan_index,source,...,1:]),-1)
                            support=np.concatenate((l['common_valid'][0,plan_index],l['common_valid'][0,plan_index,...,1:]),-1)
                        kin=queries.kinematic_valid.clone();kin[:,0]=True
                        values=torch.from_numpy(values)[None].cuda()
                        if a.relations_ablation=='zero':values=torch.zeros_like(values)
                        elif a.relations_ablation=='reverse_time':values=values.flip(-2)
                        privileged=OracleRelations(values,torch.from_numpy(support)[None].cuda()&kin[...,None,None],'CV' if source==1 else 'FUTURE')
                        result=model.oracle_probe(current,plans,queries,privileged)
                    else:
                        relation=predicted_relations(predictor(execution_view(current,plans),queries)) if predictor is not None else None
                        if kind in ('R_MIN_CURRENT','R_CURRENT'):relation=torch.cat((relation[...,:3],relation[...,1:3]),-1)
                        if a.relations_ablation=='zero':relation=torch.zeros_like(relation)
                        elif a.relations_ablation=='reverse_time':relation=relation.flip(-2)
                        result=model.camera_only(current,plans,queries,relation)
                    final=decode_ego(result.float())[0].cpu().numpy()
                    if not np.isfinite(final).all():raise FloatingPointError('Nonfinite trajectory; retain failure')
                    tmp=dest.with_suffix('.tmp')
                    with tmp.open('wb') as stream:np.savez_compressed(stream,q0=plan,q_final=final,original_S0_q0=q0)
                    tmp.replace(dest);atomic_json(receipt,{'identity':ident,'token':token,'sha256':digest(dest),'plan_index':plan_index});done+=1
    if done!=expected:raise ValueError('Assigned population incomplete')
    atomic_json(a.output/'workers'/f'{a.rank:03d}_of_{a.shards:03d}.json',{'identity':ident,'scenes':done,'GPU_hours':(time.monotonic()-start)/3600.,'resource_check':resources})
    workers=[a.output/'workers'/f'{i:03d}_of_{a.shards:03d}.json' for i in range(a.shards)]
    if all(x.exists() for x in workers):
        receipts=[read(x) for x in workers]
        if sum(r['scenes'] for r in receipts)!=len(index) or any(r['identity']!=ident for r in receipts) or {p.stem for p in (a.output/'records').glob('*.json')}!=wanted:raise ValueError('Final population mismatch')
        atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':len(index),'GPU_hours':sum(r['GPU_hours'] for r in receipts),'cache_based':True,'distribution':a.distribution})


if __name__=='__main__':main()
