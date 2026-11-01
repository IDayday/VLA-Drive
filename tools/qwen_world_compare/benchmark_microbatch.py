"""Real native-weight rank-slice layout comparison; no optimizer or formal save.

Reuses the qualified execution benchmark's model/data/gradient implementation.
This does not establish an eight-rank speedup or authorize production migration.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

from tools.qwen_world_compare.benchmark_execution import (
    torch,write,compare_gradients,build_model,checkpoint_flags,apply_checkpoint_policy,
    WorldSceneData,QwenObservationProcessor,TeacherCacheReader,CandidateBank,
    FutureTrackStore,EvidenceBank,BatchBuilder,to_device,update_denominators,configure_numerics)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--larger-micro',type=int,choices=(2,4),required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    snap=json.loads(Path(a.snapshot).read_text());identity=snap['identity'];cfg=identity['config'];old=identity['execution']['micro'];new=a.larger_micro
    if new<=old or 4%new:raise ValueError('Keep each original rank four scenes and global32')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');configure_numerics();torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda')
    data=WorldSceneData(Path(cfg['data_root'])/'TRAIN',role='TRAIN');processor=QwenObservationProcessor(cfg['qwen_path'])
    teachers=TeacherCacheReader(cfg['teacher_cache'],expected_cohort='TRAIN');bank=CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',))
    tracks=FutureTrackStore(Path(cfg['target_root'])/'TRAIN');evidence=EvidenceBank(cfg['evidence_root']);builder=BatchBuilder(data,processor,teachers,bank,tracks,evidence)
    update=snap['updates'];cursor=update*32;batches={};control={};start=time.monotonic();denoms=None
    for micro in(old,new):
        all_batches=[]
        for rank in range(8):
            for m in range(4//micro):
                positions=[rank*4+m*micro+j for j in range(micro)]
                tokens=[data.tokens[data.index(cursor+i)]for i in positions]
                all_batches.append(builder.build(tokens,positions,update))
        if denoms is None:denoms=update_denominators(all_batches,torch.device('cpu'))
        else:
            alternate=update_denominators(all_batches,torch.device('cpu'))
            if any(not torch.equal(denoms[k],alternate[k])for k in denoms):raise ValueError('Effective loss denominator changed')
        control[micro]={k:torch.cat([b[k]for b in all_batches],0)for k in['target','noise','sigma','action_valid','future_queries','future_target','future_valid']}
        batches[micro]=all_batches[:4//micro]
    if any(not torch.equal(control[old][k],control[new][k])for k in control[old]):raise ValueError('Global32 scene/noise/query/target identity changed')
    del control
    denoms={k:v.to(device)/8 for k,v in denoms.items()}
    write(out/'DATA_READY.json',dict(status='PASS',data_seconds=time.monotonic()-start,global_scene_count=32,
        unchanged_scene_noise_sigma_query_targets=True,old_micro=old,new_micro=new,rank0_scenes=4))
    print('DATA_READY',flush=True);start=time.monotonic()
    raw=torch.load(snap['model'],map_location='cpu',weights_only=False)
    model=build_model(cfg,checkpoint_config=copy.deepcopy(identity['qwen_config']),train_heads=True)
    model.load_state_dict(raw['module'],strict=True);del raw
    model.to(device).train();model.configure_stage(False)
    apply_checkpoint_policy(model,'all_off',checkpoint_flags(model))
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise TypeError('Model storage precision changed')
    versions={n:p._version for n,p in model.named_parameters()};setup=time.monotonic()-start
    rows=[];reference=None;gradient=None;reference_losses=None
    for index,micro in enumerate([old,new]*3):
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone();torch.cuda.synchronize();start=time.monotonic();values={}
        for batch in batches[micro]:
            batch=to_device(batch,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model(batch,denominators=denoms,weights=cfg['il']['loss_weights'])
            result['loss'].backward()
            for k,v in result['losses'].items():values[k]=values.get(k,0.)+float(v)
            del result,batch
        torch.cuda.synchronize();row=dict(index=index,micro=micro,warmup=index<2,seconds=time.monotonic()-start,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,losses=values,
            RNG_unchanged=torch.equal(rng,torch.get_rng_state())and torch.equal(cuda_rng,torch.cuda.get_rng_state()))
        rows.append(row);write(out/'PROGRESS.json',dict(rows=rows));print(json.dumps(row),flush=True)
        if index==2:
            reference={n:None if p.grad is None else p.grad.detach().cpu()for n,p in model.named_parameters()};reference_losses=values
        if index==3:
            gradient=compare_gradients(model,reference);del reference;reference=None
            gradient['max_loss_component_absolute_error']=max(abs(reference_losses[k]-values[k])for k in values)
    unchanged=all(p._version==versions[n]for n,p in model.named_parameters())
    medians={str(m):statistics.median(r['seconds']for r in rows if r['micro']==m and not r['warmup'])for m in(old,new)}
    equivalent=(not gradient['missing_gradient_mismatch']and gradient['relative_L2']<=1e-5
        and gradient['max_loss_component_absolute_error']<=1e-6 and unchanged and all(r['RNG_unchanged']for r in rows))
    write(out/'RESULT.json',dict(status='PASS'if equivalent else'LAYOUT_NOT_EQUIVALENT_AT_REGISTERED_TOLERANCE',
        architecture=cfg['architecture_id'],snapshot=a.snapshot,model_sha256=snap['model_sha256'],
        old_micro=old,new_micro=new,model_setup_seconds=setup,rows=rows,median_rank_slice_seconds=medians,
        rank_slice_speedup=medians[str(old)]/medians[str(new)],gradient_comparison=gradient,
        thresholds=dict(relative_gradient_L2=1e-5,loss_absolute=1e-6),all_parameters_unchanged=unchanged,
        optimizer_updates=0,scorer_calls=0,parameter_gradient_dtype='float32',forward_dtype='bfloat16',
        caveat='One-rank F/B slice excluding optimizer/collectives; even PASS needs native8rank optimizer/resume qualification before formal change'))


if __name__=='__main__':main()
