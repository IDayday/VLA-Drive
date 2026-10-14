"""Real-image, cached-context and new-process research-bundle qualification."""
import argparse
import json
import time
import numpy as np
import torch
from iqe.contracts import require, FeatureBundle
from iqe.io import atomic_json, atomic_torch, file_hash, read_json
from iqe.evaluation.retention import module_hash
from iqe.data.sources import load_observation
from iqe.robust_scorer.qualification import _capture, compare_outputs, TOLERANCES
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.context_batch_layout import OriginalBatchReplay
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.scorer_training import CONTEXT_KEYS
from iqe.specialist_study.training import runtime
from .bundle import load as load_bundle
from .config import load


@torch.no_grad()
def online_cache(config,system,device):
    source=config.source();selected=scenes_by_role(source)['stage_val'][:4]
    layout=OriginalBatchReplay(source);wanted={s.scene_id for s in selected}
    groups=list(dict.fromkeys(layout.batch_for(s.scene_id) for s in selected))
    cache=ReusedContext(source,16);spatial=VerifiedSpatialReader(source,16);rows=[]
    for group in groups:
        scenes=[layout.scenes[sid] for sid in group]
        require(all(s.split_role!='final_test' for s in scenes), 'forbidden inference qualification role')
        observations=[load_observation(s,system.adapter.contract) for s in scenes]
        output,captured=_capture(system,observations)
        indices=[i for i,s in enumerate(scenes) if s.scene_id in wanted]
        sub=[scenes[i] for i in indices];x=cache.batch(sub,device)
        planner=FeatureBundle(x['planner_scene'],x['planner_ego'],x['planner_mask'],
            system.adapter.contract_hash,tuple(s.scene_id for s in sub))
        base=system.adapter.predict_s0(planner)
        private={k:x[k] for k in ('planner_scene','planner_ego','planner_mask','world','action','current_ego')}
        candidates=torch.stack([base.physical]+[m(private).physical for m in system.experts],1)
        context={k:x[k] for k in CONTEXT_KEYS};context.update(spatial.batch(sub,device))
        predicted=system.scorer(context,candidates)['values'];differences={}
        for name,value in context.items():
            actual=value.cpu();expected=captured['context'][name][indices]
            require(actual.shape==expected.shape and actual.dtype==expected.dtype, 'cached schema mismatch: '+name)
            if actual.dtype==torch.bool:
                require(torch.equal(actual,expected), 'cached mask mismatch: '+name);differences[name]=0.
            else:
                differences[name]=float((actual-expected).abs().max())
                require(differences[name]<=TOLERANCES['context'], 'cached context drift: '+name)
        differences['raw']=float((base.raw.cpu()-captured['raw'][indices]).abs().max())
        differences['physical']=float((candidates.cpu()-output['candidates'][indices].cpu()).abs().max())
        differences['score']=float((predicted.cpu()-output['predicted_scores'][indices].cpu()).abs().max())
        require(all(differences[k]<=TOLERANCES[k] for k in ('raw','physical','score')), 'cached prediction drift')
        require(torch.equal(predicted.argmax(1).cpu(),output['selected'][indices].cpu()), 'cached choice changed')
        polluted=[x|dict(GT=torch.tensor(float('nan')),future_actors=object(),score=object()) for x in observations]
        changed,new=_capture(system,polluted);compare_outputs(changed,output)
        require(all(torch.equal(v,new['context'][k]) for k,v in captured['context'].items()), 'labels changed scorer input')
        require(torch.equal(captured['raw'],new['raw']), 'labels changed Base output')
        rows.append(dict(scene_ids=[s.scene_id for s in sub],canonical_batch=list(group),differences=differences,
            label_independence=True,calls=output['calls'],branch_calls=output['branch_calls'],cache_VLA_calls=0))
    return dict(status='PASS',rows=rows,tolerances=TOLERANCES,batch_shape_invariance_claimed=False,
        original_batch_plan_hash=layout.plan_hash,real_images=True)


@torch.no_grad()
def benchmark(config,system,device):
    require(device.startswith('cuda'), 'CUDA timing requires a CUDA device')
    scenes=scenes_by_role(config.source())['stage_val'][:32];count=len(system.experts);rows=[]
    for n,score in ((0,False),(count,False),(count,True)):
        times=[];head=[]
        torch.cuda.reset_peak_memory_stats()
        for at in range(35):
            events=[]
            def event(*_):
                x=torch.cuda.Event(enable_timing=True);x.record();events.append(x)
            hooks=[system.scorer.register_forward_pre_hook(event),system.scorer.register_forward_hook(event)] if score else []
            try:
                torch.cuda.synchronize();start=time.perf_counter()
                obs=load_observation(scenes[at%len(scenes)],system.adapter.contract)
                result=system.shared_forward([obs],expert_count=n,score=score)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
            finally:
                for h in hooks:h.remove()
            require(result['calls']==dict(visual=1,language=1), 'extra VLA forward')
            require(result['branch_calls']==dict(Base_decoder=1,expert_decoders=[1]*n,
                score_context_projections=int(score),spatial_reader=int(score)), 'private/context calls changed')
            if at>=3:
                times.append(elapsed)
                if score:
                    require(len(events)==2,'Scorer timing hook count');head.append(events[0].elapsed_time(events[1]))
        rows.append(dict(K=n+1,scorer=score,batch=1,warmup=3,repeats=32,mean_s=float(np.mean(times)),
            p50_s=float(np.median(times)),p95_s=float(np.quantile(times,.95)),
            scorer_mean_ms=float(np.mean(head)) if head else None,
            scorer_p50_ms=float(np.median(head)) if head else None,
            scorer_p95_ms=float(np.quantile(head,.95)) if head else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            all_bundle_weights_resident=True,real_images_and_preprocessing=True))
    return dict(rows=rows,hardware=torch.cuda.get_device_name(),CUDA_synchronization='before/after full forward; events around head',
        VLM_parameters=sum(p.numel() for p in system.adapter.framework.qwen_vl_interface.parameters()),
        expert_parameters=[sum(p.numel() for p in m.parameters()) for m in system.experts],
        scorer_parameters=sum(p.numel() for p in system.scorer.parameters()))


def run(config,command,device):
    from iqe.specialist_study.resource_guard import check
    check(config,'benchmark-system');device=runtime(device)
    folder=config.root/'qualification';path=folder/(command+'.json')
    bundle=config.root/'research_bundle/bundle.pt'
    deps=dict(bundle_hash=file_hash(bundle),choice_hash=file_hash(config.root/'MODEL_CHOICE.json'))
    if path.exists():
        old=read_json(path);require(old['dependencies']==deps,'stale qualification');return old
    system=load_bundle(bundle,device);scene=scenes_by_role(config.source())['stage_val'][0]
    if command=='roundtrip':
        receipt=read_json(folder/'REFERENCE.json')
        require(receipt['dependencies']==deps and receipt['observation_hash']==scene.observation_hash
            and file_hash(folder/'REFERENCE.pt')==receipt['checksum'], 'roundtrip input identity changed')
        reference=torch.load(folder/'REFERENCE.pt',map_location='cpu',weights_only=True)
        actual=system.shared_forward([load_observation(scene,system.adapter.contract)])
        evidence=dict(errors=compare_outputs(actual,reference),separate_CLI_process=True)
    else:
        before=module_hash(system)
        evidence=dict(online_cache=online_cache(config,system,device),timing=benchmark(config,system,device))
        actual=system.shared_forward([load_observation(scene,system.adapter.contract)])
        atomic_torch(folder/'REFERENCE.pt',{k:actual[k].cpu() for k in
            ('trajectory','candidates','predicted_scores','selected')},immutable=True)
        atomic_json(folder/'REFERENCE.json',dict(dependencies=deps,observation_hash=scene.observation_hash,
            checksum=file_hash(folder/'REFERENCE.pt')),immutable=True)
        require(module_hash(system)==before, 'qualification changed weights/buffers')
    result=dict(status='PASS',dependencies=deps,evidence=evidence,ACTIVE_updated=False,test_or_navtest_access=False)
    atomic_json(path,result,immutable=True);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('online','roundtrip'))
    parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    print(json.dumps(run(load(args.config),args.command,args.device),allow_nan=False))


if __name__=='__main__':main()
