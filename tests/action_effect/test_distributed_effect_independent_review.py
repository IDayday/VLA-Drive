"""Independent CPU/Gloo replay of the NCCL component's multi-forward pattern.

Synthetic pairs test distributed autograd/reduction only, not real feedback or
the actual CUDA qualification. This file dispatches no GPU or optimizer.
"""
from pathlib import Path
import json
import multiprocessing

import pytest


def _worker(rank,init,output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from starVLA.model.modules.action_effect.effect_model import ActionEffectModel,EffectModelConfig,FeedbackLabelSpec
    from starVLA.model.modules.action_effect.effect_losses import effect_loss_contributions,combine_contributions,reduce_contributions
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    try:
        spec=FeedbackLabelSpec.from_label_arrays({'names':['score','no_at_fault_collisions','drivable_area_compliance',
            'ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance','progress_raw_meters',
            'first_at_fault_time_seconds','first_ttc_state_time_seconds','first_road_time_seconds'],
            'units':['0..1']*7+['m','s','s','s']})
        cfg=EffectModelConfig(context_dim=12,world_tokens=3,action_tokens=2,hidden_dim=16,layers=1,heads=4,init_seed=42)
        g=torch.Generator().manual_seed(7)
        context={'memory':torch.randn(8,5,12,generator=g),'ego':torch.randn(8,1,4,generator=g),
                 'mask':torch.ones(8,5,dtype=torch.bool)}
        base=torch.zeros(8,8,3);base[...,0]=torch.arange(1,9)*2.5
        after=base.clone();after[...,1]=torch.linspace(0,.3,8)
        weights=dict(value=1.,ranking=.2,events=.2,temporal=.2,uncertainty=.05,repair=.2)
        model=ActionEffectModel(cfg,spec)
        ddp=DistributedDataParallel(model,find_unused_parameters=True)
        def contribution(net,start,end,empty=False):
            prediction=net({k:v[start:end] for k,v in context.items()},base[start:end],after[start:end])
            n=end-start;before=torch.full((n,11),.4);target=before+.2
            valid=torch.ones(n,11,dtype=torch.bool);valid[:,-2:]=False
            events=torch.ones(n,3,dtype=torch.bool);temporal=torch.ones(n,41,3,dtype=torch.bool);repair=torch.ones(n,dtype=torch.bool)
            if empty:valid.zero_();events.zero_();temporal.zero_();repair.zero_()
            return effect_loss_contributions(prediction,before=before,after=target,before_valid=valid,
                after_valid=valid.clone(),label_spec=spec,mode='delta_feedback',
                event_targets=torch.tensor([[0,1,2]]*n),event_valid=events,
                temporal_targets=torch.zeros(n,41,3),temporal_valid=temporal,
                repair_targets=torch.ones(n),repair_valid=repair)
        records=[]
        for empty in (False,True):
            ddp.zero_grad(set_to_none=True)
            start=rank*4
            parts=[contribution(ddp,start+i,start+i+2,empty and rank==1) for i in (0,2)]
            reduce_contributions(combine_contributions(parts),weights,distributed=True).total.backward()
            reference=ActionEffectModel(cfg,spec)
            parts=[contribution(reference,0,4),contribution(reference,4,8,empty)]
            reduce_contributions(combine_contributions(parts),weights).total.backward()
            worst=0.;used=0
            for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
                assert name==other
                if q.grad is None:
                    assert p.grad is None or not torch.count_nonzero(p.grad)
                else:
                    assert p.grad is not None
                    torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)
                    worst=max(worst,float((p.grad-q.grad).abs().max()));used+=1
            records.append({'case':'empty_rank' if empty else 'all_valid','gradient_tensors':used,'max_abs_error':worst})
        Path(output,f'rank{rank}.json').write_text(json.dumps({'status':'PASS','cases':records}))
    except BaseException as exc:
        Path(output,f'rank{rank}.json').write_text(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}))
        raise
    finally:dist.destroy_process_group()


def test_actual_model_multi_forward_reducer_and_empty_rank_do_not_mark_ready_twice(tmp_path):
    context=multiprocessing.get_context('spawn')
    ps=[context.Process(target=_worker,args=(rank,str(tmp_path/'gloo_init'),str(tmp_path))) for rank in range(2)]
    try:
        for p in ps:p.start()
        for p in ps:
            p.join(40)
            assert not p.is_alive() and p.exitcode==0
        for rank in range(2):
            result=json.loads((tmp_path/f'rank{rank}.json').read_text())
            assert result['status']=='PASS' and len(result['cases'])==2
    finally:
        for p in ps:
            if p.is_alive():p.terminate();p.join(5)
