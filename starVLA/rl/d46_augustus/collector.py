"""Original PDM scorer, actual three-step JaT behavior, detached native material."""
import time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.d44_native_rl.safety_credit import FACTORS
from .config import kernel_identity


@torch.no_grad()
def collect_scene(actor,descriptor,token,log,service,generator,cfg,epoch,version,*,cost=None,history=None):
    def add(**values):
        if cost is not None:cost(**values)
    g=cfg['group'];start=time.monotonic();add(candidate_requests=g)
    chain,grid,deltas,indices=actor.sample_action_chain(descriptor,generator=generator)
    add(physical_candidates=g);saved=actor.augustus_sampling
    if chain.shape!=(g,4,8,3)or indices.tolist()!=[0,1,2]or not torch.isfinite(chain).all():
        raise ValueError('AUGUstus three-step native action contract changed')
    if not torch.equal(chain[:,0],chain[:1,0].expand_as(chain[:,0])):
        raise ValueError('Candidates must share the same initial root')
    references=[];reference_means=[]
    from .kernel import step
    with actor._context_call(descriptor)as material:
        prefix=actor._prepare(material,True)
        for k,(sigma,nxt)in enumerate(zip(grid[:-1],grid[1:])):
            x0=actor.predict_clean(chain[:,k],sigma.expand(g),prefix,use_ref=True)
            _,mean,std=step(x0,chain[:,k],sigma.expand(g),nxt.expand(g),
                            noise_scale=cfg['augustus']['noise_scale'],noise=torch.zeros_like(x0))
            if not torch.equal(std,saved['std'][:,k]):raise ValueError('Current/reference kernel variance changed')
            references.append(x0);reference_means.append(mean)
    if actor.device.type=='cuda':torch.cuda.synchronize(actor.device)
    generation=time.monotonic()-start;add(generation_worker_seconds=generation)
    requested=ActionCodec.denorm_odo(chain[:,-1]).cpu().numpy();tick=time.monotonic()
    add(scorer_candidate_attempts=g)
    try:records=service.score((token,log,requested))
    finally:add(scoring_wait_worker_seconds=time.monotonic()-tick)
    scoring=time.monotonic()-tick
    if len(records)!=g or any(r['status']!='OK'for r in records):raise ValueError('Incomplete official candidate scores')
    factors=torch.tensor([[r['metrics'][k]for k in FACTORS]for r in records],device=actor.device,dtype=torch.float64)
    return dict(token=token,log=log,condition=descriptor,chain=chain.detach(),timesteps=grid.detach(),deltas=deltas.detach(),
                old_mean=saved['old_mean'],old_clean_x=saved['old_clean_x'],std=saved['std'],
                ref_clean_x=torch.stack(references,1).detach(),ref_mean=torch.stack(reference_means,1).detach(),
                augustus_kernel=kernel_identity(cfg),
                factors=factors,records=records,seconds_generation=generation,seconds_scoring=scoring,
                behavior_version=version,epoch=epoch,sampler_mean_max_error=0.,physical_candidates=g,PG_candidates=g,
                extra_B1=0,extra_G32=0)
