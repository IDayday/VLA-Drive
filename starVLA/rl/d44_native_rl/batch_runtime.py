"""Fresh multi-scene collection; every original candidate is retained."""
import copy
import time
import torch
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from .safety_credit import FACTORS
from .native_cleanx_aux import verified_index
from .fpo import velocity_energy,mc_predictions_energy
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma,clean_x_step
from starVLA.rl.simwam_recipe.contracts import seed_for


@torch.no_grad()
def deployment_chain(actor,condition,generator):
    """Exactly model.predict's clean-x arithmetic, including FP32 grid rounding."""
    grid=shift_sigma(torch.linspace(1,0,11,dtype=torch.float32),5.).to(actor.device)
    state=torch.randn((1,8,3),generator=generator).to(actor.device);states=[state.clone()]
    with actor._context_call(condition)as material:
        prefix=actor._prepare(material,False)
        for sigma,nxt in zip(grid[:-1],grid[1:]):
            with torch.autocast(actor.device.type,dtype=torch.bfloat16):
                prediction=actor.action_expert.denoise_x0(state,sigma.reshape(1),prefix)
            state=clean_x_step(state,prediction,sigma,nxt,validate=False);states.append(state.clone())
    return torch.stack(states,1).detach(),grid


@torch.no_grad()
def collect_scene(actor,descriptor,token,log,service,generator,cfg,epoch,version,*,cost=None,history=None):
    def add(**value):
        if cost is not None:cost(**value)
    def generate(count,forward):
        add(candidate_requests=count);tick=time.monotonic()
        try:
            result=forward();add(physical_candidates=count);return result
        finally:add(generation_worker_seconds=time.monotonic()-tick)
    def score(request):
        add(scorer_candidate_attempts=len(request[2]));tick=time.monotonic()
        try:return service.score(request)
        finally:add(scoring_wait_worker_seconds=time.monotonic()-tick)
    start=time.monotonic()
    g=cfg['group']
    chain,times,deltas,idx=generate(g,lambda:actor.sample_action_chain(descriptor,8,generator=generator))
    if chain.shape!=(g,11,8,3)or idx.tolist()!=[7,8,9]or not torch.isfinite(chain).all():raise ValueError('Native D chain contract changed')
    sampled=copy.deepcopy(actor.sampling_terms)
    mean,std=actor.transition_terms(descriptor,chain,times,deltas)
    error=max(float((m-mean[:,i]).abs().max())for i,(m,s)in enumerate(sampled))
    if error>1e-6 or any(abs(float(s)-float(std[i]))>1e-8 for i,(_,s)in enumerate(sampled)):
        raise FloatingPointError('Actual sampler/recompute mean or std mismatch: '+str(error))
    ref,refstd=actor.transition_terms(descriptor,chain,times,deltas,use_ref=True)
    if not torch.equal(std,refstd):raise ValueError('Reference kernel variance changed')
    torch.cuda.synchronize();generation=time.monotonic()-start
    requested=ActionCodec.denorm_odo(chain[:,-1]).cpu().numpy();begin=time.monotonic()
    records=score((token,log,requested));score_seconds=time.monotonic()-begin
    if len(records)!=g or any(r['status']!='OK'for r in records):raise ValueError('Incomplete real scorer response')
    factors=torch.tensor([[r['metrics'][k]for k in FACTORS]for r in records],device=actor.device,dtype=torch.float64)
    r=dict(token=token,log=log,condition=descriptor,chain=chain,timesteps=times,deltas=deltas,
        old_mean=torch.stack([m for m,s in sampled],1).detach(),ref_mean=ref.detach(),std=std.detach(),records=records,factors=factors,
        behavior_version=version,epoch=epoch,sampler_mean_max_error=error,
        seconds_generation=generation,seconds_scoring=score_seconds,
        physical_candidates=g,PG_candidates=g,extra_B1=0,extra_G32=0)
    if cfg['arm']in('D2','D6')and epoch>=cfg['rescue_start_epoch']:
        material=actor.materialize(descriptor);one=dict(material,context=material['context'][:1])
        nominal_generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',cfg.get('nominal_seed',2101),token))
        nominal,nominal_grid=generate(1,lambda:deployment_chain(actor,one,nominal_generator))
        nominal_records=score((token,log,ActionCodec.denorm_odo(nominal[:,-1]).cpu().numpy()))
        r['nominal_record']=nominal_records[0];r['nominal_chain']=nominal.detach()
        base=nominal_records[0]['metrics']['score'];teacher_records=records;teacher_chain=chain;r['extra_B1']=1;r['physical_candidates']+=1
        strict=verified_index(records,0.,improvement=0.)is not None
        if cfg.get('enable_teacher_expansion',True)and g==16 and base<.90 and not strict:
            extra,_,_,_=generate(g,lambda:actor.sample_action_chain(descriptor,8,generator=generator))
            extra_records=score((token,log,ActionCodec.denorm_odo(extra[:,-1]).cpu().numpy()))
            teacher_records=records+extra_records;teacher_chain=torch.cat((chain,extra));r['extra_G32']=g;r['physical_candidates']+=g
        selected=verified_index(teacher_records,base);r['nominal_score']=base;teacher=None;teacher_record=None
        if selected is not None:teacher=teacher_chain[selected:selected+1,-1].detach();teacher_record=teacher_records[selected]
        if history is not None and cfg.get('active_stage',{}).get('historical'):
            from .history import historical_teacher
            old=historical_teacher(history,token,base,version)
            if old is not None and(teacher_record is None or old['score']>teacher_record['metrics']['score']):
                teacher=torch.tensor(old['normalized_clean_action'],device=actor.device,dtype=torch.float32)[None]
                teacher_record=old['record'];r['teacher_ancestry']=old['version'];r['teacher_kind']='own_historical_nominal_B1'
        if teacher is not None and cfg.get('enable_positive',True):
            r['teacher']=teacher;r['teacher_record']=teacher_record
            r['nominal_chain']=nominal.detach()
            r['nominal_sigma_last']=nominal_grid[-2:-1].detach()
            # Frozen mixture of continuous sigma and actual deployment nodes.
            u=torch.rand((),generator=generator)
            sigma=(.1+.8*torch.rand((),generator=generator))if u<.5 else nominal_grid[torch.randint(10,(),generator=generator)].cpu()
            sigma=sigma.reshape(1).to(actor.device).float();eps=torch.randn(teacher.shape,generator=generator).to(actor.device)
            r.update(positive_sigma=sigma,positive_state=(1-sigma[:,None,None])*teacher+sigma[:,None,None]*eps)
    if cfg['arm']=='D6':r['enable_post_feedback']=cfg['active_stage']['post'];r['active_stage']=cfg['active_stage']
    if cfg['arm']in('D3_FPO','D3_AWM'):
        mc=cfg['mc_samples'];lo,hi=(.2,.8)if cfg['arm']=='D3_FPO'else(.15,.85)
        sigmas=(lo+(hi-lo)*torch.rand((mc,g),generator=generator)).to(actor.device)
        eps=torch.randn((mc,g,8,3),generator=generator).to(actor.device);clean=chain[:,-1]
        states=(1-sigmas[...,None,None])*clean[None]+sigmas[...,None,None]*eps
        targets=eps-clean[None];energies=[]
        with actor._context_call(descriptor)as condition:
            prefix=actor._prepare(condition,False)
            energies,_=mc_predictions_energy(actor.action_expert,prefix,states,sigmas,targets,
                fuse=cfg['arm']=='D3_FPO'and bool(getattr(actor,'execution_fuse_fm_mc',False)))
        r.update(fm_state=states.detach(),fm_sigma=sigmas.detach(),fm_target=targets.detach(),old_energy=energies.mean(0).detach())
    return r
