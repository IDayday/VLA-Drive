"""Extend the stable native trainer; PPO/FPO material and teacher paths remain separate."""
import copy
import hashlib
import json
import time
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.d44_native_rl.actor import D44Actor
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene as base_collect, deployment_chain
from starVLA.rl.d44_native_rl.config import read
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d44_native_rl.recovery import Boundary, Attempt, checkpoint
from starVLA.rl.d44_native_rl.native_cleanx_aux import verified_index, positive_loss
from starVLA.rl.d44_native_rl.fast_prefix import fast_index_select
from starVLA.rl.d44_native_rl.fpo import velocity_energy
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.flow_grpo.transactions import atomic_json
from . import history

SCHEMA='d46_verified_FPO_improvement_v1'


def inventory():
    from tools.d44_native_rl.train import code_identity
    result=code_identity()
    for root in('starVLA/rl/d46_fpo_verified','tools/d46_fpo_verified'):
        result.update({str(p):file_sha(p)for p in sorted(Path(root).rglob('*.py'))})
    return result


def compact_record(record):
    return dict(status=record['status'],metrics={k:float(v)for k,v in record['metrics'].items()})


def teacher(action,record,version,source):
    if not history.strict(record):return None
    return dict(action=action[0].detach().cpu().tolist(),record=compact_record(record),version=copy.deepcopy(version),source=source)


class VerifiedActor(D44Actor):
    def additional_objective(self,r,prefix):
        c=self.study_config['verified_improvement']
        zero=next(p for p in self.parameters()if p.requires_grad).reshape(-1)[0]*0
        cfm=last=keep=zero
        one=fast_index_select(prefix,torch.zeros(1,device=self.device,dtype=torch.long))
        if 'rescue_teacher'in r:
            many=fast_index_select(prefix,torch.zeros(c['CFM_MC'],device=self.device,dtype=torch.long))
            with torch.autocast(self.device.type,dtype=torch.bfloat16):
                pred=self.action_expert.denoise_x0(r['rescue_state'],r['rescue_sigma'],many)
            cfm=velocity_energy(r['rescue_state'],pred,r['rescue_velocity_target'],r['rescue_sigma']).mean()
        if 'rescue_teacher'in r or 'retention_target'in r:
            with torch.autocast(self.device.type,dtype=torch.bfloat16):
                pred_last=self.action_expert.denoise_x0(r['nominal_chain'][:,9],r['nominal_sigma_last'],one)
            if 'rescue_teacher'in r:last=positive_loss(pred_last,r['rescue_teacher'])
            if 'retention_target'in r:keep=positive_loss(pred_last,r['retention_target'])
        total=c['CFM_weight']*cfm+c['ODE_last_weight']*last+c['retention_weight']*r.get('retention_share',0.)*keep
        self._auxiliary_parts=dict(CFM=cfm,ODE_last=last,retention=keep)
        return total,dict(verified_CFM=cfm.detach(),verified_ODE_last=last.detach(),retention_B1=keep.detach(),
            rescue_active=zero.detach()+float('rescue_teacher'in r),retention_active=zero.detach()+float('retention_target'in r),
            retention_budget=zero.detach()+r.get('retention_share',0.),verified_auxiliary=total.detach())


class VerifiedBoundary(Boundary):
    @staticmethod
    def validate_boundary(state,batch,identity,optimizer=None,scheduler=None,**kw):
        result=Boundary.validate_boundary(state,batch,identity,optimizer,scheduler,**kw)
        bank=state.get('verified_history',{})
        owner=identity.get('training_run_id',identity['sha256'])
        if(bank.get('schema')!=history.SCHEMA or bank.get('run')!=owner
                or bank.get('commits')!=state['collections']):raise ValueError('History/run/collection commit mismatch')
        if batch:
            for r in batch['scenes']:
                if r['PG_candidates']!=16 or len(r['records'])!=16:raise ValueError('Teacher exploration leaked into FPO group')
                if 'rescue_teacher'in r:
                    if verified_index([r['rescue_record']],r['nominal_record']['metrics']['score'])is None:
                        raise ValueError('Rescue target not strictly verified/better than actual B1')
                if 'retention_target'in r:
                    history.validate_teacher(dict(action=r['retention_target'][0].detach().cpu().tolist(),
                        record=r['retention_record'],version=r['retention_version']),owner,batch['behavior_update'])
                for key in('rescue_teacher','rescue_state','rescue_sigma','rescue_velocity_target','retention_target'):
                    if key in r and(r[key].requires_grad or not torch.isfinite(r[key]).all()):raise ValueError('Nonfinite or attached teacher material')
        result['history_digest']=bank['digest']
        return result


class VerifiedExtension:
    actor_factory=VerifiedActor
    boundary=VerifiedBoundary()
    make_attempt=staticmethod(Attempt)
    def read_recipe(self,args):
        self.args=args;self.cfg=read(args.config);c=self.cfg.get('verified_improvement',{})
        if(self.cfg['arm']!='D3_FPO'or(self.cfg['global_scenes'],self.cfg['group'],self.cfg['inner_updates'])!=(64,16,2)
                or self.cfg['text_rank']!=16 or self.cfg['train_action_io']or self.cfg['film_last_layers']
                or c.get('schema')!=SCHEMA or c.get('history_capacity')!=103288 or c.get('CFM_MC')!=2
                or c.get('extra_G_as_PG')is not False or c.get('extra_G_max')!=32):raise ValueError('Unregistered verified-FPO recipe')
        if any(not 0<=c.get(k,-1)<1000 for k in('CFM_weight','ODE_last_weight','retention_weight')):raise ValueError('Invalid auxiliary coefficient')
        self.saved=None
        if args.resume:
            cp=Path(args.resume).resolve()
            if not cp.is_relative_to(Path(args.output).resolve()/'checkpoints'):raise ValueError('Only own verified-FPO native resume')
            self.saved=json.loads((cp/'identity.json').read_text())
            if self.saved.get('algorithm_schema')!=SCHEMA or self.saved['recipe']!=self.cfg:raise ValueError('Foreign improved algorithm checkpoint')
        return self.cfg
    @staticmethod
    def validate_world(world):
        if world!=8:raise ValueError('Initial improved-FPO qualification/formal is world8')
    @staticmethod
    def init_control_group():
        from starVLA.rl.simwam_recipe import checkpoint as source
        if not dist.is_initialized()or dist.get_world_size()!=8:raise ValueError('Actual world8 required')
        if source.CONTROL_GROUP is not None:raise ValueError('Existing control group')
        source.CONTROL_GROUP=dist.new_group(backend='gloo',timeout=timedelta(seconds=600))
    @staticmethod
    def make_stream(cfg):return DataStream(cfg,cfg['global_scenes'])
    def identity(self,actual):
        value=copy.deepcopy(actual);value.pop('sha256',None)
        value.update(algorithm_schema=SCHEMA,experiment_id='FPO_VERIFIED_B1_v1',source_files=inventory(),
            auxiliary_protocol='TRAIN-only real B1 and verified initial-IL/own-history; teachers excluded from FPO ratio; sigma .1-.9 MC2 CFM + real last-ODE SmoothL1',
            auxiliary_RNG='stateless SHA256(run,collection,scene) CPU generator, independent of main sampler')
        value['sha256']=fingerprint(value)
        if self.saved is not None and value!=self.saved:raise ValueError('Own native actor/data/source/config changed')
        return value
    @staticmethod
    def initialize_state(state,identity):state['verified_history']=history.initial(identity['sha256'])
    def after_restore(self,actor,conditions,stream,state,identity):
        self.actor=actor;self.conditions=conditions;self.stream=stream;self.identity_value=identity
        if self.saved is not None and self.saved['world']!=dist.get_world_size():raise ValueError('Cannot change native world silently')
        self._restored_state=state
    def configure_actor(self,actor):
        from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
        actor.execution_fast_prefix=True;actor.execution_fuse_fm_mc=False
        self.caches=[install_invariant_cache(x,rotary=True,additive=True)for x in(actor.action_expert.decision,actor.ref_action_expert.decision)]
        if self.args.qualification and self.args.resume:
            # Same qualified execution, actual saved material, no optimizer step.
            self._calibrate(self._restored_state)
    def before_collection(self,state,generator,rank,world):
        self.bank=state['verified_history'];self.rows=[]
    def before_scene(self,generator,j):pass
    def after_scene(self,generator,j):pass
    @torch.no_grad()
    def collect_scene(self,actor,descriptor,token,log,service,generator,cfg,epoch,version,*,cost=None,history=None):
        from . import history as bank_history
        collector=getattr(self,'main_collector',base_collect)
        r=collector(actor,descriptor,token,log,service,generator,cfg,epoch,version,cost=cost)
        # Base G16 and its MC4 behavior energy are frozen before extra draws.
        c=cfg['verified_improvement'];material=actor.materialize(descriptor);one=dict(material,context=material['context'][:1])
        def nominal(ref=False):
            if cost:cost(candidate_requests=1)
            tick=time.monotonic();chain,grid=deployment_chain(actor,one,torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token)),use_ref=ref)
            if cost:cost(physical_candidates=1,generation_worker_seconds=time.monotonic()-tick,scorer_candidate_attempts=1)
            tick=time.monotonic();record=service.score((token,log,ActionCodec.denorm_odo(chain[:,-1]).cpu().numpy()))[0]
            if cost:cost(scoring_wait_worker_seconds=time.monotonic()-tick)
            if record['status']!='OK':raise ValueError('Invalid actual nominal B1 scorer result')
            r['extra_B1']+=1;r['physical_candidates']+=1
            return chain,grid,record
        nominal_chain,grid,record=nominal()
        r.update(nominal_chain=nominal_chain,nominal_sigma_last=grid[-2:-1],nominal_record=compact_record(record))
        current=teacher(nominal_chain[:,-1],record,version,'own_current_behavior_B1')
        initial_teacher=None
        if token not in self.bank['rows']:
            if version['update']==0:il_chain,il_record=nominal_chain,record
            else:il_chain,_,il_record=nominal(ref=True)
            initial_version=dict(run=version['run'],update=0,collection=0)
            initial_teacher=teacher(il_chain[:,-1],il_record,initial_version,'own_verified_initial_IL46_B1')
            r.update(initial_IL_record=compact_record(il_record),initial_IL_action=il_chain[:,-1].detach())
        best=bank_history.best_teacher(self.bank['teachers'].get(token),initial_teacher,current,run=version['run'],update=version['update'])
        if best is not None:
            r.update(retention_target=torch.tensor(best['action'],device=actor.device,dtype=torch.float32)[None],
                retention_record=best['record'],retention_version=best['version'],retention_kind=best['source'],
                retention_share=bank_history.retention_share(best,record['metrics']['score']))
        seed=int.from_bytes(hashlib.sha256(f"{version['run']}:{version['collection']}:{token}".encode()).digest()[:8],'little')%(2**63-1)
        auxiliary=torch.Generator().manual_seed(seed)
        teachers=r['records'];actions=r['chain'][:,-1]
        if not bank_history.strict(record):
            if verified_index(teachers,0.,improvement=0.)is None:
                if cost:cost(candidate_requests=16)
                tick=time.monotonic();extra,*_=actor.sample_action_chain(descriptor,8,generator=auxiliary)
                if cost:cost(physical_candidates=16,generation_worker_seconds=time.monotonic()-tick,scorer_candidate_attempts=16)
                tick=time.monotonic();scored=service.score((token,log,ActionCodec.denorm_odo(extra[:,-1]).cpu().numpy()))
                if cost:cost(scoring_wait_worker_seconds=time.monotonic()-tick)
                if len(scored)!=16 or any(x['status']!='OK'for x in scored):raise ValueError('Incomplete extra teacher-only group')
                teachers=teachers+scored;actions=torch.cat((actions,extra[:,-1]));r['extra_G32']=16;r['physical_candidates']+=16
            selected=verified_index(teachers,record['metrics']['score'])
            if selected is not None:
                target=actions[selected:selected+1].detach();sigma=(.1+.8*torch.rand((c['CFM_MC'],),generator=auxiliary)).to(actor.device)
                noise=torch.randn((c['CFM_MC'],8,3),generator=auxiliary).to(actor.device)
                r.update(rescue_teacher=target,rescue_record=compact_record(teachers[selected]),rescue_sigma=sigma,
                    rescue_state=((1-sigma[:,None,None])*target+sigma[:,None,None]*noise).detach(),rescue_velocity_target=(noise-target).detach())
        self.rows.append(r)
        r['teacher_query_counts']=dict(nominal_and_initial_IL=r['extra_B1'],extra_teacher_only=r['extra_G32'])
        return r
    def after_collection(self,state,generator,rank,world):
        deltas=[]
        for r in self.rows:
            if r['padding']:continue
            best=None
            if 'retention_target'in r:best=dict(action=r['retention_target'][0].cpu().tolist(),record=r['retention_record'],version=r['retention_version'],source=r['retention_kind'])
            deltas.append(dict(token=r['token'],version=r['behavior_version'],nominal_score=r['nominal_record']['metrics']['score'],teacher=best))
        all_deltas=[r for part in checkpoint.gather(deltas)for r in part]
        history.commit(state['verified_history'],all_deltas,set(self.stream.tokens),capacity=self.cfg['verified_improvement']['history_capacity'])
    @staticmethod
    def json_result(state):
        result=dict(state)
        if 'verified_history'in result:
            h=result['verified_history'];result['verified_history']=dict(schema=h['schema'],run=h['run'],commits=h['commits'],
                scenes=len(h['rows']),teachers=len(h['teachers']),digest=h['digest'],full_state='native rank checkpoint')
        return result
    def _calibrate(self,state):
        cp=Path(self.args.resume);rank=dist.get_rank();saved=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        batch=saved['rollout'];rows=[]
        if batch is not None:
            parameters=[p for p in self.actor.parameters()if p.requires_grad]
            for r in batch['scenes']:
                if 'rescue_teacher'not in r:continue
                r=checkpoint.device_tree(r,self.actor.device);r['expect_fresh']=False
                _,metrics,parts=self.actor(r,return_components=True);terms=dict(FPO=parts['policy'],reference=parts['reference'],**self.actor._auxiliary_parts)
                norms={}
                for name,loss in terms.items():
                    grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                    norms[name]=float(torch.stack([g.float().square().sum()for g in grads if g is not None]).sum().sqrt())
                rows.append(dict(token=r['token'],raw_gradient_norms=norms,actual_teacher_improvement=r['rescue_record']['metrics']['score']-r['nominal_record']['metrics']['score']))
                break
        atomic_json(Path(self.args.output)/f'gradient_calibration_rank{rank}.json',dict(status='MEASURED'if rows else'NO_RESCUE_IN_THIS_RANK_BATCH',
            optimizer_steps=0,new_PDM_queries=0,checkpoint=str(cp),rows=rows))
