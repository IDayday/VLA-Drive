"""Thin IEG arm, loss step and strict head-only qualification helpers.

The shared action_effect.training loop owns optimizer/scheduler/checkpoints/RNG
and sampling; this module never creates an independent training framework.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf

from .common import artifact,verify_artifact,read_json,write_json,digest,load_config
from tools.st_world_policy.base_binding import read_binding
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.modules.action_effect.effect_flow_adapter import EffectFlowAdapter,EffectFlowAdapterConfig
from starVLA.model.modules.action_effect.effect_distillation import PairedFlowSample,sample_original_pair,velocity_loss
from starVLA.model.framework.effect_internalized_flow import EffectInternalizedFlow


def load_original_action_head(receipt_path,binding_path,*,device):
    """Read only the canonical extracted action branch; never instantiate Qwen."""
    binding=read_binding(binding_path)
    receipt=read_json(receipt_path)
    if (receipt.get('schema')!='action_effect_original_action_head_v1' or
            receipt.get('base_binding_hash')!=binding['binding_sha256'] or
            receipt.get('strict_original_master') is not True):
        raise ValueError('Foreign/noncanonical original action-head receipt')
    path=verify_artifact(receipt['artifact'])
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (set(payload)!={'schema','base_binding_hash','config','state_dict','inference_contract'} or
            payload['schema']!=receipt['schema'] or payload['base_binding_hash']!=binding['binding_sha256'] or
            payload['inference_contract']!=binding['inference_contract']):
        raise ValueError('Original head body/receipt contract mismatch')
    cfg=OmegaConf.create(payload['config']);expected=binding['parent_identity']['config']
    action=cfg.framework.action_model
    original=expected['framework']['action_model']
    # QwenOFT constructs DiTConfig dynamically; do not infer it from DiT-B name.
    if cfg.framework.name!='DDPActionVideoForesight' or cfg.framework.qwenvl.vl_hidden_dim!=2048:
        raise ValueError('Wrong original VLA/conditioning architecture')
    for key in ('action_dim','action_horizon','num_inference_timesteps','noise_beta_alpha',
                'noise_beta_beta','noise_s','num_timestep_buckets','repeated_diffusion_steps','hidden_size'):
        if action[key]!=original[key]:raise ValueError('Changed action contract: '+key)
    actual_dit=OmegaConf.to_container(action.DiTConfig,resolve=True)
    wanted={'num_layers':original['diffusion_model_cfg']['num_layers'],
        'input_embedding_dim':original['hidden_size'],'attention_head_dim':64,
        'num_attention_heads':original['hidden_size']//64}
    if actual_dit!=wanted:raise ValueError('Missing/wrong actual constructed DiTConfig')
    for name,value in payload['state_dict'].items():
        if not isinstance(value,torch.Tensor):raise ValueError('Non-tensor head state: '+name)
        if value.is_floating_point() and (value.dtype!=torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Head must contain finite canonical FP32 master values: '+name)
    model=get_action_model(cfg).float()
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device).eval(),{'head_receipt':artifact(receipt_path),
        'head_artifact':receipt['artifact'],'base_binding_sha256':binding['binding_sha256'],
        'config_sha256':digest(payload['config']),'strict_load':True,'Qwen_instantiated':False}


@dataclass(frozen=True)
class FlowLossRecipe:
    repeat: int = 8
    microbatch_scenes: int = 8
    velocity: float = 1.
    preserve: float = 1.
    effect: float = .2
    direction: float = .2

    def validate(self):
        if self.repeat!=8 or type(self.microbatch_scenes) is not int or self.microbatch_scenes<1:
            raise ValueError('Original repeat8 and positive scene microbatch are required')
        for v in (self.velocity,self.preserve,self.effect,self.direction):
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:
                raise ValueError('Explicit finite nonnegative loss weights required')
        if self.velocity<=0:raise ValueError('The original FM target cannot be disabled')
        return self


def build_flow_arm(head,*,arm,init_seed,hidden=256,layers=2,heads=4):
    adapter=EffectFlowAdapter(EffectFlowAdapterConfig(condition_dim=2048,hidden_dim=hidden,
        heads=heads,layers=layers,init_seed=init_seed))
    return EffectInternalizedFlow(head,adapter,arm=arm)


class FlowStep:
    """Shared-loop step_fn: one global draw, microbatch sums, then backward.

    Examples are selected immutable repair/replay records, not a new EffectPair
    storage schema. A caller's data adapter supplies scene_id,target_raw,
    is_repair,is_preserve and (B1-effect only) effect_delta/effect_valid. Raw Base
    and all deployable context are loaded solely from the bound ContextReader.
    """
    def __init__(self,examples,record_loader,*,recipe,device,noise_generator,time_generator,
                 context_batcher,teacher=None,effect_scales=None):
        self.examples=examples;self.record_loader=record_loader;self.recipe=recipe.validate()
        self.device=torch.device(device);self.noise_generator=noise_generator;self.time_generator=time_generator
        self.context_batcher=context_batcher;self.teacher=teacher;self.effect_scales=effect_scales

    def __call__(self,model,indices,optimizer_step):
        if not indices:raise ValueError('Empty optimizer scene batch')
        rows=[self.examples[i] for i in indices]
        records=[self.record_loader(x['scene_id']) for x in rows]
        if any(r['role']!='fit' for r in records):raise ValueError('Validation/dev/test record entered IEG backward')
        if any(type(x['is_repair']) is not bool or type(x['is_preserve']) is not bool for x in rows):
            raise ValueError('Explicit repair/preserve label masks required')
        if any(x['is_repair']==x['is_preserve'] for x in rows):
            raise ValueError('Every scene must be exactly one repair or identity keep')
        condition=torch.stack([r['condition'] for r in records]).to(self.device).detach()
        minus=torch.stack([r['raw'] for r in records]).to(self.device).detach()
        plus=torch.stack([torch.as_tensor(r['target_raw'],dtype=torch.float32) for r in rows]).to(self.device).detach()
        repair=torch.tensor([x['is_repair'] for x in rows],device=self.device,dtype=torch.bool)
        keep=torch.tensor([x['is_preserve'] for x in rows],device=self.device,dtype=torch.bool)
        if not torch.equal(plus[keep],minus[keep]):
            raise ValueError('Identity keep must retain the exact original raw target')
        count=len(rows);repeat=self.recipe.repeat
        # Draw once over the complete repeat-major optimizer population so a
        # changed microbatch topology never changes noise/time assignment.
        full=sample_original_pair(minus.repeat(repeat,1,1),plus.repeat(repeat,1,1),
            noise_generator=self.noise_generator,time_generator=self.time_generator)
        velocity_denom=int(repair.sum())*repeat*8*4
        preserve_denom=int(keep.sum())*repeat*8*4
        if velocity_denom==0 and preserve_denom==0:
            raise ValueError('No valid repair or preservation supervision; refusing empty optimizer update')
        effect_delta=effect_valid=None;effect_denom=direction_denom=None
        if model.arm=='B1-effect':
            if self.teacher is None or self.effect_scales is None:
                raise ValueError('B1-effect needs this campaign verified frozen G1 and fixed scales')
            effect_delta=torch.stack([torch.as_tensor(x['effect_delta'],dtype=torch.float32) for x in rows]).to(self.device)
            effect_valid=torch.stack([torch.as_tensor(x['effect_valid'],dtype=torch.bool) for x in rows]).to(self.device)&repair[:,None]
            effect_denom=int(effect_valid.sum())*repeat
            scales=torch.as_tensor(self.effect_scales,device=self.device)
            direction_denom=float((effect_delta.abs()/scales*effect_valid*(effect_delta.abs()/scales>1e-6)).sum())*repeat
        totals={'velocity':0.,'preserve':0.,'effect_consistency':0.,'effect_direction':0.}
        for at in range(0,count,self.recipe.microbatch_scenes):
            end=min(at+self.recipe.microbatch_scenes,count)
            # Same selected scenes from each repeat, maintaining original noise.
            selector=torch.cat([torch.arange(at,end,device=self.device)+r*count for r in range(repeat)])
            pair=PairedFlowSample(full.minus[selector],full.plus[selector],full.noise[selector],full.times[selector])
            cond=condition[at:end].repeat(repeat,1,1)
            kwargs={}
            if model.arm=='B1-effect':
                context=self.context_batcher(records[at:end],self.device)
                kwargs=dict(teacher=self.teacher,true_delta=effect_delta[at:end].repeat(repeat,1),
                    effect_valid=effect_valid[at:end].repeat(repeat,1),effect_scales=self.effect_scales,
                    effect_global_denominator=effect_denom,direction_global_denominator=direction_denom,
                    effect_context={k:x.repeat((repeat,)+(1,)*(x.ndim-1)) for k,x in context.items()})
            result=model(cond,pair,repair[at:end].repeat(repeat),global_denominator=velocity_denom,**kwargs)
            losses=result['losses']
            # A keep has plus==minus, hence both arms already evaluated its
            # original shared path. Reuse that velocity; no second Base pass.
            preserve=velocity_loss(result['prediction'],result['base_velocity'],
                keep[at:end].repeat(repeat),global_denominator=preserve_denom)
            loss=(self.recipe.velocity*losses['velocity']+self.recipe.preserve*preserve)
            if model.arm=='B1-effect':
                loss=loss+self.recipe.effect*losses['effect_consistency']+self.recipe.direction*losses['effect_direction']
            loss.backward()
            totals['velocity']+=float(losses['velocity'].detach());totals['preserve']+=float(preserve.detach())
            for k in ('effect_consistency','effect_direction'):
                if k in losses:totals[k]+=float(losses[k].detach())
        return {**totals,'repair_scenes':int(repair.sum()),'preserve_scenes':int(keep.sum()),
            'unique_scene_ids':len({r['scene_id'] for r in records}),'scene_presentations':count,
            'FM_noise_presentations':count*repeat,'optimizer_step':optimizer_step}


def load_flow_examples(repair_dataset,reader):
    """Thin adapter for the actual shared G0 REPAIR_DATASET, preserving evidence."""
    import numpy as np
    from starVLA.dataloader.foresight_dataset import encode_ego
    from .common import verify_identity
    dataset=verify_identity(read_json(repair_dataset))
    if (dataset.get('schema')!='action_effect_repair_dataset_v1' or dataset.get('status')!='COMPLETE'
            or dataset.get('split')!='train' or dataset.get('deployment_inputs') is not False):
        raise ValueError('Complete legal shared G0 repair dataset required')
    registration=verify_identity(read_json(verify_artifact(dataset['registration'])))
    completion=read_json(verify_artifact(dataset['population_completion']))
    if (registration['identity']!=dataset['registration_identity'] or
            dataset['base_checkpoint']!=reader.binding['base_checkpoint'] or
            registration['binding_identity']!=reader.binding['identity'] or
            completion['registration_identity']!=registration['identity']):
        raise ValueError('Repair dataset and frozen context/Base differ')
    if len({r['scene_id'] for r in dataset['entries']})!=len(dataset['entries']):
        raise ValueError('Duplicated G0 repair scene')
    examples=[];counts={'requested':len(dataset['entries']),'excluded_validation':0,'untrainable':0,'repairs':0,'preserve':0}
    label_spec=None
    for row in dataset['entries']:
        if row['role']!='fit':counts['excluded_validation']+=1;continue
        record=reader[row['scene_id']]
        if (record['role']!='fit' or record['source_log']!=row['source_log'] or
                record['observation_hash']!=row['observation_hash'] or
                row['context_entry']!=reader.rows[row['scene_id']]):
            raise ValueError('G0/current context observation/log/role join differs')
        base=torch.as_tensor(row['base_physical'],dtype=torch.float32)
        if not torch.equal(base,record['physical'].float()):raise ValueError('G0 Base trajectory differs from actual context')
        if row['trainable'] is not True:counts['untrainable']+=1;continue
        kind=row['target_kind']
        if kind not in ('identity_preservation','verified_local_repair'):
            raise ValueError('No fabricated/synthetic/unverified repair target admitted')
        before,after=row['base_labels'],row['selected_labels']
        if before['names']!=after['names'] or before['units']!=after['units']:
            raise ValueError('Base/repair physical-effect schemas differ')
        spec={'names':before['names'],'units':before['units']}
        if label_spec is None:label_spec=spec
        elif label_spec!=spec:raise ValueError('Effect label schema changes across G0 rows')
        bv=torch.as_tensor(before['values'],dtype=torch.float32);av=torch.as_tensor(after['values'],dtype=torch.float32)
        valid=torch.as_tensor(before['valid'],dtype=torch.bool)&torch.as_tensor(after['valid'],dtype=torch.bool)
        delta=torch.where(valid,av-bv,torch.zeros_like(av))
        if not torch.isfinite(delta).all():raise ValueError('Nonfinite valid verified physical effect')
        preserve=kind=='identity_preservation'
        target=(record['raw'].clone() if preserve else
                torch.from_numpy(encode_ego(np.asarray(row['target_physical'],dtype=np.float32))))
        examples.append({'scene_id':row['scene_id'],'source_log':row['source_log'],'target_raw':target,
                         'is_repair':not preserve,'is_preserve':preserve,
                         'effect_delta':delta,'effect_valid':valid,'target_kind':kind})
        counts['preserve' if preserve else 'repairs']+=1
    if not counts['repairs']:raise ValueError('BLOCKED_NO_REPAIR: no verified local repair fit scenes')
    if not counts['preserve']:raise ValueError('BLOCKED_NO_KEEP: no valid high-quality replay')
    return examples,{'dataset':artifact(repair_dataset),'counts':counts,'label_spec':label_spec,
                     'repair_log_count':len({r['source_log'] for r in examples if r['is_repair']})}


def balance_examples(examples):
    """Exactly balanced reference pool; never report repeated refs as new data."""
    repairs=[x for x in examples if x['is_repair']]
    keeps=[x for x in examples if x['is_preserve']]
    if not repairs or not keeps:
        raise ValueError('BLOCKED_NO_REPAIR/KEEP: fixed balanced recipe cannot be changed silently')
    n=math.lcm(len(repairs),len(keeps))
    if 2*n>2_000_000:raise ValueError('Balanced reference index exceeds explicit small-study memory budget')
    pool=[repairs[i%len(repairs)] for i in range(n)]+[keeps[i%len(keeps)] for i in range(n)]
    return pool,{'sampling':'uniform replacement over 1:1 balanced reference pool',
        'unique_repair_scenes':len({x['scene_id'] for x in repairs}),
        'unique_keep_scenes':len({x['scene_id'] for x in keeps}),
        'virtual_references':len(pool),'references_per_class':n,
        'reference_ids_sha256':digest([x['scene_id'] for x in pool]),
        'unique_observations_added_by_oversampling':0}


class RawActionEffectTeacher(torch.nn.Module):
    """Training-only bridge to this campaign's independently trained G1 model."""
    def __init__(self,model):
        super().__init__()
        if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
            raise ValueError('Load a frozen eval G1 teacher before building the raw bridge')
        if getattr(getattr(model,'config',None),'mode',None)!='delta_feedback':
            raise ValueError('B1-effect requires the declared paired-delta G1 arm')
        self.model=model
        self.provenance=getattr(model,'provenance',None)
        self.train(False)

    def train(self,mode=True):
        super().train(False)
        return self

    def forward(self,context,base_raw,intervention_raw):
        from starVLA.dataloader.foresight_dataset import decode_ego
        modified=base_raw+intervention_raw
        if any((x[...,2:4].square().sum(-1)<1e-12).any() for x in (base_raw,modified)):
            raise ValueError('Undefined sin/cos heading in velocity-derived teacher proxy')
        prediction=self.model(context,decode_ego(base_raw),decode_ego(modified))
        return prediction['mean_delta']


def _verify_worker(args):
    from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash
    from starVLA.dataloader.foresight_dataset import decode_ego
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda'):torch.cuda.set_device(args.device)
    model,provenance=load_original_action_head(args.head_receipt,args.base_binding,device=args.device)
    data=torch.load(args.verify_input,map_location='cpu',weights_only=True)
    outputs=[]
    with torch.no_grad():
        for row in data:
            cond=row['condition'].to(args.device)[None];noise=row['noise'].to(args.device)[None]
            raw=model.predict_action(cond,initial_noise=noise);physical=decode_ego(raw.float())
            outputs.append({'scene_id':row['scene_id'],'raw':raw.cpu(),'physical':physical.cpu(),
                            'raw_hash':tensor_hash(raw),'physical_hash':tensor_hash(physical)})
    forbidden=[name for name in sys.modules if name.startswith(('navsim.','transformers.models.qwen3_vl.modeling_qwen3_vl'))]
    if forbidden:raise RuntimeError('Head-only process unexpectedly imported a simulator/Qwen implementation: '+repr(forbidden))
    torch.save(outputs,args.worker_output)
    return {'status':'PASS','scenes':len(outputs),'strict_load':True,'provenance':provenance}


def verify_head_roundtrip(reader,*,base_binding,output,device,max_samples=1):
    """Two sequential independent processes, actual bound context/noise, no VLM.

    This parent only verifies receipts and prepares small legal tensors. Each
    child strictly rebuilds the canonical original action head independently.
    The Lead admits GPU/runtime budget before invoking this command.
    """
    from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash
    if not 1<=max_samples<=8:raise ValueError('Head qualification is bounded to1..8 actual scenes')
    output=Path(output)
    if output.exists():raise FileExistsError('Keep prior head qualification; use a new output directory')
    output.mkdir(parents=True)
    head_receipt=str(verify_artifact(reader.manifest['head_receipt']))
    records=[reader[s] for s in list(reader.rows)[:max_samples]]
    rows=[{'scene_id':r['scene_id'],'condition':r['condition'],'noise':r['noise']} for r in records]
    inp=output/'CURRENT_CONDITION_NOISE.pt';torch.save(rows,inp)
    started=time.perf_counter();processes=[];results=[]
    for attempt in range(2):
        target=output/f'process_{attempt}.pt'
        cmd=[sys.executable,'-m','tools.action_effect.train_effect_flow','--verify-worker',
            '--head-receipt',head_receipt,'--base-binding',str(base_binding),'--verify-input',str(inp),
            '--worker-output',str(target),'--device',device]
        begin=time.perf_counter()
        with (output/f'process_{attempt}.stdout.log').open('w') as stdout,(output/f'process_{attempt}.stderr.log').open('w') as stderr:
            completed=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=600,
                env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2',
                     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'})
        processes.append({'index':attempt,'returncode':completed.returncode,
                          'elapsed_seconds':time.perf_counter()-begin,'command':cmd})
        if completed.returncode:raise RuntimeError('Head qualification child failed; logs preserved')
        results.append(torch.load(target,map_location='cpu',weights_only=True))
    if any(len(result)!=len(records) for result in results):raise ValueError('Head process omitted scenes')
    comparisons=[]
    for original,a,b in zip(records,*results):
        expected_raw=original['raw'][None];expected_physical=original['physical'][None]
        if a['scene_id']!=original['scene_id'] or b['scene_id']!=original['scene_id']:
            raise ValueError('Head process reordered scene identities')
        exact=(torch.equal(a['raw'],b['raw']) and torch.equal(a['physical'],b['physical']) and
               torch.equal(a['raw'],expected_raw) and torch.equal(a['physical'],expected_physical))
        comparisons.append({'scene_id':original['scene_id'],'exact':exact,'raw_hash':a['raw_hash'],
                            'physical_hash':a['physical_hash'],'cache_raw_hash':original['raw_hash'],
                            'cache_physical_hash':original['physical_hash']})
        if not exact:raise ValueError('Actual canonical head/cache roundtrip mismatch; no tolerance relaxation')
    receipt={'schema':'action_effect_real_head_roundtrip_v1','status':'PASS','scenes':len(records),
        'context_identity':reader.manifest['identity'],'base_binding':artifact(base_binding),
        'head_receipt':artifact(head_receipt),'comparisons':comparisons,'processes':processes,
        'elapsed_seconds':time.perf_counter()-started,'Qwen_calls':0,'simulator_calls':0,'optimizer_updates':0}
    write_json(output/'COMPLETE.json',receipt)
    return receipt


def train(cfg,*,context_manifest,repair_dataset,arm,device,seed,max_updates,
          output,stop_after=None,resume=False,effect_checkpoint=None,
          velocity_weight=1.,preserve_weight=1.,effect_weight=.2,direction_weight=.2):
    """One separately initialized arm using only this campaign's fixed data."""
    from .context_data import ContextReader,batch_context
    from .training import run_training
    reader=ContextReader(context_manifest,Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json')
    examples,data_identity=load_flow_examples(repair_dataset,reader)
    examples,sampling_identity=balance_examples(examples)
    t=cfg['training']
    recipe=FlowLossRecipe(microbatch_scenes=t['microbatch_scenes'],velocity=velocity_weight,
        preserve=preserve_weight,effect=effect_weight,direction=direction_weight).validate()
    if arm not in EffectInternalizedFlow.ARMS:raise ValueError('Unknown IEG arm')
    if (arm=='B1-effect')!=(effect_checkpoint is not None):
        raise ValueError('Only B1-effect requires/permits a verified G1 checkpoint')
    if seed not in t['training_seeds']+t['finalist_seeds']:raise ValueError('Unregistered training seed')
    if not 1<=max_updates<=t['max_updates']:raise ValueError('Explicit bounded training updates required')
    head,head_identity=load_original_action_head(verify_artifact(reader.manifest['head_receipt']),
                                               cfg['base_binding'],device=device)
    model=build_flow_arm(head,arm=arm,init_seed=seed,hidden=t['hidden'],layers=t['layers'],heads=t['heads']).to(device)
    teacher=None;scales=None;teacher_identity=None
    if arm=='B1-effect':
        from .train_effect_model import load_effect_teacher
        g1=load_effect_teacher(effect_checkpoint,
            expected_base_binding_hash=reader.binding['base_checkpoint_binding_hash']).to(device)
        if (list(g1.label_spec.names)!=data_identity['label_spec']['names'] or
                list(g1.label_spec.units)!=data_identity['label_spec']['units']):
            raise ValueError('G1 physical delta schema differs from G0 repair labels')
        teacher=RawActionEffectTeacher(g1);scales=g1.label_spec.scales;teacher_identity=g1.provenance
    ng=torch.Generator(device=device).manual_seed(seed+301)
    tg=torch.Generator(device=device).manual_seed(seed+302)
    step_fn=FlowStep(examples,reader.__getitem__,recipe=recipe,device=device,
        noise_generator=ng,time_generator=tg,context_batcher=batch_context,teacher=teacher,effect_scales=scales)
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'arm':arm,'base_binding_hash':reader.binding['base_checkpoint_binding_hash'],
        'context_artifact':artifact(context_manifest),'repair_data':data_identity,
        'head':head_identity,'adapter_config':asdict(model.adapter.config),
        'sampling':sampling_identity,'loss_recipe':asdict(recipe),'effect_teacher':teacher_identity,
        'noise_seed':seed+301,'time_seed':seed+302,'flow_semantics':'original_repeat8_Beta1.5_1_s.999',
        'validation':'offline fixed endpoint Euler10 and NAVSIM; no simulator inside optimizer loop',
        'deployment':'single Euler10 generation, no teacher/refiner/simulator'}
    Path(output).mkdir(parents=True,exist_ok=True)
    write_json(Path(output)/'RESOLVED_CONFIG.json',cfg)
    return run_training(model,output=Path(output),identity=identity,dataset_size=len(examples),
        step_fn=step_fn,cfg=cfg,device=device,seed=seed,max_updates=max_updates,
        stop_after=stop_after,resume=resume,stateful={'flow_noise':ng,'flow_time':tg})


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config');p.add_argument('--context-manifest');p.add_argument('--repair-dataset')
    p.add_argument('--arm',choices=EffectInternalizedFlow.ARMS);p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-updates',type=int)
    p.add_argument('--stop-after',type=int);p.add_argument('--output');p.add_argument('--effect-checkpoint')
    p.add_argument('--resume',action='store_true',help='Restore the latest complete checkpoint in this exact arm/output; never restart.')
    p.add_argument('--dry-run',action='store_true',help='Validate declared command/config; does not load weights or run training.')
    for name,default in [('velocity',1.),('preserve',1.),('effect',.2),('direction',.2)]:
        p.add_argument('--'+name+'-weight',type=float,default=default)
    p.add_argument('--verify-head-only',action='store_true',help='Two independent strict-load original-head processes; no VLM/simulator/optimizer.')
    p.add_argument('--max-samples',type=int,default=1)
    p.add_argument('--verify-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--head-receipt',help=argparse.SUPPRESS);p.add_argument('--base-binding',help=argparse.SUPPRESS)
    p.add_argument('--verify-input',help=argparse.SUPPRESS);p.add_argument('--worker-output',help=argparse.SUPPRESS)
    return p


def main():
    p=parser();a=p.parse_args()
    if a.verify_worker:
        if any(getattr(a,k) is None for k in ('head_receipt','base_binding','verify_input','worker_output')):
            p.error('Internal head verification requires complete explicit inputs')
        print(json.dumps(_verify_worker(a)));return
    if a.config is None or a.context_manifest is None or a.output is None:
        p.error('--config, --context-manifest and --output are required')
    cfg=load_config(a.config)
    if a.verify_head_only:
        if any((a.arm,a.repair_dataset,a.effect_checkpoint,a.resume,a.stop_after,a.max_updates)):
            p.error('Head verification cannot be mixed with training/resume/teacher inputs')
        if not 1<=a.max_samples<=8:p.error('Head verification allows only1..8 scenes')
        if a.dry_run:
            print(json.dumps({'status':'NOT_RUN','operation':'head_roundtrip','device':a.device,'max_samples':a.max_samples}));return
        from .context_data import ContextReader
        reader=ContextReader(a.context_manifest,Path(cfg['artifact_root'])/'CAMPAIGN_BINDING.json')
        print(json.dumps(verify_head_roundtrip(reader,base_binding=cfg['base_binding'],output=a.output,
            device=a.device,max_samples=a.max_samples)));return
    if a.arm is None or a.repair_dataset is None or a.max_updates is None:
        p.error('Training requires --arm, --repair-dataset and --max-updates')
    if (a.arm=='B1-effect')!=(a.effect_checkpoint is not None):p.error('Only B1-effect requires/permits --effect-checkpoint')
    if not 1<=a.max_updates<=cfg['training']['max_updates']:p.error('Update budget exceeds locked configuration')
    if a.stop_after is not None and not 1<=a.stop_after<=a.max_updates:p.error('Invalid bounded --stop-after')
    FlowLossRecipe(velocity=a.velocity_weight,preserve=a.preserve_weight,
                   effect=a.effect_weight,direction=a.direction_weight).validate()
    if a.dry_run:
        print(json.dumps({'status':'NOT_RUN','arm':a.arm,'max_updates':a.max_updates,
            'output':a.output,'resume_requested':a.resume,'fixed_repair_keep_ratio':'1:1'}));return
    print(json.dumps(train(cfg,context_manifest=a.context_manifest,repair_dataset=a.repair_dataset,
        arm=a.arm,device=a.device,seed=a.seed,max_updates=a.max_updates,output=a.output,
        stop_after=a.stop_after,resume=a.resume,effect_checkpoint=a.effect_checkpoint,
        velocity_weight=a.velocity_weight,preserve_weight=a.preserve_weight,
        effect_weight=a.effect_weight,direction_weight=a.direction_weight)))


if __name__=='__main__':main()
