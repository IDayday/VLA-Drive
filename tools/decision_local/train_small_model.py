"""Bounded single-GPU frozen-representation experiments, no VLA optimizer.

Reuses the frozen-W probe's epoch/offset recipe, existing RNG capture/restore,
and the exact structured-world final_trajectory_loss. No distributed scheduler.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, identity, read, source_sha
from tools.foresight.student_state import capture_rng, restore_rng, learning_rate
from starVLA.dataloader.decision_local import FrozenConsequenceData, inputs_queries_targets, oracle_relations
from starVLA.model.modules.decision_local.consequence_head import ConsequenceQueryHead
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations
from starVLA.model.modules.decision_local.consequence_losses import consequence_loss
from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
from starVLA.model.modules.structured_world.losses import final_trajectory_loss
from starVLA.dataloader.foresight_dataset import decode_ego

KINDS = ('E_NONE','E_CURRENT_GT','E_CV_GT','E_FUTURE_GT','E_MASK_ONLY',
         'P_ABS','P_DIFF','P_QUERY_ONLY','R_CURRENT_PRED','R_FUTURE_ABS','R_FUTURE_DIFF',
         'LOG_NONE','LOG_CV','LOG_FUT','MIN_NONE','MIN_CV','MIN_FUT','R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF',
         'N_NONE','N_CV','N_FUT','R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE')


def build_model(kind, seed, dim=512):
    if kind not in KINDS: raise ValueError('Unsupported experiment')
    # Isolate initialization from scene order and anchor/proposal RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind.startswith(('LOG_','MIN_','R_MIN_','N_')) or kind in ('R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE'):
            from starVLA.model.modules.decision_local.minimal_repair_contract import MinimalRepairRefiner
            return MinimalRepairRefiner(dim=dim)
        return ConsequenceQueryHead(dim=dim, query_only=kind=='P_QUERY_ONLY') if kind.startswith('P_') else ResidualRefiner(dim=dim)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def apply_log_fold(train, heldout, partition, training_fold):
    """Read-only index views; the frozen S0 base itself is NOT out-of-fold."""
    if training_fold not in (0,1) or partition['features']!=train.feature_identity['identity']:
        raise ValueError('Foreign two-fold partition')
    if heldout is None or heldout.feature_identity!=train.feature_identity or heldout.label_identity!=train.label_identity:
        raise ValueError('OOF readout check must split the same original training population')
    log_fold=partition['log_fold'];logs={row['log'] for row in train.index}
    if set(log_fold)!=logs or set(log_fold.values())!={0,1}:
        raise ValueError('Exactly two exhaustive log folds required')
    original=train.index
    train.index=[row for row in original if log_fold[row['log']]==training_fold]
    heldout.index=[row for row in original if log_fold[row['log']]!=training_fold]
    if not train.index or not heldout.index:raise ValueError('Empty fold')
    return {'partition_identity':identity(partition),'training_fold':training_fold,
        'training_scenes':len(train.index),'heldout_scenes':len(heldout.index),
        'training_index_sha256':identity(train.index),'heldout_index_sha256':identity(heldout.index),
        'scope':'Predictor readout only; shared S0 previously trained on all these logs; half-population fit, not pure full-data overfit decomposition'}


def forward(model, kind, batch, predictor=None):
    if batch.get('local_evidence_protocol',False):
        from tools.decision_local.natural_local_training import natural_forward
        return natural_forward(model,kind,batch,predictor)
    if 'plan_in_physical' in batch:
        from tools.decision_local.minimal_repair_training import repair_forward
        return repair_forward(model,kind,batch,predictor)
    inputs, queries, targets = inputs_queries_targets(batch)
    if kind.startswith('P_'):
        raw = model(inputs, queries)
        result = consequence_loss(raw.float(), targets.relations, targets.valid, difference_weight=1. if kind=='P_DIFF' else 0.)
        losses = {'total': result['loss'], 'absolute': result['absolute'], 'difference': result['difference'], **result['channels']}
        return raw, losses, targets
    if kind in ('E_CURRENT_GT','E_CV_GT','E_FUTURE_GT','E_MASK_ONLY'):
        output = model.oracle_probe(inputs, queries, oracle_relations(batch, kind))
    elif kind == 'E_NONE':
        output = model.camera_only(inputs, queries)
    else:
        if predictor is None: raise ValueError('R requires its own normally predicted frozen relations')
        with torch.no_grad(): relations = predicted_relations(predictor(inputs, queries).float())
        if kind == 'R_CURRENT_PRED':
            relations = torch.cat((relations[..., :3], relations[..., 1:3]), -1)
        output = model.camera_only(inputs, queries, relations)
    loss, _ = final_trajectory_loss(output.float(), targets.ego.float(), targets.future_valid)
    return output, {'total':loss, 'trajectory':loss}, targets


def load_predictor(path, expected_kind):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity']['kind'] != expected_kind: raise ValueError('Wrong predictor arm')
    model = build_model(expected_kind, payload['identity']['seed'], payload['identity']['dim'])
    model.load_state_dict(payload['model'], strict=True); model.eval().requires_grad_(False)
    return model.cuda(), payload


@torch.no_grad()
def evaluate(model, kind, data, microbatch, predictor=None):
    model.eval(); metrics = []; n = len(data.index)
    for start in range(0, n, microbatch):
        batch = data.batch(list(range(start,min(n,start+microbatch))), 'cuda', evaluation=True)
        output, losses, target = forward(model, kind, batch, predictor)
        row = {'scenes':len(batch['ego']), **{k:float(v) for k,v in losses.items()}}
        if not kind.startswith('P_'):
            target_ego=target.original_ego_GT if 'plan_in_physical' in batch else target.ego
            p, gt = decode_ego(output.float()), decode_ego(target_ego.float())
            distance = (p[...,:2]-gt[...,:2]).norm(dim=-1)
            row.update(ADE=float(distance.mean()), FDE=float(distance[:,-1].mean()),
                residual_m=float((p[...,:2]-batch.get('plan_in_physical',batch.get('q0'))[...,:2]).norm(dim=-1).mean()))
        metrics.append(row)
    model.train()
    return {k:sum(r[k]*r['scenes'] for r in metrics)/n for k in metrics[0] if k!='scenes'} | {'scenes':n}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('features','labels','output'): p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--dev-features',type=Path); p.add_argument('--dev-labels',type=Path)
    p.add_argument('--kind',choices=KINDS,required=True); p.add_argument('--predictor',type=Path)
    p.add_argument('--scope',choices=('smoke','diagnostic','formal','fold_control','repair_formal','natural_formal'),required=True)
    p.add_argument('--protocol',choices=('decision_local_v1','minimal_repair_v1','natural_local_v1'),default='decision_local_v1')
    p.add_argument('--fold-partition',type=Path);p.add_argument('--training-fold',type=int)
    p.add_argument('--target-acceptance',type=Path,help='Immutable target-quality and finite-budget receipt for formal repair runs')
    p.add_argument('--repair-class-balance',type=Path,help='Explicit separate MIN diagnostic: fixed training-exposure class weights')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch',type=int,default=128); p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--dim',type=int,default=512)
    p.add_argument('--max-updates',type=int,default=0); p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--resume',action='store_true'); p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--data-reader',choices=('slice','mmap_tensors'),default='slice')
    a = p.parse_args()
    if min(a.epochs,a.batch,a.microbatch,a.max_seconds)<1: raise ValueError('Finite positive budget required')
    if a.scope=='formal' and (a.epochs not in (24,32) or a.batch!=128 or a.dim!=512):
        raise ValueError('Formal common recipe cannot silently change')
    resources = check_host(); start = time.monotonic()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed+1000); torch.cuda.manual_seed_all(a.seed+1000); np.random.seed(a.seed+1000); random.seed(a.seed+1000)
    natural=a.protocol=='natural_local_v1'
    repair=a.protocol in ('minimal_repair_v1','natural_local_v1')
    if a.repair_class_balance is not None and not repair:
        raise ValueError('Class balance cannot alter the legacy protocol')
    if natural:
        from starVLA.dataloader.natural_local_dataset import NaturalLocalData
        from tools.decision_local.natural_local_training import NATURAL_KINDS
        from tools.decision_local.minimal_repair_training import label_semantics
        if a.kind not in NATURAL_KINDS or a.scope not in ('smoke','diagnostic','natural_formal'):
            raise ValueError('Use the versioned natural-error experiment and scope')
        if a.fold_partition is not None or a.training_fold is not None or a.repair_class_balance is not None:
            raise ValueError('Old folds/repair reweighting cannot enter the natural-error study')
        train=NaturalLocalData(a.features,a.labels,reader=a.data_reader)
        dev=NaturalLocalData(a.dev_features,a.dev_labels,reader=a.data_reader) if a.dev_features else None
        codec_path=train.label_identity.get('action_codec')
        if codec_path:
            from starVLA.dataloader.a_only_codec import install_legacy_codec
            install_legacy_codec(read(codec_path))
    elif repair:
        from starVLA.dataloader.recovery_plan_dataset import RecoveryPlanData
        from tools.decision_local.minimal_repair_training import REPAIR_KINDS,label_semantics
        if a.kind not in REPAIR_KINDS+('P_ABS','P_DIFF'):raise ValueError('Wrong kind for repair protocol')
        if a.scope not in ('smoke','diagnostic','repair_formal'):raise ValueError('Use explicit new scope, preserve legacy formal checks')
        train=RecoveryPlanData(a.features,a.labels,reader=a.data_reader,fold_partition=a.fold_partition,fold=a.training_fold)
        dev=RecoveryPlanData(a.dev_features,a.dev_labels,reader=a.data_reader) if a.dev_features else None
    else:
        if a.kind.startswith('N_') or a.kind in ('R_CURRENT','R_ABS_FUTURE','R_DIFF_FUTURE'):
            raise ValueError('Natural-error kinds require their separate local-supervision protocol')
        if a.kind.startswith(('LOG_','MIN_','R_MIN_')) or a.scope=='repair_formal':raise ValueError('Explicit repair protocol required')
        train = FrozenConsequenceData(a.features,a.labels,reader=a.data_reader)
        dev = FrozenConsequenceData(a.dev_features,a.dev_labels,reader=a.data_reader) if a.dev_features else None
    if train.feature_identity['split']!='train':raise ValueError('Training must use the registered training split')
    if a.scope=='diagnostic' and len(train.index)!=4096:raise ValueError('Use the fixed 4096 training diagnostic population')
    if a.scope=='formal' and (len(train.index)!=101592 or dev is None or len(dev.index)!=1696):
        raise ValueError('Full common population required')
    fold_identity=train.fold_identity if repair else None
    acceptance=None
    if a.scope=='natural_formal':
        if not natural:raise ValueError('Explicit local-supervision protocol required')
        if (a.epochs!=24 or a.batch!=128 or a.microbatch!=16 or a.dim!=512 or a.lr!=1e-4 or
            a.seed not in (42,43) or dev is None or len(dev.index)!=1696):
            raise ValueError('Same registered small-head recipe and full independent development population required')
        if a.target_acceptance is None:raise ValueError('Finite population and training receipt required')
        acceptance=read(a.target_acceptance)
        stage='stage_P_R_training_allowed' if a.kind.startswith(('P_','R_')) else 'stage_N_training_allowed'
        if acceptance.get(stage) is not True or a.kind not in acceptance['kinds'] or a.seed not in acceptance['seeds']:
            raise ValueError('Conditional formal stage has not been registered')
        role='B_predictor' if a.kind.startswith('P_') else 'B_refiner'
        if (train.label_identity['role']!=role or train.label_identity['scope']=='engineering_historical_anchor_only' or
            acceptance['anchor_identity']!=train.feature_identity['anchor_identity'] or
            acceptance['source_audit_identity']!=train.label_identity['source_audit_identity']):
            raise ValueError('Wrong natural correction/base population')
        if train.label_identity.get('local_fact_identity_layout')!='per_query_channel_int32_lookup_v1':
            raise ValueError('Formal local evidence must preserve per-fact actor/map and time identity')
        if acceptance.get('label_semantics')!=label_semantics(train.label_identity):
            raise ValueError('Formal population receipt covers another local-label version')
        if not a.kind.startswith('P_'):
            review=read(acceptance['complete_repair_execution_review'])
            if (digest(acceptance['complete_repair_execution_review'])!=acceptance['complete_repair_execution_review_sha256'] or
                review.get('labels_identity')!=train.label_identity['identity'] or
                review.get('all_admitted_repairs_independently_checked') is not True):
                raise ValueError('Full REPAIR targets need their independent execution review')
        if label_semantics(train.label_identity)!=label_semantics(dev.label_identity):raise ValueError('Train/dev local-label semantics differ')
    if a.scope=='repair_formal':
        if any(train.label_identity[k].get('velocity_boundary')!='measured_vector_v2' for k in ('recovery','constraints','query_config')):
            raise ValueError('Corrected formal repair must preserve the complete legal current velocity vector')
        if a.epochs!=24 or a.batch!=128 or a.microbatch!=16 or a.dim!=512 or a.lr!=1e-4 or a.seed not in (42,43):
            raise ValueError('Registered common repair recipe required')
        if a.target_acceptance is None:raise ValueError('Formal repair requires actual target-quality evidence')
        acceptance=read(a.target_acceptance)
        stage='stage_b_training_allowed' if a.kind.startswith(('P_','R_MIN_')) else 'stage_a_training_allowed'
        if acceptance.get(stage) is not True or a.kind not in acceptance['kinds'] or a.seed not in acceptance['seeds']:
            raise ValueError('This arm/seed is outside the registered accepted stage')
        if acceptance['label_semantics']!=label_semantics(train.label_identity):raise ValueError('Target quality receipt covers another label version')
        p_fold=acceptance['predictor_training_fold']
        if p_fold not in (0,1):raise ValueError('Invalid registered predictor fold')
        expected_fold=p_fold if a.kind.startswith('P_') else 1-p_fold
        if a.training_fold!=expected_fold or fold_identity is None or dev is None or len(dev.index)!=1696:
            raise ValueError('Use the registered complete P/R folds and entire independent development split')
        if label_semantics(train.label_identity)!=label_semantics(dev.label_identity):raise ValueError('Train/dev scientific label definition differs')
    if a.scope=='fold_control':
        if a.kind not in ('P_ABS','P_DIFF') or a.epochs!=24 or a.batch!=128 or a.microbatch!=16 or a.dim!=512 or a.lr!=1e-4 or a.seed!=42:
            raise ValueError('Only the matched registered predictor OOF check is supported')
        if len(train.index)!=101592 or a.fold_partition is None:raise ValueError('Split the complete original training population')
        fold_identity=apply_log_fold(train,dev,read(a.fold_partition),a.training_fold)
    elif not repair and (a.fold_partition is not None or a.training_fold is not None):
        raise ValueError('Fold inputs cannot alter the primary full-population arms')
    if dev and {r['log'] for r in train.index}&{r['log'] for r in dev.index}: raise ValueError('Train/dev log leakage')
    balance_receipt=class_weights=None
    if a.repair_class_balance is not None:
        from tools.decision_local.minimal_repair_training import load_repair_class_balance
        balance_receipt,class_weights=load_repair_class_balance(a.repair_class_balance,a.kind,train)
    if a.scope=='repair_formal':
        expected_balance=acceptance.get('repair_class_balance_sha256')
        actual_balance=digest(a.repair_class_balance) if a.repair_class_balance is not None else None
        if actual_balance!=expected_balance:
            raise ValueError('Formal acceptance must explicitly cover this loss weighting')
    model = build_model(a.kind,a.seed,a.dim); initial_hash = tensor_hash(model.state_dict()); model=model.cuda().train()
    if class_weights is not None:
        model.repair_class_weights=class_weights  # Loss-only Python metadata; no state_dict/forward change.
    predictor=payload=None
    if a.kind.startswith('R_'):
        if a.predictor is None: raise ValueError('Frozen corresponding predictor required')
        predictor,payload=load_predictor(a.predictor,'P_DIFF' if a.kind in ('R_FUTURE_DIFF','R_MIN_DIFF','R_DIFF_FUTURE') else 'P_ABS')
        if natural:
            pi=payload['identity']
            if pi.get('protocol')!='natural_local_v1' or pi.get('repair_label_semantics')!=label_semantics(train.label_identity):
                raise ValueError('Predictor local supervision contract changed')
            if pi['anchor_identity']!=train.feature_identity['anchor_identity']:raise ValueError('Wrong shared base')
            if {r.rsplit('_',2)[0] for r in pi['training_log_names']}&{r['log'].rsplit('_',2)[0] for r in train.index}:
                raise ValueError('P and R share a recording')
            if a.scope=='natural_formal' and (pi['repair_fold']['role']!='B_predictor' or payload['epoch']!=24 or payload['offset']!=0):
                raise ValueError('R must use the completed held-out P endpoint')
        elif repair:
            pi=payload['identity']
            if pi.get('protocol')!='minimal_repair_v1' or pi.get('repair_label_semantics')!=label_semantics(train.label_identity):raise ValueError('Predictor target/query contract mismatch')
            if a.scope=='repair_formal' and (pi['repair_fold']['fold']!=acceptance['predictor_training_fold'] or pi['repair_fold']['partition_sha256']!=fold_identity['partition_sha256'] or payload['epoch']!=24 or payload['offset']!=0):raise ValueError('R requires completed opposite-fold P; full-population P forbidden')
            if set(pi.get('training_log_names',[]))&{row['log'] for row in train.index}:raise ValueError('P saw R training logs')
            if pi['anchor_identity']!=train.feature_identity['anchor_identity']:raise ValueError('Wrong predictor anchor')
        elif payload['identity']['features']!=train.feature_identity['identity'] or payload['identity']['labels']!=train.label_identity['identity']:
            raise ValueError('Predictor trained on foreign anchor or labels')
        if a.scope=='formal' and (payload['epoch']!=24 or payload['offset']!=0): raise ValueError('Use common completed predictor endpoint')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001)
    generators={'shuffle':torch.Generator().manual_seed(a.seed+13000)}
    updates_per_epoch=math.ceil(len(train.index)/a.batch); horizon=updates_per_epoch*a.epochs
    # Explicit scheduler state included; same cosine horizon for every arm.
    warmup=min(updates_per_epoch,max(0,horizon-1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: learning_rate(min(step,horizon-1),a.lr,a.lr*.1,warmup,horizon)/a.lr)
    source_files=[*sorted((ROOT/'starVLA/model/modules/decision_local').glob('*.py')),
                  ROOT/'starVLA/dataloader/decision_local.py',Path(__file__)]
    run_identity={'schema':'decision_local_small_head_v1','source_sha':source_sha(ROOT),
        'source_files':{str(f.relative_to(ROOT)):digest(f) for f in source_files},'kind':a.kind,'seed':a.seed,
        'scope':a.scope,'features':train.feature_identity['identity'],'labels':train.label_identity['identity'],
        'data_reader':a.data_reader,
        'anchor_identity':train.feature_identity['anchor_identity'],
        'feature_storage_dtype':train.feature_identity['storage_dtype'],
        'query_config':train.label_identity['query_config'],
        'dev_features':dev.feature_identity['identity'] if dev else None,'dev_labels':dev.label_identity['identity'] if dev else None,
        'epochs':a.epochs,'batch':a.batch,'microbatch':a.microbatch,'lr':a.lr,'dim':a.dim,'weight_decay':.001,
        'scheduler':'one epoch linear warmup then cosine to0.1LR','clip':1.,'compute_precision':'FP32','TF32':False,
        'initial_model_sha256':initial_hash,'predictor_sha256':digest(a.predictor) if a.predictor else None,
        'trainable_parameters':sum(x.numel() for x in model.parameters()),
        'inactive_memory_projection_parameters':sum(x.numel() for x in model.memory.parameters()) if a.kind=='P_QUERY_ONLY' else 0,
        'frozen_base_optimizer_updates':0}
    if fold_identity is not None:run_identity['predictor_OOF_control']=fold_identity
    if repair:
        extra=[ROOT/'starVLA/dataloader/recovery_plan_dataset.py',ROOT/'tools/decision_local/minimal_repair_training.py']
        run_identity['source_files'].update({str(f.relative_to(ROOT)):digest(f) for f in extra})
        run_identity.update(protocol=a.protocol,schema='minimal_repair_small_head_v1',repair_fold=fold_identity,
            repair_label_semantics=label_semantics(train.label_identity),repair_targets=train.label_identity['identity'],
            recovery_config=train.label_identity['recovery'],constraint_config=train.label_identity['constraints'],
            training_index_sha256=identity(train.index),training_log_names=sorted({row['log'] for row in train.index}),
            target_eligibility='LOG and MIN both require valid GT and validated KEEP/REPAIR; all scenes counted, no-target batches skip optimizer',
            plan_exposure='one plan per original scene/epoch; 50%native25%lateral25%time; common deterministic plan/anchor schedule')
        if acceptance is not None:run_identity['target_acceptance_sha256']=digest(a.target_acceptance)
        if balance_receipt is not None:
            run_identity['repair_class_balance']={'receipt':balance_receipt,'sha256':digest(a.repair_class_balance),
                'reduction':'fixed class multipliers; original optimizer-batch eligible-scene denominator'}
        if natural:
            extra=[ROOT/'starVLA/dataloader/natural_local_dataset.py',ROOT/'tools/decision_local/natural_local_training.py']
            run_identity['source_files'].update({str(f.relative_to(ROOT)):digest(f) for f in extra})
            run_identity.update(schema='natural_local_small_head_v1',plan_exposure=train.label_identity['selection'],
                target_eligibility='P: each local channel/time independently; N/R: strict complete KEEP/REPAIR only; unknown never KEEP',
                action_codec_identity=train.label_identity.get('action_codec_identity'),
                natural_anchor_record=train.label_identity['anchor_record'],
                source_audit_identity=train.label_identity['source_audit_identity'])
    a.output.mkdir(parents=True,exist_ok=True); saved_path=a.output/'latest.pt'
    completed=epoch=offset=exposure=0; order=torch.randperm(len(train.index),generator=generators['shuffle'])
    previous_gpu_hours=0.;effective_updates=0;exposure_counts={k:0 for k in ('target_eligible','KEEP','REPAIR','native','lateral','time')}
    if a.resume:
        s=torch.load(saved_path,map_location='cpu',weights_only=False)
        if s['identity']!=run_identity: raise ValueError('Exact resume identity mismatch')
        model.load_state_dict(s['model'],strict=True); optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        restore_rng(s['rng'],generators)
        completed,epoch,offset,exposure,order=[s[k] for k in ('completed','epoch','offset','exposure','order')]
        previous_gpu_hours=s['GPU_hours']
        if repair:effective_updates=s['effective_optimizer_updates'];exposure_counts=s['exposure_counts']
    elif (a.output/'identity.json').exists(): raise FileExistsError('Use explicit --resume')
    else: atomic_json(a.output/'identity.json',run_identity)
    def checkpoint(name='latest.pt'):
        torch.cuda.synchronize()
        s={'identity':run_identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'rng':capture_rng(generators),'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'order':order,
           'GPU_hours':previous_gpu_hours+(time.monotonic()-start)/3600.}
        if repair:s.update(effective_optimizer_updates=effective_updates,exposure_counts=exposure_counts)
        tmp=a.output/(name+'.tmp'); torch.save(s,tmp); tmp.replace(a.output/name)
        atomic_json(a.output/'STATUS.json',{'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,
            'GPU_hours':s['GPU_hours'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'resource_check':resources})
    if completed==0: checkpoint()
    # Previous pause evidence remains; this attempt has its own invocation log.
    atomic_json(a.output/('attempt_%07d.json'%completed),{'resume':a.resume,'starting_update':completed,
        'max_seconds':a.max_seconds,'max_updates':a.max_updates,'resource_check':resources})
    timings=[]; last=time.monotonic()
    while epoch<a.epochs:
        if (a.output/'STOP_REQUESTED').exists() or time.monotonic()-start>=a.max_seconds or (a.max_updates and completed>=a.max_updates):
            checkpoint(); atomic_json(a.output/'PAUSED.json',{'reason':'finite session/step budget or explicit STOP','completed':completed}); return
        ids=order[offset:offset+a.batch].tolist(); optimizer.zero_grad(set_to_none=True); losses_sum={}
        eligible=len(ids)
        if repair:
            selected_plans=train.selection(ids,epoch)
            eligible=sum(row[3] for row in selected_plans)
            exposure_counts['target_eligible']+=eligible
            for _,_,plan_index,ok,status in selected_plans:
                exposure_counts['native' if plan_index==0 else 'lateral' if plan_index<3 else 'time']+=1
                if ok:exposure_counts['KEEP' if status==0 else 'REPAIR']+=1
        for sub in range(0,len(ids),a.microbatch):
            selected=ids[sub:sub+a.microbatch]; batch=train.batch(selected,'cuda',epoch=epoch)
            output,losses,targets=forward(model,a.kind,batch,predictor)
            weight=len(selected)/(max(eligible,1) if repair and not a.kind.startswith('P_') else len(ids)); (losses['total']*weight).backward()
            for k,v in losses.items(): losses_sum[k]=losses_sum.get(k,0)+v.detach()*weight
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(losses_sum['total']): raise FloatingPointError('Nonfinite loss; optimizer not stepped')
        stepped=not repair or a.kind.startswith('P_') or eligible>0
        if stepped:optimizer.step();effective_updates+=1
        scheduler.step(); completed+=1; offset+=len(ids); exposure+=len(ids)
        # One packed read/synchronization per optimizer boundary, including work
        # already submitted. Timings include input IO and complete backward.
        names=sorted(losses_sum); values=torch.stack([losses_sum[k] for k in names]+[norm]).tolist()
        now=time.monotonic(); timings.append(now-last); last=now
        row={'update':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'seconds':timings[-1],
             'lr_next':optimizer.param_groups[0]['lr'],'grad_norm':values[-1],**dict(zip(names,values[:-1]))}
        if repair:row.update(effective_optimizer_updates=effective_updates,eligible_this_batch=eligible,exposure_counts=dict(exposure_counts))
        with (a.output/'train.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if completed%20==0 or completed==1: print(json.dumps(row),flush=True)
        if offset==len(order):
            epoch+=1; offset=0; order=torch.randperm(len(train.index),generator=generators['shuffle'])
            if epoch in (6,12,18,24,32) or epoch==a.epochs:
                checkpoint('epoch_%02d.pt'%epoch)
                if dev:
                    atomic_json(a.output/('dev_epoch_%02d.json'%epoch),evaluate(model,a.kind,dev,a.microbatch,predictor))
                checkpoint(); last=time.monotonic()
        if completed%a.save_every==0: checkpoint(); last=time.monotonic()
    checkpoint()
    atomic_json(a.output/'COMPLETE.json',{'epoch':epoch,'completed':completed,'exposure':exposure,
        'full_training_population':len(train.index)==101592,'median_update_s':float(np.median(timings[5:] or timings)),
        'p90_update_s':float(np.percentile(timings[5:] or timings,90)), 'GPU_hours':previous_gpu_hours+(time.monotonic()-start)/3600.,
        **({'effective_optimizer_updates':effective_updates,'exposure_counts':exposure_counts,'fold':fold_identity} if repair else {})})


if __name__=='__main__': main()
