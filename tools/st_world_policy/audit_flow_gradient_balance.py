"""Read-only four-rank C1 BC/FPO gradient decomposition at a saved boundary.

No optimizer, new scoring, model save, changed reward or changed BC definition.
A single allocated GPU sequentially reconstructs four original rank batches.
BF16 per-microbatch execution gradients are accumulated in FP32 on CPU: this is
an objective diagnostic, NOT a bit-exact reproduction of joint ZeRO2 backward.
Only --execute consumes models, original train images/GT and saved feedback.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

import torch

from tools.st_world_policy import audit_flow_gradient as replay
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.flow_posttraining import (
    DiTFlowPolicy, FlowRandomStreams, RolloutBatch, CoupledFPOBatch, FeedbackBatch,
    group_centered_advantages, fpo_surrogate, module_hash, tensor_hash)
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash, initialization_seed
from tools.foresight.student_state import capture_rng, restore_rng

SCHEMA = 'st_flow_gradient_balance_v1'
require, read_json, verified_artifact = replay.require, replay.read_json, replay.verified_artifact
ROOT = Path(__file__).resolve().parents[2]
LIMITS = {'max_seconds': 1800, 'max_read_bytes': 192 << 30,
          'max_rss_bytes': 64 << 30, 'max_output_bytes': 8 << 20}


def next_rank_entries(manifest, state, execution):
    """Original strided rank ownership; no padding, shuffle or new subsampling."""
    require(execution['world_size'] == 4 and execution['global_batch'] == 32
            and execution['micro_batch'] == 4, 'Only original world4/global32/micro4 is supported')
    # Sampler itself is world-independent. Reuse its original epoch transition.
    entries, cursor = replay.next_scene_entries(manifest, state, {**execution, 'world_size': 1})
    require(len(entries) == 32, 'This registered diagnostic requires one complete 32-scene batch')
    require(len({x['scene_id'] for x in entries}) == 32, 'Duplicate scene in original optimizer batch')
    return [entries[r::4] for r in range(4)], cursor


def validate_registration(path):
    r = read_json(path)
    expected = {'schema', 'source_sha', 'training_registration', 'checkpoint_tag',
                'checkpoint_receipt', 'rollout_receipts', 'budget', 'output', 'resource', 'diagnostic'}
    require(set(r) == expected and r['schema'] == SCHEMA, 'Unknown/missing gradient audit registration field')
    require(r['diagnostic'] == {'cell': 'C1', 'global_scenes': 32, 'original_world_size': 4,
        'optimizer_steps': 0, 'new_score_calls': 0, 'analysis': 'separate_micro_gradients_fp32_sum'},
        'Diagnostic mathematical/expense scope changed')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    require(r['source_sha'] == actual, 'Audit source commit is not the registered reviewed commit')
    dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--',
        'tools/st_world_policy', 'tools/foresight/student_state.py',
        'starVLA/model/modules/flow_posttraining'], cwd=ROOT).returncode
    require(dirty == 0, 'Uncommitted audit/replay sources are not an admitted runtime')
    require(set(r['budget']) == set(LIMITS) and all(type(r['budget'][k]) is int and
        0 < r['budget'][k] <= cap for k, cap in LIMITS.items()), 'Unbounded/unsupported diagnostic budget')
    require(set(r['resource']) == {'host', 'gpu_uuid', 'cpu_threads'} and
        r['resource']['cpu_threads'] in (1, 2) and r['resource']['host'] and
        r['resource']['gpu_uuid'].startswith('GPU-'), 'Explicit host/one GPU/CPU budget required')
    training_path = verified_artifact(r['training_registration'])
    training = read_json(training_path)
    cfg = read_json(verified_artifact(training['artifacts']['config']))
    run = (Path(cfg['runtime']['output_root']) / training['run_id']).resolve()
    tag = r['checkpoint_tag']
    require(isinstance(tag, str) and tag not in ('', '.', '..') and Path(tag).name == tag, 'Unsafe checkpoint tag')
    cp = verified_artifact(r['checkpoint_receipt'])
    require(cp.resolve() == run / 'checkpoints' / tag / 'COMPLETE.json', 'Checkpoint receipt escaped source lineage')
    receipt = read_json(cp)
    step = receipt['state']['completed']
    require(receipt['world_size'] == 4 and receipt['state']['policy_version'] == step
        and receipt['state']['pending_feedback'] is None and receipt['state']['pending_old_policy'] is None,
        'Checkpoint is not a complete four-rank policy boundary')
    require(len(r['rollout_receipts']) == 4, 'Four rank receipts required')
    for rank, item in enumerate(r['rollout_receipts']):
        p = verified_artifact(item)
        require(p.resolve() == run / 'rollouts' / f'boundary_{step:06d}' / f'rank_{rank}_COMPLETE.json',
                'Rollout rank/boundary mismatch')
        old = read_json(p)['identity']['old_policy']
        require(old['optimizer_step'] == old['version'] == step, 'Checkpoint is not BEFORE the scored update')
    require(cfg['cell'] == 'C1' and cfg['loss'] == {'bc_weight': 1., 'policy_weight': 1.}, 'Original C1 weights required')
    out = Path(r['output'])
    require(out.is_absolute() and out != run and run not in out.parents, 'Audit must use a separate absolute output namespace')
    return r, training, cfg, receipt, run


def load_rank_payload(path, rank, *, numeric_hash, step, binding_hash, entries, fpo_config):
    """Rank-aware extension of the existing strict single-rank replay reader."""
    path = Path(path)
    receipt = read_json(path)
    require(set(receipt) == {'identity', 'payload'}, 'Feedback receipt fields changed')
    payload_path = verified_artifact(receipt['payload'])
    require(payload_path.resolve() == path.with_name(f'rank_{rank}.pt').resolve(), 'Feedback escaped rank boundary')
    p = torch.load(payload_path, map_location='cpu', weights_only=False)
    require(set(p) == {'schema', 'identity', 'rollout', 'comparison', 'feedback'} and
        p['schema'] == 'st_flow_update_feedback_v1' and p['identity'] == receipt['identity'], 'Feedback payload schema mismatch')
    rollout = RolloutBatch.from_state_dict(p['rollout'])
    comparison = CoupledFPOBatch.from_state_dict(p['comparison'], rollout)
    feedback = FeedbackBatch.from_state_dict(p['feedback'], rollout)
    require(rollout.policy.optimizer_step == rollout.policy.version == step and
        rollout.policy.base_binding_hash == binding_hash, 'Wrong policy/checkpoint boundary')
    require(rollout.scene_ids == tuple(e['scene_id'] for e in entries) and
        rollout.observation_hashes == tuple(e['observation_hash'] for e in entries), 'Rank sample/observation order differs')
    require(comparison.config.to_dict() == fpo_config and comparison.config.mc_samples == 50, 'Original MC50 contract changed')
    expected = {'rollout_id': rollout.rollout_id, 'comparison_id': comparison.comparison_id,
        'numeric_execution_hash': numeric_hash, 'old_policy': vars(rollout.policy),
        'feedback': {'protocol': feedback.score_protocol_hash, 'contexts': list(feedback.context_hashes),
            'reward_kind': feedback.reward_kind, 'rewards': feedback.rewards.tolist(), 'statuses': list(feedback.statuses)}}
    require(receipt['identity'] == expected and feedback.reward_kind == 'C1-PDMS', 'Saved official C1 identity differs')
    feedback.validate(rollout)
    return rollout, comparison, feedback


def prepare_inputs(registration):
    """Execution-only verification; intentionally reuses the mature native reader."""
    r, training, cfg, receipt, run = validate_registration(registration)
    cfg, binding, native = replay.flow_checkpoint_identity(r['training_registration']['path'], r['checkpoint_tag'])
    require(read_json(verified_artifact(training['artifacts']['environment'])) == replay.environment_record(), 'Runtime differs from original')
    require(read_json(verified_artifact(training['artifacts']['teacher_manifest'])) == replay.NO_TEACHER, 'External teacher forbidden')
    require(read_json(verified_artifact(training['artifacts']['target_contract'])) == replay.target_contract(cfg), 'Target contract differs')
    manifest = read_json(verified_artifact(training['artifacts']['data_manifest']))
    require(manifest['schema'] == 'st_flow_training_manifest_v1' and manifest['split'] == 'train' and
        manifest['identity'] == identity_hash({k:v for k,v in manifest.items() if k != 'identity'}) and
        manifest['base_binding_hash'] == binding['binding_sha256'] and
        manifest['score_protocol_hash'] == cfg['feedback']['score_protocol_hash'] and
        manifest['root'] == str(Path(cfg['data']['root']).resolve()) and
        manifest['scope'] == cfg['execution']['scope'], 'Training manifest contract changed')
    identity = read_json(run / 'identity.json')
    require(identity['data'] == manifest['identity'] and
        manifest['data_identity'] == binding['parent_identity']['data'] and
        manifest['ego_identity'] == binding['parent_identity']['ego'], 'Parent training population/GT identity changed')
    require(file_sha256(verified_artifact(manifest['current_preprocessing'])) ==
        file_sha256(ROOT / 'starVLA/dataloader/foresight_dataset.py'), 'Input/GT converter changed')
    entries, cursor = next_rank_entries(manifest, receipt['state'], cfg['execution'])
    payloads = []
    for rank, rows in enumerate(entries):
        for e in rows:
            for a in (e['current'], e['ego'], *e['images']): verified_artifact(a)
            require(Path(e['ego']['path']).resolve() == Path(cfg['data']['root']).resolve() / 'ego' / (e['scene_id']+'.pt'), 'Foreign GT path')
            require(e['observation_hash'] == identity_hash({'current': e['current'], 'images': e['images'],
                'preprocess': manifest['current_preprocessing']['sha256']}), 'Input identity mismatch')
        payloads.append(load_rank_payload(r['rollout_receipts'][rank]['path'], rank,
            numeric_hash=identity['numeric_execution_hash'], step=native['completed'],
            binding_hash=binding['binding_sha256'], entries=rows, fpo_config=cfg['fpo']))
    require(len({p[0].policy.state_hash for p in payloads}) == 1, 'Rank policies differ')
    return {'registration': r, 'training': training, 'cfg': cfg, 'binding': binding, 'native': native,
        'receipt': receipt, 'manifest': manifest, 'rank_entries': entries, 'cursor': cursor, 'payloads': payloads,
        'frozen_encoder_hash': receipt['state']['frozen_encoder_hash'],
        'replay_sources': replay.replay_source_contract(training['source_sha'])}


class GradientSum:
    """Retain only two FP32 CPU vectors; scalar products use chunked float64."""
    def __init__(self, module):
        self.names = tuple(n for n,p in module.named_parameters() if p.requires_grad)
        self.sums = {k: {} for k in ('bc', 'policy')}
        self.calls = {'bc': 0, 'policy': 0}
        self.execution_dtypes = set()

    def add(self, kind, module):
        require(kind in self.sums, 'Unknown objective')
        params = {n:p for n,p in module.named_parameters() if p.requires_grad}
        require(tuple(params) == self.names, 'Trainable inventory changed')
        for name, p in params.items():
            require(p.grad is not None and torch.isfinite(p.grad).all(), 'Missing/nonfinite gradient: '+name)
            self.execution_dtypes.add(str(p.grad.dtype))
            g = p.grad.detach().to(device='cpu', dtype=torch.float32)
            if name in self.sums[kind]: self.sums[kind][name].add_(g)
            else: self.sums[kind][name] = g.clone()
        self.calls[kind] += 1

    def report(self, policy_weight=1.):
        require(self.calls['bc'] == self.calls['policy'] > 0, 'Incomplete objective decomposition')
        grouped = {k: {'bc_squared_l2': 0., 'policy_squared_l2': 0., 'dot': 0., 'elements': 0}
                   for k in ('all', 'DiT', 'qwen_proj', 'other_action')}
        parameters = {}
        for name in self.names:
            b, p = self.sums['bc'][name], self.sums['policy'][name]
            s = {'bc_squared_l2': 0., 'policy_squared_l2': 0., 'dot': 0., 'elements': b.numel()}
            for x,y in zip(b.flatten().split(1<<20),p.flatten().split(1<<20)):
                x,y=x.double(),y.double()
                s['bc_squared_l2'] += float(x.square().sum())
                s['policy_squared_l2'] += float(y.square().sum())
                s['dot'] += float((x*y).sum())
            parameters[name] = derived_statistics(s, policy_weight)
            group = 'qwen_proj' if name.startswith('action_head.qwen_proj.') else 'DiT' if name.startswith('action_head.model.') else 'other_action'
            for k in ('all', group):
                for field,value in s.items(): grouped[k][field] += value
        return {'groups': {k:derived_statistics(v,policy_weight) for k,v in grouped.items()},
            'parameters': parameters, 'microbatch_counts': self.calls,
            'execution_gradient_dtypes': sorted(self.execution_dtypes), 'analysis_accumulation': 'torch.float32_CPU',
            'dot_reduction': 'chunked_float64', 'policy_weight': policy_weight,
            'exact_ZeRO2_update_claim': False}


def derived_statistics(s, weight):
    b,p,d = s['bc_squared_l2'],s['policy_squared_l2'],s['dot']
    bn,pn=math.sqrt(b),math.sqrt(p)
    joint=max(0.,b+weight*weight*p+2*weight*d)
    return {**s, 'bc_l2':bn, 'policy_l2':pn, 'weighted_policy_l2':abs(weight)*pn,
        'cosine': d/(bn*pn) if bn*pn else None,
        'weighted_policy_over_bc':abs(weight)*pn/bn if bn else None,
        'combined_gradient_l2': math.sqrt(joint), 'weighted_cross_term':2*weight*d,
        'first_order_BC_change_under_combined_descent_per_unit_lr':-(b+weight*d),
        'first_order_policy_change_under_combined_descent_per_unit_lr':-(d+weight*p)}


def audit_rank(policy, condition, targets, rollout, comparison, feedback, streams, accumulator,
               *, global_scenes, micro_batch, mc_chunk, checkpoint_velocity, autocast_factory=nullcontext):
    """Original BC then FPO, separated backward for gradient diagnosis only.

    world_size=1 and true global population yields each microbatch's global
    contribution. Sum across original ranks; do not average local means again.
    """
    require(tuple(tensor_hash(v) for v in condition) == rollout.condition_hashes and not condition.requires_grad,
            'Frozen original condition differs')
    require(module_hash(policy) == rollout.policy.state_hash, 'Execution policy changed')
    feedback.validate(rollout); comparison.validate(rollout)
    advantage=group_centered_advantages(feedback.rewards,feedback.valid).flatten()
    losses={'bc':0.,'policy':0.}; cfm=[]
    for start in range(0,len(condition),micro_batch):
        end=min(start+micro_batch,len(condition)); sl=slice(start*4,end*4)
        policy.train(); policy.zero_grad(set_to_none=True)
        with autocast_factory():
            bc=policy.bc_anchor(condition[start:end], targets[start:end],
                noise_generator=streams['bc_noise'],time_generator=streams['bc_time'],
                global_scene_count=global_scenes,world_size=1)
        require(torch.isfinite(bc),'Nonfinite original BC')
        bc.backward(); accumulator.add('bc',policy); losses['bc']+=float(bc.detach())
        policy.zero_grad(set_to_none=True)
        before_rng=replay.rng_identity(condition.device)
        with autocast_factory():
            current=policy(condition[start:end].repeat_interleave(4,0), rollout.raw_actions.flatten(0,1)[sl],
                comparison.samples.slice(sl),mc_chunk=mc_chunk,checkpoint_velocity=checkpoint_velocity)
            cfm.append(replay.exact_tensor(current, comparison.old_cfm[sl], 'current/old CFM'))
            loss,diagnostic=fpo_surrogate(comparison.old_cfm[sl],current,advantage[sl],feedback.valid.flatten()[sl],
                comparison.config,global_count=global_scenes*4,world_size=1)
        require(torch.equal(diagnostic['ratio'],torch.ones_like(diagnostic['ratio'])),'Old/current ratio differs')
        loss.backward(); accumulator.add('policy',policy); losses['policy']+=float(loss.detach())
        require(replay.rng_identity(condition.device)==before_rng, 'Policy branch consumed BC dropout RNG')
    policy.zero_grad(set_to_none=True)
    require(module_hash(policy)==rollout.policy.state_hash,'Diagnostic mutated private weights/buffers')
    return {'losses_global_contribution':losses,'nonzero_advantages':int(torch.count_nonzero(advantage)),
        'zero_advantage_scenes':int((advantage.view(-1,4)==0).all(1).sum()),'current_CFM':cfm}


@contextmanager
def isolated_native_rng(saved, streams):
    """GPU-only production scope; restore caller RNG even if an audit fails."""
    before=capture_rng(streams.generators)
    try:
        restore_rng(saved,streams.generators)
        yield
    finally:
        restore_rng(before,streams.generators)


def execute(data, budget):
    import socket
    r,cfg,binding=data['registration'],data['cfg'],data['binding']
    require(socket.gethostname()==r['resource']['host'],'Wrong admitted host')
    require(torch.cuda.device_count()==1,'Exactly one explicitly allocated GPU required')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',r['resource']['gpu_uuid']]).decode().strip()
    require(uuid==r['resource']['gpu_uuid'] and os.environ.get('CUDA_VISIBLE_DEVICES')==uuid,'Visible GPU must be the registered UUID')
    device='cuda:0';torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(cfg['runtime']['deterministic'])
    from tools.foresight.checkpoints import load_student
    with initialization_seed(cfg['execution']['seed']):
        encoder=load_student(binding['run'],binding['tag'],binding['parent_identity'],device='cpu',precision='fp32',strip=False)
    action=encoder.action_model;del encoder.action_model
    folder=Path(data['native']['run'])/'checkpoints'/data['native']['tag']
    require(file_sha256(folder/'COMPLETE.json') == data['native']['native_receipt_sha256'], 'Native receipt changed during parent loading')
    native_path=folder/'mp_rank_00_model_states.pt'
    require(file_sha256(native_path) == data['receipt']['files'][native_path.name]['sha256'], 'Native execution model changed after preflight')
    native=torch.load(native_path,map_location='cpu',mmap=True,weights_only=False)
    old=replay.restore_native_execution(action,native,data['receipt'],data['payloads'][0][0].policy.state_hash)
    del native
    old.action_head.model.gradient_checkpointing=cfg['runtime']['gradient_checkpointing']
    encoder.strip_auxiliary_heads();encoder.requires_grad_(False).eval();encoder.inference_fp32=False
    encoder.qwen_vl_interface.model.model.language_model.gradient_checkpointing_disable()
    from tools.st_world_policy.training_config import official_runtime_policy
    from tools.st_world_policy.official_runtime import apply_official_model_runtime
    official=official_runtime_policy(cfg)
    require(official is not None,'This diagnostic binds the original official FA2 runtime')
    apply_official_model_runtime(vlm=encoder.qwen_vl_interface.model,
        language=encoder.qwen_vl_interface.model.model.language_model,
        action_transformer=action.model,policy=official,training_copy=True)
    encoder.qwen_vl_interface.model.model.language_model.gradient_checkpointing_disable()
    encoder.bfloat16().to(device);old.to(device)
    require(replay.frozen_encoder_hash(encoder)==data['frozen_encoder_hash'],'Original frozen encoder differs')
    caller_rng=replay.rng_identity(device); conditions=[];requests=[];calls={'visual':0,'language':0}
    def hook(name):
        def count(*_): calls[name]+=1
        return count
    handles=[encoder.qwen_vl_interface.model.model.visual.register_forward_hook(hook('visual')),
        encoder.qwen_vl_interface.model.model.language_model.register_forward_hook(hook('language'))]
    try:
        for rank,entries in enumerate(data['rank_entries']):
            observations=replay.current_inputs({**data,'entries':entries})
            with torch.autocast('cuda',dtype=torch.bfloat16):
                condition=replay.encode_conditions(encoder,observations,cfg['execution']['micro_batch'])
            rollout,comparison,feedback=data['payloads'][rank]
            require(tuple(tensor_hash(x) for x in condition)==rollout.condition_hashes,'Original online condition differs')
            conditions.append(condition)
            gpu_rollout=RolloutBatch.from_state_dict(rollout.state_dict(),device=device)
            requests.extend(replay.score_requests(gpu_rollout,entries,binding,cfg['feedback'],data['native']['training_numeric_execution_hash']))
            budget.check()
    finally:
        for h in handles:h.remove()
    require(calls=={'visual':8,'language':8},'Original eight microbatch encodes required')
    require(replay.frozen_encoder_hash(encoder)==data['frozen_encoder_hash'] and
        all(p.grad is None for p in encoder.parameters()),'Frozen encoder mutated')
    del encoder;torch.cuda.empty_cache()
    require(replay.rng_identity(device)==caller_rng,'Frozen reconstruction consumed training RNG')
    # GPU raw->physical reproduces original request keys. No directory scan,
    # metric-cache load, scoring service construction, new call or retry.
    records,official_receipts=replay.read_completed_official_feedback(data['native']['run'],requests,
        run_identity=data['native']['run_identity'],cfg=cfg['feedback'])
    policy=deepcopy(old).requires_grad_(True).train();acc=GradientSum(policy);rank_reports=[]
    require(not ({p.untyped_storage().data_ptr() for p in old.parameters()} &
                 {p.untyped_storage().data_ptr() for p in policy.parameters()}),'Private policy aliases old')
    for rank,((rollout,comparison,feedback),condition,entries) in enumerate(zip(data['payloads'],conditions,data['rank_entries'])):
        rollout=RolloutBatch.from_state_dict(rollout.state_dict(),device=device)
        comparison=CoupledFPOBatch.from_state_dict(comparison.state_dict(),rollout,device=device)
        feedback=FeedbackBatch.from_state_dict(feedback.state_dict(),rollout,device=device)
        a,b=rank*32,(rank+1)*32
        verified=replay.feedback_from_records(rollout,requests[a:b],records[a:b],'C1',cfg['feedback']['score_protocol_hash'])
        replay.exact_tensor(verified.rewards,feedback.rewards,'saved canonical rewards')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            endpoints=replay.replay_endpoints(old,condition,rollout.initial_noise)
            replay.exact_tensor(endpoints,rollout.raw_actions,'ten-Euler replay')
            wrapper=replay.MatchedOldMicrobatches(replay.ReadOnlyPolicy(old,rollout.policy),4)
            cfm=wrapper(condition.repeat_interleave(4,0),rollout.raw_actions.flatten(0,1),comparison.samples,mc_chunk=cfg['runtime']['mc_chunk'])
            replay.exact_tensor(cfm,comparison.old_cfm,'old CFM50 replay')
        targets=[]
        for row in entries:
            label=torch.load(verified_artifact(row['ego']),map_location='cpu',weights_only=True)
            require(label['identity']==data['manifest']['ego_identity']['identity'] and label['token']==row['scene_id'], 'Original GT identity differs')
            targets.append(label['ego'])
        targets=torch.stack(targets).to(device)
        require(targets.shape==(8,8,4) and torch.isfinite(targets).all(),'GT shape/finite mismatch')
        path=folder/f'rng_rank{rank}.pt'
        require(file_sha256(path)==data['receipt']['files'][path.name]['sha256'],'Saved rank RNG changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        streams=FlowRandomStreams(cfg['execution']['seed']+rank,device=device)
        with isolated_native_rng(saved,streams):
            before=replay.rng_identity(device)
            report=audit_rank(policy,condition,targets,rollout,comparison,feedback,streams,acc,
                global_scenes=32,micro_batch=4,mc_chunk=cfg['runtime']['mc_chunk'],
                checkpoint_velocity=cfg['runtime']['checkpoint_velocity'],
                autocast_factory=lambda:torch.autocast('cuda',dtype=torch.bfloat16))
            report.update(rank=rank,scene_ids=list(rollout.scene_ids),native_rng_path=str(path),
                rng_before=before,rng_after_BC=replay.rng_identity(device),
                bc_streams_after={k:tensor_hash(streams[k].get_state()) for k in ('bc_noise','bc_time')})
            rank_reports.append(report)
        budget.check()
    require(module_hash(old)==module_hash(policy)==data['payloads'][0][0].policy.state_hash and
        all(p.grad is None for p in old.parameters()),'Original policy changed')
    require(replay.rng_identity(device)==caller_rng,'Caller RNG was not restored')
    return {'status':'DIAGNOSTIC_COMPLETE','gradient_balance':acc.report(cfg['loss']['policy_weight']),
        'rank_reports':rank_reports,'official_receipts':official_receipts,'encoder_calls':calls,
        'optimizer_steps':0,'new_score_calls':0,'original_weights_unchanged':True,'caller_RNG_unchanged':True,
        'original_precision':'BF16_execution_FP32_Euler_CFM_reduction_FA2_TF32off',
        'GT_access':'32 original training targets, BC only','metric_cache_read':False,
        'gpu_name':torch.cuda.get_device_name(), 'gpu_capability':list(torch.cuda.get_device_capability()),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'interpretation':'One fixed training boundary; no Adam/preconditioning/clip/update/gain inference. '
        'Separate BF16 backward and FP32 summation differ from joint ZeRO2 rounding; cross-device rank RNG mapped to one identical GPU architecture.'}


class Budget:
    def __init__(self,limits):
        self.limits=limits;self.start=time.monotonic();self.read_start=self.read_bytes()
    @staticmethod
    def read_bytes():
        return int(dict(line.split(':') for line in Path('/proc/self/io').read_text().splitlines())['rchar'])
    def check(self):
        elapsed=time.monotonic()-self.start;reads=self.read_bytes()-self.read_start
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        require(elapsed<=self.limits['max_seconds'] and reads<=self.limits['max_read_bytes'] and
            rss<=self.limits['max_rss_bytes'],'Registered wall/read/RSS budget exceeded')
        return {'seconds':elapsed,'logical_read_bytes':reads,'peak_RSS_bytes':rss}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',required=True,help='Immutable Lead registration binding source, native boundary, four receipts and resource limits')
    p.add_argument('--execute',action='store_true',help='Read real images/GT/models and run bounded gradient diagnostic; never train or score')
    args=p.parse_args(argv)
    result={'schema':SCHEMA,'optimizer_steps':0,'new_score_calls':0,'formal_training':False}
    out=None;old_handler=None
    try:
        r,training,cfg,receipt,run=validate_registration(args.registration)
        if not args.execute:
            print(json.dumps({'status':'SMALL_RECEIPTS_VALIDATED','execution':'NOT_RUN','boundary':receipt['state']['completed'],
                'GT_model_manifest_read':False,'registered_budget':r['budget']}));return 0
        candidate=Path(r['output']);candidate.mkdir(parents=True,exist_ok=False);out=candidate
        atomic_json(out/'REGISTRATION.json',r,immutable=True)
        torch.set_num_threads(r['resource']['cpu_threads']);budget=Budget(r['budget'])
        def timeout(*_):raise TimeoutError('Registered diagnostic wall limit exhausted')
        old_handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,r['budget']['max_seconds'])
        data=prepare_inputs(args.registration);budget.check()
        result.update(execute(data,budget),native_checkpoint=data['native'],cursor=data['cursor'],
            replay_sources=data['replay_sources'],resources=budget.check())
        code=0
    except Exception as exc:
        result.update(status='DIAGNOSTIC_REJECTED',error={'type':type(exc).__name__,'message':str(exc)},traceback=traceback.format_exc());code=1
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    if out is not None:
        encoded=json.dumps(result,allow_nan=False).encode()
        if len(encoded)>r['budget']['max_output_bytes']:
            result={'schema':SCHEMA,'status':'DIAGNOSTIC_REJECTED','error':'Report output budget exceeded'};code=1
        atomic_json(out/'RESULT.json',result,immutable=True)
    print(json.dumps({'status':result['status'],'output':str(out) if out else None,'error':result.get('error')}))
    return code


if __name__=='__main__':raise SystemExit(main())
