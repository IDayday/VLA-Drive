"""Read-only real SMOKE audit; never instantiate a model or score a trajectory."""
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import time

import torch

from starVLA.model.modules.flow_posttraining.advantage_scale import receipt_sha256, normalization_contract
from starVLA.model.modules.flow_posttraining.fpo_objective import group_centered_advantages
from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash
from starVLA.model.modules.flow_posttraining.rollout_contract import RolloutBatch, FeedbackBatch, CoupledFPOBatch
from tools.st_world_policy import training_state, train_flow_policy


ROOT = Path('/mnt/project/st-world-policy-20261016/smoke/scene_center_rollout_std_v1_4update')
RUN = ROOT / 'runs/C1_rollout_std_smoke_v1'
OUT = Path(__file__).resolve().parent
SOURCE = '751fcfa2b41549ae16f7b3204aef8dd18b912016'
reads = []


def read(path):
    path = Path(path)
    data = path.read_bytes()
    reads.append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return data


def js(path):
    return json.loads(read(path))


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    start = time.monotonic()
    result = js(RUN / 'RESULT.json')
    outer = js(ROOT / 'EXIT.json')
    identity = js(RUN / 'identity.json')
    assert outer['exit_code'] == 0
    assert result['source_sha'] == SOURCE and identity['source_sha'] == SOURCE
    assert result['completed'] == result['new_optimizer_updates'] == result['policy_version'] == 4
    assert result['status'] == 'PAUSED' and result['termination_reason'] == 'RUNNING_CHECKPOINT'
    assert result['exposure'] == 128 and result['frozen_encoder_unchanged'] is True
    assert result['scientific_result'] == 'NOT_EVALUATED' and not result['formal_launch_approval']
    assert not identity['weights_usable_for_formal_init']
    steps = [json.loads(line) for line in read(RUN / 'steps.jsonl').splitlines() if line]
    assert [s['step'] for s in steps] == [1, 2, 3, 4]
    assert all(finite(s) for s in steps)
    assert all(s['fp32_master_update_verified'] and s['sampled_changed_elements'] > 0 for s in steps)
    assert all(s['gradient_norm'] > 0 for s in steps)
    assert [s['official_admitted_calls'] for s in steps] == [128, 256, 384, 512]

    # Do this expensive check exactly once: only the final complete native tag.
    folder = RUN / 'checkpoints' / result['tag']
    metadata = js(folder / 'COMPLETE.json')
    total = sum(v['bytes'] for v in metadata['files'].values())
    assert metadata['state']['completed'] == 4 and total < 12 * 2**30
    original_hash = training_state.file_sha256
    checkpoint_files = []
    def measured_hash(path):
        before = Path(path).stat()
        value = original_hash(path)
        after = Path(path).stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        checkpoint_files.append({'path':str(path), 'bytes':before.st_size, 'sha256':value})
        return value
    training_state.file_sha256 = measured_hash
    # The earlier CPU-sum replay attempt already completed this checksum pass.
    # Reuse that sealed verification, never reread 11.47 GB merely to fix the
    # independent auditor's CPU-vs-CUDA reward reduction order.
    prior = OUT / 'NATIVE_CHECKPOINT_VERIFICATION.json'
    if prior.exists():
        checkpoint_cost = json.loads(prior.read_text())
        assert checkpoint_cost['native_verify_calls'] == 1
        assert checkpoint_cost['logical_checksum_bytes'] == total
        verified = {Path(x['path']).name: {'bytes':x['bytes'],'sha256':x['sha256']}
                    for x in checkpoint_cost['files']}
        assert verified == metadata['files']
        assert all(Path(x['path']).parent == folder and Path(x['path']).stat().st_size == x['bytes']
                   for x in checkpoint_cost['files'])
        checked = metadata
    else:
        tick = time.monotonic(); cpu = time.process_time()
        checked = training_state.verify_checkpoint(folder, identity=identity['identity'], world_size=4)
        checkpoint_cost = {'wall_seconds':time.monotonic()-tick, 'cpu_seconds':time.process_time()-cpu,
                           'logical_checksum_bytes':sum(v['bytes'] for v in checkpoint_files),
                           'files':checkpoint_files, 'native_verify_calls':1,
                           'model_tensor_deserialization':False}
        assert checked == metadata and checkpoint_cost['logical_checksum_bytes'] == total
        prior.write_text(json.dumps(checkpoint_cost,indent=2)+'\n')

    boundary_results = []
    for boundary, step in enumerate(steps):
        raw_parts = []; common = None; common_hash = None
        assert len(step['per_rank']) == 4
        for rank in range(4):
            receipt = js(RUN / 'rollouts' / f'boundary_{boundary:06d}' / f'rank_{rank}_COMPLETE.json')
            ident = receipt['identity']; norm = ident['advantage_normalization']; local = norm['local']
            if common is None: common, common_hash = norm['common'], norm['common_sha256']
            assert norm['common'] == common and norm['common_sha256'] == common_hash == receipt_sha256(common)
            assert common['definition'] == normalization_contract()
            assert common['count'] == common['expected_global_count'] == 128 and common['world_size'] == 4
            assert common['identity']['numeric_execution_hash'] == identity['numeric_execution_hash']
            assert common['identity']['old_policy'] == step['old_policy'] == ident['old_policy']
            assert local['rank'] == rank and local['local_count'] == 32
            assert common['rank_records'][rank] == {k:v for k,v in local.items() if k != 'identity'}
            assert local['local_identity_sha256'] == receipt_sha256(local['identity'])
            data = read(receipt['payload']['path'])
            assert hashlib.sha256(data).hexdigest() == receipt['payload']['sha256']
            payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            assert payload['schema'] == 'st_flow_update_feedback_v2' and payload['identity'] == ident
            rollout = RolloutBatch.from_state_dict(payload['rollout'])
            feedback = FeedbackBatch.from_state_dict(payload['feedback'], rollout)
            comparison = CoupledFPOBatch.from_state_dict(payload['comparison'], rollout)
            assert comparison.comparison_id == ident['comparison_id']
            assert local['identity'] == train_flow_policy.feedback_scale_identity(rollout, feedback)
            assert feedback.valid.all() and all(x == 'valid_scored' for x in feedback.statuses)
            assert feedback.rewards.dtype == torch.float32 and feedback.rewards.shape == (8,4)
            assert tensor_hash(feedback.rewards) == local['rewards_sha256']
            assert tensor_hash(feedback.valid) == local['valid_sha256']
            # This fixed G4 paired reduction exactly reproduces the SAVED CUDA
            # raw hash. The default CPU torch.sum uses a different FP32 tree.
            r = feedback.rewards
            raw = r - (((r[:,0]+r[:,1])+(r[:,2]+r[:,3]))/4)[:,None]
            assert tensor_hash(raw) == local['raw_advantages_sha256']
            normalized = payload['normalized_advantages']
            assert normalized.dtype == torch.float32 and not normalized.requires_grad
            assert normalized.shape == (8,4) and torch.isfinite(normalized).all()
            assert tensor_hash(normalized) == ident['normalized_advantages_sha256'] == local['normalized_advantages_sha256']
            expected = (raw.double() / (common['std'] + 1e-10)).float()
            assert torch.equal(normalized, expected)
            assert torch.equal(normalized[raw == 0], raw[raw == 0])
            assert list(rollout.scene_ids) == step['per_rank'][rank]['sample_ids']
            assert list(rollout.scene_ids) == common['identity']['global_scene_ids'][rank::4]
            assert step['per_rank'][rank]['official']['valid_candidates'] == 32
            assert step['per_rank'][rank]['advantage_normalization']['common_sha256'] == common_hash
            raw_parts.append(raw)
        all_raw = torch.cat(raw_parts).double().flatten()
        expected_m2 = (all_raw - all_raw.mean()).square().sum().item()
        # FP64 distributed vs serial summation differs only in reduction order.
        assert math.isclose(expected_m2, common['centered_m2'], rel_tol=5e-15, abs_tol=1e-20)
        assert math.isclose(math.sqrt(expected_m2/127), common['std'], rel_tol=5e-15, abs_tol=1e-20)
        boundary_results.append({'boundary':boundary, 'common_sha256':common_hash,
             'N':128, 'nonzero_count':common['nonzero_count'], 'std':common['std'], 'gain':common['gain'],
             'all_tensor_hashes_and_cpu_formula_exact':True, 'raw_replay_tree':'FP32 ((r0+r1)+(r2+r3))/4; exact saved CUDA raw hash required', 'distributed_m2_absolute_error':abs(expected_m2-common['centered_m2'])})

    # Actual recorded native-engine qualification on each rank after first update.
    engines=[]
    for rank in range(4):
        e=js(RUN/'official_runtime'/f'updated_engine_rank{rank}.json')
        assert e['execution_dtype'] == e['gradient_accumulation_dtype'] == 'bfloat16'
        assert e['master_dtype'] == 'float32' and e['moments_required']
        assert e['moment_elements'] == 2*e['master_elements'] and e['master_elements'] > 0
        engines.append(e)
    # Invoke the production consumed-normalization guard, not model resume.
    norm_reads=[]; old_verified=train_flow_policy._verified_artifact
    def record_verified(artifact):
        value=old_verified(artifact)
        norm_reads.append({'path':artifact['path'],'bytes':Path(artifact['path']).stat().st_size,
                           'sha256':artifact['sha256']})
        return value
    train_flow_policy._verified_artifact=record_verified
    before=torch.get_rng_state().clone()
    for rank in range(4):
        train_flow_policy.verify_restored_normalization(RUN, checked['state'], rank=rank,
                                                      numeric_hash=identity['numeric_execution_hash'])
    assert torch.equal(before,torch.get_rng_state())
    assert checked['state']['last_advantage_scale_common_sha256'] == boundary_results[-1]['common_sha256']

    output={'status':'PASS_REAL_FOUR_RANK_SMOKE','source_sha':SOURCE,'run':str(RUN),
      'result':result,'outer_exit':outer,'identity_hash':identity['identity'],
      'numeric_execution_hash':identity['numeric_execution_hash'],
      'validated_feedback_candidates':512,'scene_presentations':128,'unique_scenes':32,
      'normalized_advantages_dtype':'torch.float32','boundaries':boundary_results,
      'steps':[{'step':s['step'],'losses':s['losses'],'gradient_norm':s['gradient_norm'],
               'lr':s['lr'],'slowest_rank_seconds':s['slowest_rank_seconds'],
               'normalization_rank_max_seconds':max(x['seconds']['advantage_normalization'] for x in s['per_rank']),
               'peak_allocated_max':max(x['peak_allocated'] for x in s['per_rank']),
               'peak_reserved_max':max(x['peak_reserved'] for x in s['per_rank'])} for s in steps],
      'recorded_engine_checks':engines,'engine_check_scope':'Actual runtime checks after first optimizer update; no independent model or Adam tensor deserialization.',
      'checkpoint':{'path':str(folder),'receipt_identity':checked['receipt_identity'],**checkpoint_cost},
      'consumed_normalization_guard':{'ranks':[0,1,2,3],'RNG_unchanged':True,'payload_checksum_reads':norm_reads},
      'direct_read_inventory':reads,'direct_read_bytes':sum(x['bytes'] for x in reads),
      'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time(),
      'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'reviewer_model_loaded':False,'reviewer_gpu_used':False,'new_score_calls':0,'new_optimizer_steps':0,
      'real_GPU_resume_executed':False,'prior_cpu_sum_replay_failed_and_retained':True,'scientific_gain':'UNTESTED','formal_training_approved':False}
    (OUT/'ACTUAL_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['status','validated_feedback_candidates','direct_read_bytes','wall_seconds','cpu_seconds','maxrss_kib']}))


if __name__ == '__main__':
    main()
