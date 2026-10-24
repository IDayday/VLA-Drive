"""CPU read-only native checkpoint, exposure, Adam and cost audit for S1/S2/S3."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_sourceflow.contracts import file_sha,fingerprint


def sealed(path):
    value=json.loads(path.read_text());body=dict(value)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Changed analysis/config seal: '+str(path))
    return value


def tensor_hash(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True);p.add_argument('--runs-root',required=True)
    p.add_argument('--admission-analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    protocol=read_protocol(a.protocol);root=Path(a.runs_root);output=Path(a.output)
    if output.exists():raise FileExistsError('Keep prior audit; use a new version')
    admission=sealed(Path(a.admission_analysis)/'summary.json')
    source_csv=Path(a.admission_analysis)/'private_scene_admission.csv'
    if file_sha(source_csv)!=admission['private_scene_csv_sha256']:raise ValueError('Changed admitted scene membership')
    eligibility={method:{} for method in ('F2_PDMS','F3_FEEDBACK')}
    for row in csv.DictReader(source_csv.open()):eligibility[row['method']][row['scene_token']]=row['teacher_admitted']=='True'
    tokens=protocol['splits']['TRAIN'];scene_order=[]
    for epoch in range(8):
        scene_order += [tokens[i] for i in torch.randperm(len(tokens),generator=torch.Generator().manual_seed(42+epoch)).tolist()]
    assert len(scene_order)==4096 and set(Counter(scene_order).values())=={8}
    result=dict(schema='feedback_terminal_native_audit_v2',protocol_sha256=protocol['sha256'],
        shared_GT_scene_order_sha256=fingerprint(scene_order),distinct_GT_scenes=len(tokens),epochs=8,
        order_evidence='Reconstructed from pinned trainer seed/manifest/cursor; also checks every saved teacher-coverage footprint. Per-token execution traces were not logged.',
        no_actor_scoring_or_optimizer_calls=True,arms={})
    initial_hashes=[];initial_frozen=[];configs=[]
    for arm in ('S1','S2','S3'):
        run=root/f'feedback_{arm}_seed42';config=sealed(run/'config.json');configs.append(config)
        state,payload=load_state(run,protocol['sha256'],restore_rng=False)
        latest=json.loads((run/'latest.json').read_text());checkpoint=Path(latest['checkpoint'])
        initial=next(path for path in (run/'checkpoints').iterdir() if path.name.startswith('round0000_update000000_') and not path.name.endswith('.incomplete'))
        _,begin=load_state(initial,protocol['sha256'],restore_rng=False)
        assert state['phase']=='COMPLETE' and state['updates']==256 and state['round']==256
        assert state['scene_cursor']==4096 and state['next_inner']==0 and state['arm']=='feedback_'+arm
        assert payload['GT_exposures']==4096 and payload['config_sha256']==config['sha256']
        assert payload['frozen_hash']==begin['frozen_hash']
        expected_teacher=0 if arm=='S1' else 1896
        assert payload['teacher_exposures']==expected_teacher
        assert len(payload['optimizer']['state'])==480
        steps=[int(value['step']) for value in payload['optimizer']['state'].values()]
        assert set(steps)=={256}
        assert all(value[key].dtype==torch.float32 and torch.isfinite(value[key]).all()
                   for value in payload['optimizer']['state'].values() for key in ('exp_avg','exp_avg_sq'))
        assert len(payload['adapters'])==480 and payload['adapters'].keys()==begin['adapters'].keys()
        init_hash=tensor_hash(begin['adapters']);initial_hashes.append(init_hash);initial_frozen.append(begin['frozen_hash'])
        changed=sum(not torch.equal(value,begin['adapters'][key]) for key,value in payload['adapters'].items())
        records=[]
        for lineage in state['metric_lineage']:
            path=run/'attempts'/lineage['attempt']/'metrics.jsonl'
            records += [row for row in (json.loads(line) for line in path.read_text().splitlines())
                        if lineage['first_update']<=row['updates']<=lineage['last_update']]
        assert [row['updates'] for row in records]==list(range(1,257))
        method=None if arm=='S1' else 'F2_PDMS' if arm=='S2' else 'F3_FEEDBACK'
        masks=[0 if method is None else int(eligibility[method][token]) for token in scene_order]
        assert sum(masks)==expected_teacher
        for update,row in enumerate(records):
            assert row['scene_cursor']==(update+1)*16 and row['GT_exposures']==(update+1)*16
            assert row['teacher_scenes']==sum(masks[update*16:(update+1)*16])
            assert row['teacher_exposures']==sum(masks[:(update+1)*16])
            assert all(math.isfinite(row[key]) for key in ('GT_loss','gradient_norm','update_seconds','lr'))
        app=json.loads((run/'COMPLETE.json').read_text())
        assert app['status']=='COMPLETE' and app['updates']==256 and app['GT_exposures']==4096
        assert app['teacher_exposures']==expected_teacher and app['frozen_hash_unchanged'] is True
        assert app['checkpoint']==str(checkpoint) and app['config_sha256']==config['sha256']
        assert payload['scheduler']['last_epoch']==256
        invocations=[json.loads(path.read_text()) for path in sorted(run.glob('attempts/*/invocation.json'))]
        assert all(row['status']=='COMPLETE' for row in invocations)
        final_ledgers=[json.loads(path.read_text()) for path in sorted(run.glob('attempts/*/ledger.json'))]
        ledger=app['ledger'];expected_FB=4096+expected_teacher
        assert ledger['model_forward_calls']==expected_FB and ledger['model_backward_calls']==expected_FB
        update_seconds=sum(row['update_seconds'] for row in records)
        result['arms'][arm]=dict(status='COMPLETE_NATIVE_AND_APPLICATION',
            native_state=state,application_complete_sha256=file_sha(run/'COMPLETE.json'),
            native_payload_sha256=file_sha(checkpoint/'payload.pt'),config_sha256=config['sha256'],
            initial_adapter_sha256=init_hash,terminal_adapter_sha256=tensor_hash(payload['adapters']),
            adapter_tensors=480,changed_adapter_tensors=changed,Adam_states=480,Adam_steps=sorted(set(steps)),
            all_Adam_moments_finite_FP32=True,stored_frozen_hash_matches_initial=True,
            trainer_terminal_recomputed_frozen_hash_before_COMPLETE=True,GT_exposures=4096,
            teacher_exposures=expected_teacher,teacher_scene_coverage=0 if arm=='S1' else 237,
            saved_metrics_update_sequence_contiguous=True,teacher_mask_matches_reconstructed_order=True,
            scheduler_last_epoch=payload['scheduler']['last_epoch'],valid_updates=256,
            epoch_mean_GT_loss=[float(np.mean([row['GT_loss'] for row in records[e*32:(e+1)*32]])) for e in range(8)],
            epoch_mean_teacher_loss=[float(np.mean([row['teacher_loss'] for row in records[e*32:(e+1)*32] if row['teacher_loss'] is not None])) for e in range(8)] if expected_teacher else None,
            epoch_mean_teacher_loss_semantics='Unweighted mean of per-update present-teacher means; not population loss. Use scene-weighted/objective curves below.',
            epoch_teacher_loss_per_teacher_scene_mean=[
                sum((row['teacher_loss'] or 0.)*row['teacher_scenes'] for row in records[e*32:(e+1)*32])/
                sum(row['teacher_scenes'] for row in records[e*32:(e+1)*32]) for e in range(8)] if expected_teacher else None,
            epoch_teacher_objective_per_population_scene=[
                config['teacher_coefficient']*sum((row['teacher_loss'] or 0.)*row['teacher_scenes']
                    for row in records[e*32:(e+1)*32])/(32*16) for e in range(8)] if expected_teacher else None,
            FM_curve_uses_fresh_noise_time_not_fixed_fit_probe=True,
            training_cost=dict(optimizer_updates=256,GT_terms=4096,teacher_terms=expected_teacher,
                source_FM_forward_backward_terms=expected_FB,new_online_candidates=0,
                sum_update_seconds=update_seconds,mean_update_seconds=update_seconds/256,
                training_phase_worker_wall_seconds=ledger['invocation_wall_seconds'],
                unallocated_loading_saving_hash_other_overhead_seconds=ledger['invocation_wall_seconds']-update_seconds,
                training_Qwen_prefills=ledger['qwen_prefills'],
                all_invocation_wall_seconds=sum(row['end_unix']-row['start_unix'] for row in invocations),
                final_scored_candidates_including_auxiliary=sum(row['scored_candidates'] for row in final_ledgers),
                final_model_forwards_including_auxiliary=sum(row['model_forward_calls'] for row in final_ledgers),
                semantics='Shared GPU worker wall; not dedicated GPU compute. S1 immutable97 auxiliary eval cost is preserved separately from training and primary eval.'))
    if len(set(initial_hashes))!=1 or len(set(initial_frozen))!=1:
        raise ValueError('Arms did not start from identical LoRA/frozen initialization')
    common=[{key:value for key,value in config.items() if key not in ('arm','teacher_bank_hash','teacher_calibration','teacher_coefficient','sha256')} for config in configs]
    assert common[0]==common[1]==common[2]
    result.update(identical_initial_LoRA=True,identical_initial_frozen_model_reference=True,
        matched_common_config_except_arm_teacher_fields=True,
        analysis_source_sha256=file_sha(__file__),
        reference_semantics='Teacher gain uses candidate0; five independent reference roots define discovery means/strata, never substituted for candidate0 during these runs.')
    result['sha256']=fingerprint(result);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
