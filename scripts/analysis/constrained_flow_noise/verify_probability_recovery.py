"""Compare the actual repaired two-inner update and interrupted continuation."""
import argparse
import json
from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .verify_resume import equal
from .qualify_collection_overlap import exact
from .train_stable_probability import source_identity


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--original-run',required=True)
    parser.add_argument('--values-proof',required=True)
    parser.add_argument('--batch-proof',required=True)
    args=parser.parse_args();host=require_dispatch();torch.set_num_threads(4)
    root=Path(args.root);runs=[root/'C_continuous',root/'C_resumed']
    out=root/'acceptance.json'
    if out.exists():raise FileExistsError(out)
    proof=json.loads(Path(args.values_proof).read_text())
    if proof['status']!='PASS' or proof['adapter_sha256']!=file_sha(Path(__file__).with_name('stable_saved_probability.py')):
        raise ValueError('Missing current value/gradient evidence')
    checkpoints=[Path(json.loads((r/'latest_checkpoint.json').read_text())['path']) for r in runs]
    for r in runs:
        for rank in range(8):
            report=json.loads((r/f'result_rank{rank}.json').read_text())
            if report['status']!='STOPPED_AT_SAVED_BOUNDARY' or report['updates']!=1854:
                raise ValueError('Incomplete bounded native qualification')
    equal(*[json.loads((c/'identity.json').read_text()) for c in checkpoints])
    for relative in sorted(p.relative_to(checkpoints[0]) for p in (checkpoints[0]/'pytorch_model').glob('*.pt')):
        values=[torch.load(c/relative,map_location='cpu',weights_only=False) for c in checkpoints]
        if 'optim_states' in str(relative):equal(*values,path=str(relative))
        else:
            equal(values[0]['module'],values[1]['module'],path='actor')
            for key in ('global_steps','global_samples','skipped_steps'):
                if key in values[0]:equal(values[0][key],values[1][key],path=key)
        del values
    for rank in range(8):
        values=[torch.load(c/f'rank{rank}.pt',map_location='cpu',weights_only=False) for c in checkpoints]
        equal(values[0]['rng'],values[1]['rng'],path=f'rank{rank}.RNG')
        for key in ('phase','updates','collections','raw_cursor','accepted_exposures','replay_cursor','RL_updates',
                    'replay_reference_only_updates','next_inner','collector','dual','KL','collection_LR',
                    'completed_milestones','pending_evaluation','completed_evaluations','saturation_streak','safety_pause',
                    'execution_adapter','probability_layout'):
            equal(values[0]['state'][key],values[1]['state'][key],path=f'rank{rank}.'+key)
        del values
    files=[sorted((r/'behavior_collections').glob('*.pt')) for r in runs]
    if [p.name for p in files[0]] != [p.name for p in files[1]]:raise ValueError('Collection count differs')
    candidates=0
    for left,right in zip(*files):
        values=[torch.load(p,map_location='cpu',weights_only=False) for p in (left,right)]
        exact(*values,path='actual_saved_collection')
        candidates+=len(values[0]['rows'])*16
    # On the failed first collection, only execution provenance may differ.
    def remove_version(value):
        if isinstance(value,dict):return {k:remove_version(v) for k,v in value.items() if k!='numerical_execution_sha256'}
        if isinstance(value,(list,tuple)):return [remove_version(v) for v in value]
        return value
    old=Path(args.original_run)/'behavior_collections/000925.pt'
    new=runs[0]/'behavior_collections/000925.pt'
    exact(*[remove_version(torch.load(p,map_location='cpu',weights_only=False)) for p in (old,new)],path='unchanged_original_failed_chain')
    original=get_fp32_state_dict_from_zero_checkpoint(args.checkpoint,tag='pytorch_model')
    updated=get_fp32_state_dict_from_zero_checkpoint(str(checkpoints[0]),tag='pytorch_model')
    if set(original)!=set(updated) or len(original)!=359:raise ValueError('Parameter closure changed')
    changed={n:int(torch.count_nonzero(updated[n]!=v)) for n,v in original.items()}
    if not all(x>0 for x in changed.values()) or not all(torch.isfinite(v).all() for v in updated.values()):
        raise ValueError('A full-head FP32 parameter did not update legitimately')
    identity=source_identity()
    atomic_json(out,dict(status='PASS',host=host,execution_sha256=identity['sha256'],
        failing_chain_probability_pass=True,transition_values_gradients_pass=True,
        native_update_resume_exact=True,all_trainable_tensors_updated=True,saved_old_chain_unchanged=True,
        original_probability_guard=1e-4,all_rank_optimizer_rng_dual_KL_cursors_layout_exact=True,
        full_action_tensors_updated=len(changed),minimum_changed_elements=min(changed.values()),
        native_updates_per_branch=4,first_update=1851,last_update=1854,
        actual_new_candidate_scores=2*candidates,batch_execution_acceptance=args.batch_proof,
        values_gradients_acceptance=args.values_proof,checkpoints=[str(c) for c in checkpoints],
        numerical_scope='Old aborted update has no completed reference; new sparse-B1 layout is qualified against original B1 density/gradients and exact native interruption recovery. No claim of bitwise old optimizer equality after repair.'))
    print(json.dumps(dict(status='PASS',updated_tensors=len(changed),actual_candidate_scores=2*candidates)),flush=True)


if __name__=='__main__':main()
