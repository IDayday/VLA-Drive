"""Compare actual continuous and resumed actor/FP32 optimizer/dual/RNG states."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise ValueError('Tensor mismatch: '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('Numpy mismatch: '+path)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Key mismatch: '+path)
        for key in a:equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Length mismatch: '+path)
        for i,(left,right) in enumerate(zip(a,b)):equal(left,right,path+'.'+str(i))
    elif type(a) is type(b) and hasattr(a,'__dict__'):
        equal(vars(a),vars(b),path+'.__dict__')
    elif a!=b:raise ValueError('Scalar mismatch: '+path)


def without_execution_timing(value):
    timing={'rollout_seconds','scoring_seconds','reference_seconds','scoring_cpu_seconds',
            'geometry_seconds','scene_scoring_cpu_seconds','scene_scoring_wall_seconds'}
    if isinstance(value,dict):return {k:without_execution_timing(v) for k,v in value.items() if k not in timing}
    if isinstance(value,(list,tuple)):return [without_execution_timing(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True)
    p.add_argument('--output',required=True);p.add_argument('--chains-only',action='store_true')
    a=p.parse_args();host=require_dispatch();torch.set_num_threads(4)
    roots=[Path(a.continuous),Path(a.resumed)]
    if Path(a.output).exists():raise FileExistsError(a.output)
    identities=[json.loads((r/'run_identity.json').read_text()) for r in roots]
    equal(*identities)
    checkpoints=[Path(json.loads((r/'latest_checkpoint.json').read_text())['path']) for r in roots]
    reports=[json.loads((r/'result_rank0.json').read_text()) for r in roots]
    if any(x['status']!='COMPLETE_FIXED_RAW_BUDGET' for x in reports):raise ValueError('Both runs must reach their same registered endpoint')
    if a.chains_only:
        files=[sorted((root/'behavior_collections').glob('*.pt')) for root in roots]
        if [p.name for p in files[0]]!=[p.name for p in files[1]]:raise ValueError('Behavior collection count changed')
        for left,right in zip(*files):
            values=[without_execution_timing(torch.load(p,map_location='cpu',weights_only=False)) for p in (left,right)]
            equal(*values,path=left.name)
        atomic_json(Path(a.output),dict(status='PASS',host=host,identity=identities[0]['sha256'],
                    collections=len(files[0]),all_saved_chains_means_stds_logprobs_advantages_lambdas_and_feedback_exact=True,
                    excluded='Only seven explicitly named execution timings; physical event times and all simulations compared'))
        return
    evidence=[]
    for relative in sorted(p.relative_to(checkpoints[0]) for p in (checkpoints[0]/'pytorch_model').glob('*.pt')):
        values=[torch.load(r/relative,map_location='cpu',weights_only=False,mmap=True) for r in checkpoints]
        # Actual DS files contain scientific states as well as save/load bookkeeping.
        if 'optim_states' in str(relative):
            equal(*values,path=str(relative));evidence.append(str(relative))
        else:
            equal(values[0]['module'],values[1]['module'],path='current_action_and_noise')
            for key in ('global_steps','global_samples','skipped_steps'):
                if key in values[0]:equal(values[0][key],values[1][key],path=key)
            evidence.append('current_action_and_noise; engine_step_counters')
        del values
    for rank in range(8):
        values=[torch.load(r/f'rank{rank}.pt',map_location='cpu',weights_only=False) for r in checkpoints]
        equal(values[0]['rng'],values[1]['rng'],path=f'rank{rank}.RNG')
        for key in ('phase','updates','collections','raw_cursor','accepted_exposures','replay_cursor','RL_updates',
                    'replay_reference_only_updates','next_inner','collector','dual','KL','collection_LR','completed_milestones'):
            equal(values[0]['state'][key],values[1]['state'][key],path=f'rank{rank}.'+key)
        for key in ('pending_evaluation','completed_evaluations','saturation_streak','safety_pause'):
            if key in values[0]['state'] or key in values[1]['state']:
                equal(values[0]['state'][key],values[1]['state'][key],path=f'rank{rank}.'+key)
    exports=[r/f'export_raw{reports[0]["raw_cursor"]:06d}'/'pytorch_model.pt' for r in roots]
    exported=[torch.load(path,map_location='cpu',weights_only=True,mmap=True) for path in exports]
    equal(*exported,path='original_ODE_full_export');del exported
    atomic_json(Path(a.output),dict(status='PASS',host=host,identity=identities[0]['sha256'],
                actor_and_noise_exact=True,FP32_optimizer_all_8_partitions_exact=True,RNG_all_8_ranks_exact=True,
                dual_optimizer_exact=True,data_raw_accepted_replay_and_inner_cursors_exact=True,KL_controller_exact=True,
                original_ODE_export_exact=True,checked_optimizer_files=evidence,
                excluded='wall/CPU timing, output paths and sampled diagnostic probe history; no scientific state excluded'))


if __name__=='__main__':main()
