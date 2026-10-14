"""Algorithm-state adapter around the existing Accelerator/ZeRO checkpoint API."""
import json
import os
from pathlib import Path
import shutil
import torch
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.checkpoint import capture_rng, restore_rng, directory_seal
from starVLA.rl.flow_grpo.distributed import rank0_call, rank0_result, synchronized_call
from starVLA.rl.flow_grpo.loading import weight_path
from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import file_sha, SFT_SHA256


def save_checkpoint(accelerator, engine, root, state, identity, *, label, behavior=None):
    final=Path(root)/'checkpoints'/label; temporary=final.with_name('.'+label+'.incomplete')
    def prepare():
        if final.exists() or temporary.exists(): raise FileExistsError(final)
        temporary.mkdir(parents=True)
    rank0_call(prepare,accelerator.device)
    # DeepSpeed owns all parameter/master-weight/moment partition collectives.
    accelerator.save_state(str(temporary),safe_serialization=False)
    synchronized_call(lambda:torch.save(dict(rng=capture_rng(),state=state),temporary/f'rank{accelerator.process_index}.pt'),accelerator.device)
    def publish():
        if state['phase']=='updating':
            if behavior is None: raise ValueError('Pending inner epoch requires the frozen old behavior')
            torch.save({k:v.detach().cpu() for k,v in behavior.state_dict().items()},temporary/'old_behavior.pt')
        atomic_json(temporary/'identity.json',identity)
        atomic_json(temporary/'boundary.json',dict(schema='cfn_boundary_v1',phase=state['phase'],
                    update=state['updates'],collection=state['collections'],raw_exposure=state['raw_cursor'],
                    next_inner=state['next_inner'],world=accelerator.num_processes))
        atomic_json(temporary/'files.json',directory_seal(temporary))
        (temporary/'COMPLETE').write_text('Atomic complete actor/dual/RNG/collector boundary\n')
        os.replace(temporary,final)
        atomic_json(Path(root)/'latest_checkpoint.json',dict(path=str(final),label=label,boundary=state['phase']))
    rank0_call(publish,accelerator.device)
    return final


def resume_checkpoint(accelerator, engine, path, identity, behavior=None):
    path=Path(path)
    def verify():
        if not (path/'COMPLETE').is_file() or json.loads((path/'identity.json').read_text()) != identity:
            raise ValueError('Incomplete/changed resume identity')
        if json.loads((path/'files.json').read_text()) != directory_seal(path,('files.json','COMPLETE')):
            raise ValueError('Checkpoint content changed')
        if json.loads((path/'boundary.json').read_text())['world'] != accelerator.num_processes:
            raise ValueError('World-size change invalidates exact resume')
    rank0_call(verify,accelerator.device)
    accelerator.load_state(str(path))
    value=synchronized_call(lambda:torch.load(path/f'rank{accelerator.process_index}.pt',map_location='cpu',weights_only=False),accelerator.device)
    restore_rng(value['rng'])
    state=value['state']
    if state['phase'] not in ('ready','collecting','updating','complete'):
        raise ValueError('Unknown saved phase')
    if state['phase']=='updating' and not (state['collector'] and 0 <= state['next_inner'] < 2):
        raise ValueError('Missing fixed old chain/advantage/lambda')
    if state['phase']=='updating':
        if behavior is None: raise ValueError('Missing frozen old behavior receiver')
        behavior.load_state_dict(torch.load(path/'old_behavior.pt',map_location='cpu',weights_only=True),strict=True)
    return state


def export_policy(actor, sft_checkpoint, output, identity):
    """Merge only the updated action head into the exact original SFT export.

    The original encoder/history state is never converted by DeepSpeed. Noise
    parameters, scoring, cached features and teacher state are not exported.
    """
    output=Path(output); temporary=output.with_name(output.name+'.incomplete')
    if output.exists() or temporary.exists(): raise FileExistsError(output)
    source=Path(sft_checkpoint); path=weight_path(source)
    if file_sha(path) != SFT_SHA256: raise ValueError('Original SFT export changed')
    temporary.mkdir(parents=True)
    with torch.serialization.safe_globals([set]):
        original=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    original=original.get('module',original)
    current=actor.policy.action_model.state_dict()
    head_keys={name.removeprefix('action_model.') for name in original if name.startswith('action_model.')}
    if head_keys != set(current): raise ValueError('Action export key closure changed')
    result={k:(current[k.removeprefix('action_model.')].detach().cpu().to(v.dtype)
               if k.startswith('action_model.') else v) for k,v in original.items()}
    if any(k.startswith('noise_head.') for k in result): raise ValueError('Noise head entered deployment')
    torch.save(result,temporary/'pytorch_model.pt')
    for name in ('config.yaml','normalization.json','sft_parameter_manifest.json'):
        if (source/name).exists(): shutil.copy2(source/name,temporary/name)
    if (source/'processor').is_dir(): shutil.copytree(source/'processor',temporary/'processor')
    atomic_json(temporary/'cfn_export_identity.json',dict(run_identity=identity,SFT_sha256=SFT_SHA256,
                export_sha256=file_sha(temporary/'pytorch_model.pt'),changed_prefix='action_model.',
                upstream_preserved_bytewise_tensor_values=True,noise_head_in_export=False,
                deployment='original_single_candidate_10_step_ODE'))
    (temporary/'COMPLETE').write_text('Original inference export; no training dependencies\n')
    os.replace(temporary,output)
    return output
