"""Adapter-only export and direct plain-Linear merge; no reference actor needed."""
import gc
import json
import os
from pathlib import Path
import uuid
import torch
from torch import nn
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint
from starVLA.rl.qwen_sourceflow.actor import attention_paths
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.flow_grpo.transactions import atomic_json


DEPLOYMENT_ARMS = ('P0','P1','P2','P3','P4','P5','S1','S2','S3')


def checkpoint_training_identity(checkpoint, state, payload, protocol, *, arm, seed, qualification):
    """Validate each existing trainer's own sealed identity, never invent one.

    Feedback students store config.json/config_sha256 and state.arm=feedback_Sn;
    preservation stores identity.json/identity_sha256 and state.arm=Pn. Both
    bind the same immutable full base through the sealed common protocol.
    """
    if arm not in DEPLOYMENT_ARMS:
        raise ValueError('Unsupported native research deployment arm')
    checkpoint_arm = 'feedback_'+arm if arm.startswith('S') else arm
    if state['arm'] != checkpoint_arm or state['seed'] != seed or state['phase'] != 'COMPLETE':
        raise ValueError('Wrong/incomplete native research policy')
    run = Path(checkpoint).parent.parent
    if arm.startswith('S'):
        identity = json.loads((run/'config.json').read_text())
        body = dict(identity)
        if (body.pop('sha256') != fingerprint(body) or identity['schema'] != 'native_feedback_student_v1' or
                identity['arm'] != arm or identity['seed'] != seed or qualification or
                identity['common_protocol_sha256'] != protocol['sha256'] or
                payload['config_sha256'] != identity['sha256'] or state['updates'] != identity['updates']):
            raise ValueError('Wrong/incomplete feedback student configuration')
        return identity, checkpoint_arm, 'config_sha256'
    identity = json.loads((run/'identity.json').read_text())
    body = dict(identity)
    if (body.pop('sha256') != fingerprint(body) or identity['arm'] != arm or identity['seed'] != seed or
            identity['common_protocol_sha256'] != protocol['sha256'] or
            identity['base_model_sha256'] != protocol['base_model_sha256'] or
            payload['identity_sha256'] != identity['sha256']):
        raise ValueError('Wrong preservation policy or changed common base')
    if bool(identity['qualification']) != bool(qualification):
        raise ValueError('Qualification policy cannot become formal deployment evidence')
    return identity, checkpoint_arm, 'identity_sha256'


@torch.no_grad()
def merge_native_adapters_(head, adapters, *, rank=16, alpha=32., expected_layers=None):
    """Exactly source FP32 (B@A)*(alpha/r), only native action attention qkvo.

    B@A is computed on CPU in FP32, matching the original CPU merged export.
    The current base is never wrapped, and a reference network is not created.
    """
    if getattr(head, '_study_merged_adapters', None) is not None:
        raise ValueError('Refuse applying an adapter delta twice')
    if rank != 16 or alpha != 32. or getattr(head, 'action_dim', None) != 3:
        raise ValueError('Locked native3D source attention LoRA rank16/alpha32 required')
    if expected_layers is not None and len(head.blocks) != expected_layers:
        raise ValueError('Wrong native action architecture')
    paths = attention_paths(head)
    expected = {'action_expert.'+path+'.'+suffix for path in paths for suffix in ('lora_A','lora_B')}
    if set(adapters) != expected:
        raise ValueError('Unexpected/missing native attention adapter tensor; no loose loading')
    digest = tensor_digest(adapters)
    for path in paths:
        linear = head.get_submodule(path)
        A = adapters['action_expert.'+path+'.lora_A']
        B = adapters['action_expert.'+path+'.lora_B']
        if type(linear) is not nn.Linear or linear.weight.dtype != torch.float32:
            raise ValueError('Deployment merge requires original plain FP32 action projections')
        if A.shape != (rank, linear.in_features) or B.shape != (linear.out_features, rank):
            raise ValueError('Native adapter shapes do not match original attention projection')
        if A.dtype != torch.float32 or B.dtype != torch.float32 or not (A.isfinite().all() and B.isfinite().all()):
            raise ValueError('Malformed/nonfinite FP32 adapter artifact')
    # Reject malformed payloads before changing any original base weight.
    for path in paths:
        linear = head.get_submodule(path)
        A = adapters['action_expert.'+path+'.lora_A']
        B = adapters['action_expert.'+path+'.lora_B']
        delta = (B.detach().cpu().float() @ A.detach().cpu().float()) * (alpha/rank)
        linear.weight.add_(delta.to(linear.weight.device))
    if any('lora_A' in name or 'lora_B' in name for name in head.state_dict()):
        raise ValueError('Deployment unexpectedly retained LoRA wrappers')
    head.requires_grad_(False).eval()
    head._study_merged_adapters = digest
    return dict(schema='native_plain_action_adapter_merge_v1', adapter_sha256=digest,
                projections=len(paths), tensors=len(adapters), rank=rank, alpha=alpha,
                scaling=alpha/rank, merge_dtype='FP32 CPU B@A, FP32 add to base',
                plain_ActionDiT=True, reference_actor_constructed=False,
                merge_source_sha256=file_sha(__file__))


def resolve_checkpoint(path_or_run):
    path = Path(path_or_run).resolve()
    if (path/'latest.json').exists():
        latest = json.loads((path/'latest.json').read_text())
        path = Path(latest['checkpoint'])
        if file_sha(path/'COMPLETE') != latest['seal_sha256']:
            raise ValueError('Changed research latest seal')
    if path.parent.name != 'checkpoints':
        raise ValueError('Expected existing native research checkpoint directory')
    return path


def export_adapters(checkpoint, output, protocol, *, arm, seed, qualification=False):
    """One offline extraction; final adapter package has no optimizer/dual/data.

    load_state checks the original full training transaction. Its nondeployment
    fields are immediately discarded; inference later consumes this small pure
    package and never follows any archive/teacher/reference path.
    """
    checkpoint = resolve_checkpoint(checkpoint)
    state, payload = load_state(checkpoint, protocol['sha256'], restore_rng=False)
    run_identity, checkpoint_arm, identity_field = checkpoint_training_identity(
        checkpoint,state,payload,protocol,arm=arm,seed=seed,qualification=qualification)
    adapters = {key: value.detach().cpu().clone() for key, value in payload['adapters'].items()}
    metadata = dict(schema='native_preservation_adapter_deployment_v1', protocol_sha256=protocol['sha256'],
        base_model_sha256=protocol['base_model_sha256'], arm=arm, training_seed=seed,
        checkpoint=str(checkpoint), checkpoint_seal_sha256=file_sha(checkpoint/'COMPLETE'),
        checkpoint_updates=state['updates'], checkpoint_rounds=state['round'],
        checkpoint_arm=checkpoint_arm, source_training_identity_field=identity_field,
        scientific_training_identity=run_identity['sha256'], qualification=qualification,
        adapter_sha256=tensor_digest(adapters), rank=16, alpha=32., scaling=2.,
        no_archive_dual_optimizer_reference_or_teacher_in_payload=True)
    del payload, run_identity
    gc.collect()
    output = Path(output).resolve()
    if output.exists():
        _, previous = read_adapter_package(output, protocol, arm=arm, seed=seed, qualification=qualification)
        for key, value in metadata.items():
            if previous[key] != value:
                raise ValueError('Existing deployment package belongs to another snapshot')
        return output
    attempt = output.with_name(output.name+'.incomplete.'+uuid.uuid4().hex)
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_torch(attempt/'adapters.pt', {'adapters': adapters})
    metadata['adapters_file_sha256'] = file_sha(attempt/'adapters.pt')
    metadata['sha256'] = fingerprint(metadata)
    atomic_json(attempt/'deployment.json', metadata)
    atomic_json(attempt/'COMPLETE', dict(metadata_sha256=file_sha(attempt/'deployment.json')))
    os.rename(attempt, output)
    return output


def read_adapter_package(path, protocol, *, arm, seed, qualification=False):
    if arm not in DEPLOYMENT_ARMS:
        raise ValueError('Unsupported native research deployment arm')
    path = Path(path)
    seal = json.loads((path/'COMPLETE').read_text())
    metadata = json.loads((path/'deployment.json').read_text())
    body = dict(metadata)
    if (body.pop('sha256') != fingerprint(body) or seal['metadata_sha256'] != file_sha(path/'deployment.json') or
            metadata['adapters_file_sha256'] != file_sha(path/'adapters.pt') or
            metadata['protocol_sha256'] != protocol['sha256'] or metadata['base_model_sha256'] != protocol['base_model_sha256'] or
            metadata['arm'] != arm or metadata['training_seed'] != seed or metadata['qualification'] != qualification or
            metadata['checkpoint_arm'] != ('feedback_'+arm if arm.startswith('S') else arm) or
            metadata['rank'] != 16 or metadata['alpha'] != 32. or metadata['scaling'] != 2.):
        raise ValueError('Changed/wrong native adapter deployment package')
    value = torch.load(path/'adapters.pt', map_location='cpu', weights_only=True)
    if set(value) != {'adapters'} or tensor_digest(value['adapters']) != metadata['adapter_sha256']:
        raise ValueError('Adapter-only deployment artifact changed or contains training dependencies')
    return value['adapters'], metadata
