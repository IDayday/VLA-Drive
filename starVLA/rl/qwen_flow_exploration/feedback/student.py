"""Small source-FM student utilities; model/scoring/checkpoint live in common code."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
from starVLA.model.modules.qwen_sourceflow.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.flow_grpo.transactions import atomic_json


class EvaluationStopped(RuntimeError):
    """Owned stop request after an atomic scene; incomplete is never COMPLETE."""


def independent_fm_draw(seed, exposure, kind):
    generator = torch.Generator().manual_seed(seed_for('feedback_source_FM_'+kind, seed, str(exposure)))
    noise = torch.randn((1, 8, 3), generator=generator)
    u = torch.rand((1,), generator=generator)
    schedule = WanContinuousFlowMatchScheduler(1000, shift=5.)
    times = schedule._phi(u, schedule.shift)*schedule.num_train_timesteps
    return noise, times


def adapter_digest(actor):
    digest = hashlib.sha256()
    for name, value in sorted(adapter_state(actor).items()):
        digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(env):
    digest = hashlib.sha256()
    for prefix, module in [('model', env.model), ('reference', env.actor.ref_action_expert)]:
        for name, value in list(module.named_parameters())+list(module.named_buffers()):
            if value.requires_grad:
                continue
            digest.update((prefix+'/'+name).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_teacher_bank(path, protocol, *, expected_method):
    root = Path(path)
    records, hashes = {}, {}
    for token in protocol['splits']['TRAIN']:
        source = root/'scenes'/f'{token}.json'
        value = json.loads(source.read_text())
        body = dict(value)
        if body.pop('sha256') != fingerprint(body) or value['status'] != 'COMPLETE' or value['split'] != 'TRAIN':
            raise ValueError('Unsealed/non-TRAIN teacher data')
        if value['scene_token'] != token or value['result']['method'] not in expected_method:
            raise ValueError('Wrong fixed teacher method/population')
        if value['common_protocol_sha256'] != protocol['sha256'] or value['base_model_sha256'] != protocol['base_model_sha256']:
            raise ValueError('Teacher bank belongs to a different common base/protocol')
        rows = value['teacher_candidates']
        if any(row['split'] != 'TRAIN' or row.get('on_policy_logprob') is not None or row['base_checkpoint_id']!=protocol['base_model_sha256'] for row in rows):
            raise ValueError('External teacher has invalid split/probability identity')
        records[token] = rows
        hashes[token] = file_sha(source)
    return records, fingerprint(hashes)


def choose_teacher(candidates, seed, exposure):
    if not candidates:
        return None
    scores = torch.tensor([row['raw_reward'] for row in candidates], dtype=torch.float32)
    probabilities = torch.softmax(scores/.1, 0)
    generator = torch.Generator().manual_seed(seed_for('feedback_teacher_choice', seed, str(exposure)))
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    return candidates[index]


def teacher_normalized_target(row, codec):
    physical = torch.tensor(row['planned_trajectory'], dtype=torch.float32)
    normalized = (torch.tensor(row['normalized_action'], dtype=torch.float32)
                  if 'normalized_action' in row else codec.norm_odo(physical))
    if normalized.shape != (8, 3) or not torch.isfinite(normalized).all():
        raise ValueError('Wrong native teacher labels')
    if float((codec.denorm_odo(normalized)-physical).abs().max()) > 1e-5:
        raise ValueError('Teacher labels do not encode the scored requested trajectory')
    return normalized


@torch.no_grad()
def evaluate_student(env, tokens, split, *, seeds, namespace='native_qwen_sourceflow_ODE', output=None,
                     stop_requested=None, qualification=False):
    rows = []
    root = Path(output) if output is not None else None
    identity = dict(common_protocol_sha256=env.protocol['sha256'],protocol_sha256=env.protocol['sha256'],
                    base_model_sha256=env.base['model_sha256'],qualification=qualification,
                    behavior=env.behavior_identity,tokens_sha256=fingerprint(tokens),split=split,seeds=list(seeds),namespace=namespace,
                    scenes=len(tokens),evaluation_seeds=list(seeds),
                    readout='single_candidate_source_shift5_ODE_K10',
                    inference='single_candidate_source_shift5_ODE_K10',model_batch=1,
                    scorer_config_hash=env.scorer_config_hash,sampler_config_hash=fingerprint(env.protocol['samplers']),
                    precision=env.protocol['precision'],
                    evaluation_role='primary_noise_compatible_helper' if namespace=='native_qwen_sourceflow_ODE' and not qualification else 'auxiliary',
                    paired_compatible_with_primary=namespace=='native_qwen_sourceflow_ODE' and not qualification)
    identity['sha256'] = fingerprint(identity)
    if root is not None:
        root.mkdir(parents=True,exist_ok=True)
        if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('Changed student evaluation identity')
        atomic_json(root/'identity.json',identity)
    reused = 0
    for token in tokens:
        if stop_requested is not None and stop_requested():
            raise EvaluationStopped('Stop after the previous committed evaluation scene')
        destination = root/'scenes'/f'{token}.json' if root is not None else None
        if destination is not None and destination.exists():
            saved=json.loads(destination.read_text());body=dict(saved)
            if body.pop('sha256')!=fingerprint(body) or saved['identity']!=identity['sha256'] or saved['status']!='COMPLETE' or saved['scene_token']!=token:
                raise ValueError('Invalid saved student evaluation scene')
            rows.extend(saved['records']);reused+=1;continue
        condition = env.condition(token, split)
        actions = []
        for seed in seeds:
            root_noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for(namespace,seed,token)))
            actions.append(env.sample_ode(condition,root_noise))
        actions=torch.cat(actions)
        records=env.score_actions(token, actions, split, proposal_source='single_student_ODE', seed=list(seeds))
        rows.extend(records)
        if destination is not None:
            saved=dict(status='COMPLETE',identity=identity['sha256'],scene_token=token,records=records)
            saved['sha256']=fingerprint(saved);destination.parent.mkdir(exist_ok=True);atomic_json(destination,saved)
    metrics = sorted(rows[0]['native_subscores'])
    return dict(scenes=len(tokens), inference_seeds=list(seeds), single_candidate_each_seed=True,reused_scenes=reused,
                mean={key: float(np.mean([row['native_subscores'][key] for row in rows])) for key in metrics},
                records=rows)
