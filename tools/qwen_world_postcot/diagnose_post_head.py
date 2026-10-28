"""Read-only held-out audit of the training-only six-metric trajectory head.

The two already-scored requests are fixed IL40k/IL53k outputs. They are NOT
new policy candidates, actor supervision, or a deployment selection mechanism.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint, require_dispatch
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.model.modules.qwen_world_postcot.post_action import PostActionHead, METRIC_NAMES, OFFICIAL_METRIC_NAMES


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, model_wave):
    root.mkdir(parents=True, exist_ok=False)
    wave = Path(model_wave); pin = read(wave/'PINNED.json')['models']['WORLD_IL_060000']
    export = wave/'WORLD_IL_060000/export'; meta = read(export/'export.json')
    if read(export/'COMPLETE')['export_sha256'] != file_sha(export/'export.json'):
        raise ValueError('Unsealed matching IL60k deployment export')
    from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint
    info = inspect_checkpoint(pin['checkpoint'], verify_files='model')
    if info['model_sha256'] != meta['IL_parent']['model_sha256'] or info['seal']['updates'] != 60000:
        raise ValueError('Post head and deployed policy must be the same native IL checkpoint')
    raw = torch.load(Path(pin['checkpoint'])/info['seal']['model_file'], map_location='cpu', weights_only=False, mmap=True)
    # Clone selected storages: a small tensor view must not serialize an entire
    # ZeRO flat parameter storage into this compact head-only diagnostic asset.
    state = {k.removeprefix('post_head.'):v.detach().clone() for k,v in raw['module'].items() if k.startswith('post_head.')}
    head = PostActionHead(**info['identity']['model_config'].get('post_head', {}))
    head.load_state_dict(state, strict=True)
    if not all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Native post-head state must be complete finite FP32')
    torch.save(dict(state_dict=state, config=info['identity']['model_config'].get('post_head', {})), root/'post_head.pt')
    data = Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST')
    tokens = read(data/'manifest.json')['tokens']
    banks = [Path('/mnt/project/qwen-sourceflow-native-20261018/selected_WORLD40_RL77_NAVTEST_20261027T0940Z/WORLD_IL040000/evaluation'),
             Path('/mnt/project/qwen-world-postcot-20261025/additional_IL_RL_NAVTEST_20261028T013728Z/WORLD_IL_053000/evaluation')]
    ids = [read(p/'identity.json') for p in banks]
    def labels(token):
        rows = [read(p/'scenes'/(token+'.json')) for p in banks]
        for row, identity in zip(rows, ids):
            from tools.qwen_world_postcot.evaluate import validate_scene_row
            validate_scene_row(row, token=token, identity_sha256=identity['sha256'],
                protocol_sha256=identity['protocol']['sha256'], expected_log=row['log'])
        return [[row['feedback']['metrics'][n] for n in OFFICIAL_METRIC_NAMES] for row in rows]
    # Both source populations were previously fully sealed. Revalidate payloads
    # before selecting the diagnostic; no post-head predictions affect selection.
    with ThreadPoolExecutor(max_workers=4) as pool:
        feedback = dict(zip(tokens, pool.map(labels, tokens)))
    order = lambda salt: sorted(tokens, key=lambda t: hashlib.sha256((salt+':'+t).encode()).digest())
    natural = order('post-head-natural-v1')[:2048]; selected = set(natural); targeted = set()
    for j in range(3):
        boundary = [t for t in order('post-head-boundary-v1-'+str(j))
                    if feedback[t][0][j] != feedback[t][1][j] and t not in selected]
        targeted.update(boundary[:96])
    chosen = natural + sorted(targeted)
    registration = dict(schema='world_post_head_diagnostic_v1', checkpoint=pin['checkpoint'], updates=60000,
        native_model_sha256=info['model_sha256'], export=str(export), model_sha256=meta['model_sha256'],
        post_head_sha256=file_sha(root/'post_head.pt'), data=str(data), data_sha256=file_sha(data/'manifest.json'),
        banks=[str(p) for p in banks], bank_identity_sha256=[x['sha256'] for x in ids],
        tokens=chosen, natural_tokens=natural, targeted_tokens=sorted(targeted), world=4,
        fixed_candidates_per_scene=2, new_policy_samples=0, new_score_calls=0,
        actor_update=False, natural_sampling='hash-order uniform selection independent of labels/predictions',
        targeted_sampling='up to96 extra NC/DAC/TTC old-policy boundary differences each; report separately',
        caveat='IL60k head on fixed IL40k/IL53k trajectories; not accuracy on newly generated IL60k requests; Test historically observed')
    registration['sha256'] = fingerprint(registration)
    atomic_json(root/'registration.json', registration)
    print(dict(status='PREPARED', natural=len(natural), targeted=len(targeted), scenes=len(chosen)), flush=True)


def predict(root, rank):
    host = require_dispatch(); reg = read(root/'registration.json'); body = dict(reg)
    if body.pop('sha256') != fingerprint(body) or file_sha(root/'post_head.pt') != reg['post_head_sha256']:
        raise ValueError('Changed diagnostic registration/head')
    if not 0 <= rank < reg['world']:
        raise ValueError('Unregistered rank')
    from starVLA.rl.qwen_world_postcot.data import WorldSceneData
    from starVLA.rl.qwen_world_postcot.deployment import load_export
    from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    started = time.monotonic(); model, meta = load_export(reg['export'], 'cuda')
    if meta['model_sha256'] != reg['model_sha256'] or meta['rl_lora']:
        raise ValueError('Wrong/unexpected actor')
    payload = torch.load(root/'post_head.pt', map_location='cpu', weights_only=True)
    head = PostActionHead(**payload['config']); head.load_state_dict(payload['state_dict'], strict=True)
    head = head.cuda().eval().requires_grad_(False)
    processor = QwenObservationProcessor(str(Path(reg['export'])/'processor'))
    data = WorldSceneData(reg['data'], role='NAVTEST')
    if data.sha != reg['data_sha256']:
        raise ValueError('Changed current-observation data')
    tokens = reg['tokens'][rank::reg['world']]; output = root/'scenes'; output.mkdir(exist_ok=True)
    record = dict(status='RUNNING', rank=rank, host=host, requested=len(tokens), new=0,
                  model_load_seconds=time.monotonic()-started, forward_seconds=0., qwen_calls=0,
                  actor_replay_calls=0, optimizer_updates=0, scorer_calls=0)
    with torch.inference_mode():
        for token in tokens:
            dest = output/(token+'.json')
            if dest.exists():
                raise FileExistsError('No implicit attempt overwrite/reuse')
            observation = data.observation(token); inputs = processor([observation])
            ego = torch.tensor([observation['proprio']], dtype=torch.float32, device='cuda')
            rows = [read(Path(p)/'scenes'/(token+'.json')) for p in reg['banks']]
            # Actor receives only legal current observation and exact planned
            # coordinates; labels are extracted AFTER predictions below.
            trajectories = torch.tensor([r['requested_trajectory'] for r in rows], dtype=torch.float32, device='cuda')
            tick = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                memory = model.encode_condition(inputs, ego); context = model.actor.prepare_context(memory)
                hidden = model.actor.forward_features(ActionCodec.norm_odo(trajectories),
                    torch.zeros(2, dtype=torch.float32, device='cuda'),
                    context.index_select(torch.zeros(2, dtype=torch.long, device='cuda')))
                logits = head(hidden).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite diagnostic prediction')
            torch.cuda.synchronize(); record['forward_seconds'] += time.monotonic()-tick
            labels = [[r['feedback']['metrics'][n] for n in OFFICIAL_METRIC_NAMES] for r in rows]
            scores = [r['feedback']['metrics']['score'] for r in rows]
            row = dict(token=token, log=data.records[token]['log'], registration_sha256=reg['sha256'],
                logits=logits.cpu().tolist(), labels=labels, canonical_scores=scores,
                group='natural' if token in set(reg['natural_tokens']) else 'targeted',
                source_payload_sha256=[r['sha256'] for r in rows])
            row['sha256'] = fingerprint(row); atomic_json(dest, row)
            record['new'] += 1; record['qwen_calls'] += 1; record['actor_replay_calls'] += 1
            record['wall_seconds'] = time.monotonic()-started
            if record['new'] % 16 == 0:
                atomic_json(root/f'rank{rank}.json', record)
        record.update(status='COMPLETE', wall_seconds=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(root/f'rank{rank}.json', record)


def metric_summary(logits, target):
    from sklearn.metrics import roc_auc_score, average_precision_score
    probability = 1/(1+np.exp(-np.clip(logits, -80, 80)))
    result = dict(n=int(target.size), MAE=float(np.mean(np.abs(probability-target))),
                  RMSE=float(np.mean((probability-target)**2)**.5),
                  mean_prediction=float(probability.mean()), mean_target=float(target.mean()),
                  soft_label_count=int(((target != 0)&(target != 1)).sum()))
    clipped = np.clip(target, 1e-15, 1-1e-15)
    floor = -(target*np.log(clipped)+(1-target)*np.log1p(-clipped))
    bce = np.logaddexp(0,logits)-target*logits
    result.update(BCE=float(bce.mean()), perfect_soft_target_BCE_floor=float(floor.mean()),
                  excess_BCE=float((bce-floor).mean()), bias=float((probability-target).mean()),
                  R2=float(1-np.sum((probability-target)**2)/np.sum((target-target.mean())**2)) if np.var(target)>0 else None)
    hard = (target == 0)|(target == 1); y = 1-target[hard]; risk = 1-probability[hard]
    fail = y == 1; pred = risk >= .5
    if hard.any():
        tp = int((pred&fail).sum()); fp = int((pred&~fail).sum()); positives=int(fail.sum())
        result['hard_binary'] = dict(n=int(hard.sum()), failed=positives, passed=int((~fail).sum()),
            Brier=float(np.mean((risk-y)**2)), accuracy=float((pred==fail).mean()),
            always_pass_accuracy=float((~fail).mean()),
            violation_precision_at_half=tp/(tp+fp) if tp+fp else None,
            violation_recall_at_half=tp/positives if positives else None,
            true_positive=tp, false_positive=fp,
            AUROC=float(roc_auc_score(y,risk)) if np.unique(y).size==2 else None,
            AUPRC=float(average_precision_score(y,risk)) if np.unique(y).size==2 else None)
    return result


def summarize(root):
    reg=read(root/'registration.json'); rows=[]
    for token in reg['tokens']:
        row=read(root/'scenes'/(token+'.json')); body=dict(row); seal=body.pop('sha256')
        if seal!=fingerprint(body) or row['registration_sha256']!=reg['sha256'] or row['token']!=token:
            raise ValueError('Incomplete/changed diagnostic scene')
        rows.append(row)
    if {p.stem for p in (root/'scenes').glob('*.json')}!=set(reg['tokens']):
        raise ValueError('Duplicate/unexpected diagnostic population')
    results={}
    for group in ['natural','targeted']:
        subset=[r for r in rows if r['group']==group]
        z=np.asarray([r['logits'] for r in subset], dtype=np.float64)
        y=np.asarray([r['labels'] for r in subset], dtype=np.float64)
        p=1/(1+np.exp(-np.clip(z,-80,80))); true=np.asarray([r['canonical_scores'] for r in subset])
        assembled=y[:,:,0]*y[:,:,1]*(5*y[:,:,3]+5*y[:,:,2]+2*y[:,:,4])/12
        if not np.allclose(assembled,true,atol=1e-14,rtol=0):
            raise ValueError('Scorer aggregation differs from original canonical readout')
        pred=p[:,:,0]*p[:,:,1]*(5*p[:,:,3]+5*p[:,:,2]+2*p[:,:,4])/12
        pair={}
        for j,name in enumerate(METRIC_NAMES):
            dy=y[:,1,j]-y[:,0,j]; dz=z[:,1,j]-z[:,0,j]; valid=np.abs(dy)>1e-8
            pair[name]=dict(unequal_pairs=int(valid.sum()), correct_order=float((dy[valid]*dz[valid]>0).mean()) if valid.any() else None)
        results[group]=dict(scenes=len(subset),
            by_candidate={str(i):{name:metric_summary(z[:,i,j],y[:,i,j]) for j,name in enumerate(METRIC_NAMES)} for i in range(2)},
            canonical_PDMS_composite=dict(MAE=float(np.abs(pred-true).mean()), RMSE=float(((pred-true)**2).mean()**.5),
                prediction_mean=float(pred.mean()), true_mean=float(true.mean()),
                note='Product of predicted factor values; not an exact expected joint PDMS.'),
            pair_order=pair)
    result=dict(status='COMPLETE', registration_sha256=reg['sha256'], updates=reg['updates'],
                source_checkpoint=reg['checkpoint'], caveat=reg['caveat'], groups=results,
                logical_scene_count=len(rows), fixed_candidate_count=2*len(rows),
                new_policy_rollouts=0, new_score_calls=0, optimizer_updates=0,
                worker_costs=[read(root/f'rank{i}.json') for i in range(reg['world'])])
    atomic_json(root/'analysis.json', result)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--model-wave')
    p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--rank',type=int)
    a=p.parse_args();root=Path(a.root)
    if a.prepare:prepare(root,a.model_wave)
    elif a.summarize:summarize(root)
    elif a.rank is not None:predict(root,a.rank)
    else:p.error('Choose prepare, summarize or a prediction rank')


if __name__=='__main__':
    main()
