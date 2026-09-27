"""Recompute current predictions/graphs from immutable post-Qwen features and a strict head override."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from starVLA.model.modules.joint_world.current_refinement import restore_current_head
from starVLA.model.modules.joint_world.local_cache import build_payload, load_payload, signature
from starVLA.model.modules.joint_world.public_baseline import sha256
from starVLA.model.modules.structured_world.rehab import ReferenceAgentHeads
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle, observation_from_files
from tools.local_interaction_mask_v2.train_foundation import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'perception-checkpoint', 'dataset', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    rank, count = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if count > 1: dist.init_process_group('gloo')
    source, output = Path(args.cache), Path(args.output)
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['failed']:
        raise ValueError('Incomplete source current cache')
    identity = dict(manifest['identity'])
    if identity.get('perception_override') is not None:
        raise ValueError('Recompute from original frozen features, not a previous override')
    saved = torch.load(args.perception_checkpoint, map_location='cpu', weights_only=False)
    state = saved['head']
    head = ReferenceAgentHeads(state['shared.1.weight'].shape[1], slots=len(state['references']),
        classes=state['classifier.weight'].shape[0]-1, steps=state['motion.weight'].shape[0]//2,
        dim=state['shared.1.weight'].shape[0]).cuda()
    identity['perception_override'] = restore_current_head(head, args.perception_checkpoint,
        identity['foundation_sha256'], identity['public_origin'], identity.get('language_numerics'))
    head.eval().requires_grad_(False)
    spec = json.loads(Path(args.dataset).read_text())
    tokens = json.loads(Path(spec['tokens']).read_text())
    if tokens != [record['token'] for record in manifest['records']]:
        raise ValueError('Current dataset order/token set differs')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record in manifest['records'][rank::count]:
        if (output / 'STOP_REQUESTED').exists(): break
        token = record['token']; payload = load_payload(source, record, manifest)
        metadata = (json.loads((Path(spec['current_records']) / (token + '.json')).read_text())
                    if 'current_records' in spec else current_metadata_from_training_pickle(Path(spec['meta_root']) / (token + '.pkl'), token))
        observation, _ = observation_from_files(Path(spec['observations']) / (token + '.npz'), metadata, verify_transform=True)
        if observation.fingerprint() != payload['observation_identity']:
            raise ValueError('Current observation changed during head refinement')
        native = payload['native_actions'].cuda()
        agents = payload['full_current']['actor_features'][:, 1:].cuda().bfloat16()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = head(agents)
        prediction['agent_features'] = agents
        prediction['scene_features'] = payload['full_current']['context'][:, native.shape[1]:].cuda().bfloat16()
        refreshed = build_payload(native, prediction, observation, identity)
        for name in ('actor_features', 'context'):
            if not torch.equal(refreshed['full_current'][name], payload['full_current'][name]):
                raise AssertionError('Refinement changed upstream current features')
        destination = output / (token + '.pt')
        if destination.exists(): raise FileExistsError('Use a fresh refinement cache')
        torch.save(refreshed, destination)
        records.append({'token': token, 'sha256': sha256(destination)})
        if len(records) % 100 == 0: print(json.dumps({'rank': rank, 'completed': len(records)}), flush=True)
    atomic_json(output / f'shard_{rank}.json', {'records': records, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
    if count > 1: dist.barrier()
    if rank == 0:
        records = sum([json.loads((output / f'shard_{i}.json').read_text())['records'] for i in range(count)], [])
        order = {token: i for i, token in enumerate(tokens)}; records.sort(key=lambda record: order[record['token']])
        complete = len(records) == len(tokens)
        atomic_json(output / 'manifest.json', {'identity': identity, 'identity_sha256': signature(identity), 'records': records,
            'expected': len(tokens), 'failed': 0, 'complete': complete, 'frozen_feature_source_manifest_sha256': sha256(source / 'manifest.json')})
        atomic_json(output / 'status.json', {'status': 'complete' if complete else 'paused', 'completed': len(records), 'expected': len(tokens)})
    if count > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
