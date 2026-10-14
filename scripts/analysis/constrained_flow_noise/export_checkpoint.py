"""Export a saved ZeRO-2 actor through the existing original-ODE writer.

This read-only checkpoint consumer does not start a trainer, reconstruct Adam,
or change the registered training/evaluation observation schedule.
"""
import argparse
import json
from pathlib import Path

import torch

from starVLA.rl.constrained_flow_noise.checkpoint import export_policy
from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy


def export_checkpoint(checkpoint, output):
    require_dispatch()
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / 'COMPLETE').is_file():
        raise ValueError('Checkpoint has no atomic COMPLETE marker')
    seal = json.loads((checkpoint / 'files.json').read_text())
    for name in ('identity.json', 'boundary.json'):
        if file_sha(checkpoint / name) != seal[name]:
            raise ValueError('Checkpoint metadata changed: ' + name)
    identity = json.loads((checkpoint / 'identity.json').read_text())
    boundary = json.loads((checkpoint / 'boundary.json').read_text())
    if boundary['phase'] not in ('ready', 'complete') or boundary['next_inner'] != 0:
        raise ValueError('Export requires a completed behavior-update boundary')
    if boundary['world'] != 8 or identity['qualification_only']:
        raise ValueError('Expected a formal eight-rank checkpoint')
    for name, digest in identity['source_files'].items():
        if file_sha(name) != digest:
            raise ValueError('Registered source changed: ' + name)
    # ZeRO-2 keeps the complete BF16 actor module on rank zero. Deployment uses
    # these same parameters, not a separately rounded reconstruction of Adam's
    # FP32 master weights. Optimizer partitions are not needed for inference.
    models = sorted(checkpoint.glob('pytorch_model/*model_states.pt'))
    if len(models) != 1:
        raise ValueError('Expected exactly one complete ZeRO-2 model state')
    model = models[0]
    relative = str(model.relative_to(checkpoint))
    if file_sha(model) != seal[relative]:
        raise ValueError('Saved actor checksum changed')
    value = torch.load(model, map_location='cpu', mmap=True, weights_only=False)
    module = value['module']
    if not module or any(not torch.isfinite(t).all() for t in module.values()):
        raise ValueError('Empty or non-finite saved actor')
    kernel = identity['kernel']
    actor, config, _ = make_noise_policy(
        identity['variant'], (kernel['min_std'], kernel['max_std']), device='cpu')
    actor.load_state_dict(module, strict=True)
    if any(not torch.equal(t.detach().cpu(), module[name])
           for name, t in actor.state_dict().items()):
        raise ValueError('Saved actor changed during CPU loading')
    export_identity = dict(
        training_identity=identity, checkpoint=str(checkpoint), boundary=boundary,
        checkpoint_files_sha256=file_sha(checkpoint / 'files.json'),
        model_states_sha256=seal[relative],
        export_source_sha256=file_sha(__file__),
        module_exact_after_load=True,
        validation_scope='sealed metadata and actor; optimizer not loaded',
        does_not_change_registered_checkpoint_selection=True,
    )
    result = export_policy(actor, config['sft_checkpoint'], output, export_identity)
    print(json.dumps(dict(status='COMPLETE', export=str(result), boundary=boundary)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
