r"""Bounded offline current-DINO readout extraction from a complete IL checkpoint.

Example (one to sixteen explicit TRAIN or DEV tokens)::

    python -m tools.qwen_world_postcot.extract_student_features \
      --checkpoint /run/checkpoints/update001000 --data /data/TRAIN \
      --cohort TRAIN --teacher-cache /teachers/TRAIN_resolved_v3 \
      --token SCENE_TOKEN --output /analysis/student_features.pt --device cuda:0

Only current-image Qwen encoding and the trained DINO readout run. The teacher
cache supplies identity metadata, never tensors or a teacher forward. The result
is directly accepted by visualize_features.py. This tool never trains a model.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import uuid

import torch

from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT
from starVLA.model.modules.qwen_world_postcot.teacher_targets import CAMERAS, TeacherCacheReader, _read_sealed
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.qwen_sourceflow.export import PROCESSOR_FILES
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint, require_dispatch
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint


SCHEMA = 'world_postcot_student_DINO_features_v1'
FEATURE_SHAPE = (3, 9, 16, 1024)
REPO = Path(__file__).resolve().parents[2]


def validate_tokens(tokens, data, cohort):
    if cohort not in ('TRAIN', 'DEV') or data.manifest['cohort'] != cohort:
        raise ValueError('Student visualization extraction is TRAIN/DEV only')
    tokens = list(tokens)
    if not 1 <= len(tokens) <= 16 or len(set(tokens)) != len(tokens):
        raise ValueError('Provide one to sixteen distinct explicit scene tokens')
    if not set(tokens).issubset(data.tokens):
        raise ValueError('Requested scene is outside the sealed data role')
    return tokens


def load_il_with_readout(checkpoint, device):
    info = inspect_checkpoint(checkpoint, verify_files='model')
    identity = info['identity']
    if (identity.get('schema') != 'world_postcot_full_IL_v1'
            or identity.get('uses_rl') is not False
            or identity.get('uses_metric_supervision') is not True
            or identity.get('train_heads') is not True or info['seal']['updates'] < 1):
        raise ValueError('A complete trained full-framework IL checkpoint with its DINO readout is required')
    # Reuse the checkpoint's complete native model, including every auxiliary
    # head. Strict load rejects absent DINO tensors or unexpected state keys.
    raw = torch.load(Path(info['path'])/info['seal']['model_file'], map_location='cpu', weights_only=False)
    state = raw['module']
    if any(value.dtype != torch.float32 for value in state.values()
           if isinstance(value, torch.Tensor) and value.is_floating_point()):
        raise ValueError('Stored IL parameters and buffers must preserve FP32 values')
    model = QwenWorldPostCoT(None, config=copy.deepcopy(identity['model_config']),
                            checkpoint_config=copy.deepcopy(identity['qwen_config']), train_heads=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, info


def verify_sources(info):
    expected = info['identity']['source_files']
    if not expected:
        raise ValueError('Checkpoint lacks its actual training source identity')
    for name, digest in expected.items():
        target = REPO/name
        if not target.resolve().is_relative_to(REPO) or file_sha(target) != digest:
            raise ValueError('Checkpoint source changed; use its immutable source snapshot: '+name)
    return dict(checkpoint_source_files=expected, extractor_file_sha256=file_sha(__file__))


def processor_binding(info, override=None):
    receipt = info['identity']['initialization']['qwen']['receipt']
    root = Path(override or receipt['path']).resolve()
    files = {}
    for name in PROCESSOR_FILES:
        target = root/name
        if name not in receipt['files'] or file_sha(target) != receipt['files'][name]:
            raise ValueError('Processor/tokenizer differs from the IL initialization contract: '+name)
        files[name] = receipt['files'][name]
    return root, dict(path=str(root), repository=receipt['repository'], revision=receipt['revision'],
                      files=files, sha256=fingerprint(files), current_camera_order=list(CAMERAS))


def teacher_binding(info, data, teacher_root, tokens):
    identity = info['identity']
    trained_root = Path(identity['config']['teacher_cache'])
    if file_sha(trained_root/'IDENTITY.json') != identity['teacher_identity_sha256']:
        raise ValueError('The IL checkpoint training teacher identity changed')
    trained = _read_sealed(trained_root/'IDENTITY.json')
    reader = TeacherCacheReader(teacher_root, expected_cohort=data.manifest['cohort'], max_open_shards=1)
    try:
        target = reader.identity
        if (target['dino'] != trained['dino'] or target['camera_names'] != list(CAMERAS)
                or target['population_sha256'] != data.sha
                or target['records_sha256'] != data.manifest['records_sha256']
                or target['layout']['fields']['dino']['shape'] != list(FEATURE_SHAPE)):
            raise ValueError('Teacher/data/current-DINO spatial contract differs from this IL checkpoint')
        # Validate the token-to-observation join without opening a teacher shard.
        for token in tokens:
            if (token not in reader.records or reader.records[token]['metadata']['source_record_hash']
                    != data.records[token]['source_record_hash']):
                raise ValueError('Teacher target scene/observation identity mismatch: '+token)
        train_manifest_path = Path(identity['config']['training_data'])/'manifest.json'
        if file_sha(train_manifest_path) != identity['data_sha256']:
            raise ValueError('The checkpoint training population changed')
        train_manifest = json.loads(train_manifest_path.read_text())
        if data.manifest['role_partition_sha256'] != train_manifest['role_partition_sha256']:
            raise ValueError('Requested DEV/TRAIN data is from another split partition')
        if data.manifest['cohort'] == 'TRAIN' and data.sha != identity['data_sha256']:
            raise ValueError('Requested TRAIN is not this IL checkpoint population')
        return dict(teacher_identity=target['sha256'], training_teacher_identity=trained['sha256'],
                    teacher_identity_file_sha256=file_sha(Path(teacher_root)/'IDENTITY.json'),
                    teacher_manifest_file_sha256=file_sha(Path(teacher_root)/'MANIFEST.json'),
                    DINO_contract=copy.deepcopy(target['dino']))
    finally:
        reader.close()


def current_identity(data, token):
    row = data.records[token]
    images = {camera: None if row['images'].get(camera) is None
              else file_sha(row['images'][camera]) for camera in CAMERAS}
    value = dict(token=token, source_record_hash=row['source_record_hash'],
                 timestamp=row['timestamp'], current_image_sha256=images,
                 proprio=row['proprio'], navigation=row['navigation'])
    value['sha256'] = fingerprint(value)
    return value


@torch.inference_mode()
def extract_current(model, data, processor, tokens, device):
    """Current-only API: intentionally no target/cache/teacher argument."""
    features, valid, observations, dtypes = {}, {}, {}, {}
    for token in tokens:
        before = current_identity(data, token)
        observation = data.observation(token)
        batch = processor([observation])
        ego = torch.tensor([observation['proprio']], device=device, dtype=torch.float32)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            memory = model.encode_condition(batch, ego)
            prediction = model.dino_readout(memory)
        if prediction.shape != (1, *FEATURE_SHAPE) or not torch.isfinite(prediction).all():
            raise ValueError('Trained DINO readout produced an invalid current feature tensor')
        if current_identity(data, token) != before:
            raise ValueError('Current image/observation changed during extraction: '+token)
        # float() losslessly stores the computed BF16 student values as FP32;
        # it does not claim that the readout was computed with FP32 matrices.
        features[token] = prediction[0].float().cpu().contiguous()
        valid[token] = torch.tensor([observation['images'].get(camera) is not None for camera in CAMERAS],
                                    dtype=torch.bool)[:, None, None].expand(3, 9, 16).clone()
        observations[token], dtypes[token] = before, str(prediction.dtype)
    return dict(features=features, valid=valid, observations=observations,
                readout_compute_dtypes=dtypes)


def publish(output, artifact):
    output = Path(output)
    if output.suffix != '.pt':
        raise ValueError('Student feature output must be a .pt artifact')
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix('.receipt.json')
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or receipt.exists():
            raise FileExistsError('Existing feature artifact/receipt is immutable: '+str(output))
        temp = output.with_name(output.name+'.writing.'+uuid.uuid4().hex)
        try:
            with temp.open('xb') as stream:
                torch.save(artifact, stream)
                stream.flush(); os.fsync(stream.fileno())
            os.link(temp, output)
        finally:
            temp.unlink(missing_ok=True)
        result = dict(schema=SCHEMA, status='COMPLETE', file=str(output.resolve()), sha256=file_sha(output),
                      count=len(artifact['features']), model_sha256=artifact['checkpoint']['model_sha256'],
                      teacher_identity=artifact['teacher_identity'], metadata_sha256=fingerprint(artifact['metadata']))
        result['receipt_sha256'] = fingerprint(result)
        temp_receipt = receipt.with_name(receipt.name+'.writing.'+uuid.uuid4().hex)
        temp_receipt.write_text(json.dumps(result, indent=2, allow_nan=False))
        os.link(temp_receipt, receipt); temp_receipt.unlink()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--cohort', choices=('TRAIN', 'DEV'), required=True)
    parser.add_argument('--teacher-cache', required=True)
    parser.add_argument('--token', action='append', required=True, help='Repeat for at most16 distinct scenes')
    parser.add_argument('--processor', help='Optional relocated byte-identical original processor directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.suffix != '.pt' or output.exists() or output.with_suffix('.receipt.json').exists():
        raise ValueError('Choose a new .pt output; completed or partial artifacts are never overwritten')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Extraction supports qualified CPU/CUDA BF16 autocast only')
    if device.type == 'cuda':
        require_dispatch()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
        raise ValueError('Extraction preserves the IL CUBLAS numerical contract')
    configure_numerics(); torch.set_num_threads(4)
    data = WorldSceneData(args.data, role=args.cohort)
    tokens = validate_tokens(args.token, data, args.cohort)
    model, info = load_il_with_readout(args.checkpoint, device)
    sources = verify_sources(info)
    processor_root, processor_info = processor_binding(info, args.processor)
    teachers = teacher_binding(info, data, args.teacher_cache, tokens)
    extracted = extract_current(model, data, QwenObservationProcessor(processor_root), tokens, device)
    metadata = dict(cohort=args.cohort, tokens=tokens, count=len(tokens), data_manifest_sha256=data.sha,
        data_records_sha256=data.manifest['records_sha256'], role_partition_sha256=data.manifest['role_partition_sha256'],
        source=sources, processor=processor_info, teacher=teachers,
        feature_shape=list(FEATURE_SHAPE), feature_storage_dtype='torch.float32',
        parameter_storage_dtype='torch.float32', matrix_compute='BF16 autocast; native FP32 residuals retained',
        current_observations_only=True, teacher_forward=False, actor_forward=False,
        future_or_GT_or_metric_inputs=False, qualification=info['identity']['qualification'],
        device=str(device), torch=str(torch.__version__), numerics=info['identity']['numerics'])
    artifact = dict(schema=SCHEMA, teacher_identity=teachers['teacher_identity'],
        checkpoint=dict(path=info['path'], model_sha256=info['model_sha256'],
                        identity_sha256=info['seal']['identity_sha256'], updates=info['seal']['updates']),
        metadata=metadata, **extracted)
    print(json.dumps(publish(args.output, artifact), indent=2))


if __name__ == '__main__':
    main()
