"""Bounded real-data audit before training a geometric scorer. No optimizer."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import pickle
import numpy as np
import torch
from PIL import Image
from iqe.contracts import require
from iqe.io import atomic_json, file_hash, digest, read_json
from iqe.specialist_study.config import StudyConfig
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from .calibration import CalibrationReader
from .geometry import project_points, candidate_anchors, neighborhood_indices


class TrustedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def run(source_root, output, count):
    require(1 <= count <= 256, "geometry audit capped at256 legitimate fit scenes")
    source = StudyConfig(**read_json(Path(source_root)/'resolved_config.json'))
    contract = read_json(source.root/'OFFLINE_BASE_CONTRACT.json')
    from iqe.query_base import bind_source
    bind_source(contract['source_root'], contract['source_commit'])
    from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics
    from navsim.visualization.camera import _transform_pcs_to_images
    import navsim.visualization.camera as official_projection
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    # Source identity supplies the raw metadata location, never a guessed path.
    data_identity_path = Path('/mnt/project/ddp-foresight-artifacts/20260928/teacher_data_full_v1/identity.json')
    source_identity = read_json(data_identity_path)
    raw_root = Path(source_identity['raw_log_root'])
    metadata, bank = load_bank(source, 'deploy', 'scorer_fit')
    all_scenes = scenes_by_role(source)['scorer_fit']
    require(metadata['scene_ids'] == [s.scene_id for s in all_scenes], "candidate bank join")
    selected = sorted(range(len(all_scenes)), key=lambda i: digest([source.seed, 'geometry-audit', all_scenes[i].scene_id]))[:count]
    scenes = [all_scenes[i] for i in selected]
    require(all(s.split_role == 'incremental_fit' for s in scenes), "fit only")
    out = Path(output)
    registration = dict(schema='current_calibration_audit_v1', requested=count,
        source_campaign=str(source.root), base_binding_hash=file_hash(source.root/'BASE_BINDING.json'),
        bank_hash=metadata['hash'], scene_ids=[s.scene_id for s in scenes],
        selection='seeded hash before reading candidate quality, not a failure subset',
        camera_models=['navsim_pinhole', 'brown_conrady_diagnostic_only'],
        anchor_heights_metres=[0., 1.], ground_height_measured=False,
        official_projection_file_hash=file_hash(official_projection.__file__),
        qwen_implementation_hash=file_hash(modeling_qwen3_vl.__file__),
        data_identity_hash=file_hash(data_identity_path), raw_root=str(raw_root),
        no_model_updates=True, no_navtest=True, no_label_inputs=True)
    if not (out/'REGISTRATION.json').exists():
        atomic_json(out/'REGISTRATION.json', registration|{'registered_utc': datetime.now(timezone.utc).isoformat()}, immutable=True)
    else:
        old=read_json(out/'REGISTRATION.json'); require(all(old[k] == v for k,v in registration.items()), 'audit identity changed')
    calibration = CalibrationReader(count)
    visual = VerifiedSpatialReader(source, 1)
    logs = OrderedDict(); rows=[]
    for scene, index in zip(scenes, selected):
        if scene.source_log_id not in logs:
            with (raw_root/(scene.source_log_id+'.pkl')).open('rb') as f:
                frames = TrustedNumpyUnpickler(f).load()
            logs[scene.source_log_id] = {f['token']: f for f in frames}
            while len(logs) > 2: logs.popitem(last=False)
        raw = logs[scene.source_log_id][scene.scene_id]
        require(np.allclose(raw['lidar2ego'], np.eye(4), atol=1e-6, rtol=0), 'nonidentity lidar2ego requires an explicit coordinate adapter')
        record = read_json(scene.observation_ref)
        require(raw['timestamp'] == record['timestamp'], 'raw/current timestamp mismatch')
        c = calibration.batch([scene], 'cpu')
        max_k_error=0.;max_e_error=0.;sizes=[]
        for view, (name, path) in enumerate(zip(contract['input']['views'], record['image_paths'])):
            original = raw['cams'][name]
            require(Path(original['data_path']).name == Path(path).name, 'current camera image mismatch')
            with Image.open(path) as im:
                size=im.size
            sizes.append(size)
            k,_ = crop_resize_intrinsics(np.asarray(original['cam_intrinsic']), size, (1024,576))
            e = np.eye(4); e[:3,:3] = original['sensor2lidar_rotation']; e[:3,3] = original['sensor2lidar_translation']
            max_k_error=max(max_k_error,float(np.max(np.abs(k-np.asarray(record['current_calibration']['intrinsics'][view])))))
            max_e_error=max(max_e_error,float(np.max(np.abs(e-np.asarray(record['current_calibration']['extrinsics'][view])))))
        require(max_k_error < 1e-9 and max_e_error < 1e-9, 'source camera calibration mismatch')
        tensors = visual.read(scene)
        grid=tensors['visual_grid'].long()
        require(grid.shape == (3,3) and torch.equal(grid, torch.tensor([[1,36,64]]*3)), 'new actual Qwen layout requires explicit audit')
        require(tensors['visual'].shape[:2] == (3,576) and tensors['visual_valid'].all().item(), 'unverified merged V layout')
        points = candidate_anchors(torch.from_numpy(bank['trajectories'][index:index+1])).double()
        # Double precision and exact saved calibration for independent official comparison.
        k=torch.tensor(record['current_calibration']['intrinsics'], dtype=torch.float64)[None]
        e=torch.tensor(record['current_calibration']['extrinsics'], dtype=torch.float64)[None]
        size=c['camera_image_sizes'].double()
        uv, visible=project_points(points,k,e,size)
        max_projection_error=0.
        for candidate in range(points.shape[1]):
            for view in range(3):
                ref, mask = _transform_pcs_to_images(points[0,candidate].numpy().T,e[0,view,:3,:3].numpy(),
                    e[0,view,:3,3].numpy(),k[0,view].numpy(),img_shape=(576,1024))
                require(np.array_equal(mask,visible[0,candidate,view].numpy()), 'official projection visibility mismatch')
                if mask.any():
                    max_projection_error=max(max_projection_error,float(np.abs(ref[mask]-uv[0,candidate,view].numpy()[mask]).max()))
        require(max_projection_error < 1e-8, 'official pinhole projection mismatch')
        d=torch.tensor(record['current_calibration']['distortion'],dtype=torch.float64)[None]
        distorted,dvalid=project_points(points,k,e,size,camera_model='brown_conrady',distortion=d)
        common=visible & dvalid
        shifts=(uv-distorted).norm(dim=-1)[common].numpy()
        idx, local=neighborhood_indices(uv,visible,size,torch.tensor([[[32,18]]*3]))
        rows.append(dict(scene_id=scene.scene_id,source_log=scene.source_log_id,observation_hash=scene.observation_hash,
            observation_file_hash=file_hash(scene.observation_ref),actual_image_sizes=sizes,
            calibration_match=True,official_projection_max_abs=max_projection_error,
            grid=grid.tolist(),candidate_anchors=int(visible.shape[1]*visible.shape[-1]),
            any_view_visible=int(visible.any(2).sum()),per_view_visible=visible.sum((0,1,3)).tolist(),
            valid_local_tokens=int(local.sum()),
            distortion_changes_visibility=int((visible != dvalid).sum()),
            distortion_common_shift_pixels_quantiles=np.quantile(shifts,[.5,.95,1]).tolist() if len(shifts) else None))
    report=dict(status='GEOMETRY_INTERFACE_AUDITED',registration_hash=file_hash(out/'REGISTRATION.json'),
        scenes=len(rows),source_logs=len({r['source_log'] for r in rows}),rows=rows,
        candidate_anchor_any_view_coverage=sum(r['any_view_visible'] for r in rows)/sum(r['candidate_anchors'] for r in rows),
        official_projection_max_abs=max(r['official_projection_max_abs'] for r in rows),
        all_calibrations_match=True,all_current_timestamps_match=True,all_lidar2ego_identity=True,
        V_order='merged row-major verified in installed Qwen3VL rotary positions and merger;18x32 per view',
        pinhole_model='exact locked NAVSIM visualization convention; not proof of physical lens rectification',
        distortion_metadata_used_by_original_visualization=False,
        geometric_limitations=['no measured road height/pitch', 'projection is not visibility/occlusion truth',
                              'future occupancy cannot be obtained from this current image projection'],
        trainable_updates=0,scientific_effect='UNTESTED')
    atomic_json(out/'REPORT.json',report,immutable=True)
    return {k:v for k,v in report.items() if k != 'rows'}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-campaign',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-scenes',type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(2)
    import json
    print(json.dumps(run(args.source_campaign,args.output,args.max_scenes),sort_keys=True))


if __name__ == '__main__': main()
