"""Archive exact Git blobs for research implementations; never import or execute them.

Identical bytes are stored once. The version manifest preserves original paths,
commit and blob identities, so code reuse is visible without merging old branches.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OUT = ROOT / 'implementation'
VERSIONS = [
    ('C', 'd1d40854299b9599b2accc382bcfc4b676dd7623', None, 'Full current/future DINO + GT-MAE; C0-C5 target/W matrix'),
    ('S', '1493deda247107efe076b9294863a6753391298a', 'C', 'Eight-frame / video targets, GT action auxiliary condition, S4 direct W planner'),
    ('A', 'f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882', 'S', 'A_ACTION / A_NO_MAE implementation and memory/query interfaces'),
    ('V_QUERY', '7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c', 'A', 'Actual V_QUERY formal source; keep interface/runtime changes explicit'),
    ('R1_S0', 'ee2f086d7fcb94a6ee442ff8747e378d8be016e6', 'S', 'Optimized-label route, S0 original FM'),
    ('R1_R_S3', '32de37db070f0e4df2ce66f25e850bed2561828e', 'R1_S0', 'Optimized-label Action-Only/S3 original FM; original logged auxiliary condition'),
    ('OBJECTIVE_AXES', '7e406a122fd662830327a0050241db728576cbb7', 'R1_R_S3', '512-update mature pilots: normalized axes and rollout'),
    ('OBJECTIVE_GAIN', '1da528e0dee0a1f81bc38b06524905b52fff724f', 'OBJECTIVE_AXES', 'Absolute x and heading FM gain pilots'),
    ('OBJECTIVE_ENDPOINT', 'b3cefdd994146526cf70f259010fce5077fc5692', 'OBJECTIVE_GAIN', 'Endpoint kinematics/braking pilots and four generic-start formal runs'),
    ('OBJECTIVE_V2', '6ec45b0e8705c7919fb555c20324f91580add976', 'OBJECTIVE_ENDPOINT', 'Curriculum/normalization/weak rollout, actual optimizer groups; rejected cold-start recipe'),
    ('FM_STEPS', '771c802021e24d7ad3c3ff4f77c6de3824669edd', 'C', 'Fixed-C1 inference Euler-step sweep, not another training objective'),
    ('G_NAVSIM', '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d', 'S', 'Single-frame geometry, time-conditioned Bt, genuine detached proposal, FGTR residual; original FM'),
    ('G_NUSCENES', '41da7c198cf0ac4f799aee2a27a0a8126de30230', 'G_NAVSIM', 'Actual six-camera/six-time formal recipe and dataset class statistics'),
    ('G_P0_WINDOW_SOURCE', '1e5b109', 'G_NAVSIM', 'Actual engineering_fast performance-window model source; not adopted formal execution'),
    ('G_HOST_FAST', '30f70d63556342382367aaff25ddce4342ff256c', 'G_P0_WINDOW_SOURCE', 'Conservative host-only/visual-cache/world-scaling candidates; not fully accepted'),
]
EXACT = {
    'starVLA/model/framework/DDPForesight.py', 'starVLA/model/framework/QwenOFT.py',
    'starVLA/model/framework/ddp_full_foresight.py', 'starVLA/model/framework/ddp_action_video_foresight.py',
    'starVLA/model/framework/vla_structured_fgtr.py',
    'starVLA/model/modules/action_model/GR00T_ActionHeader.py',
    'starVLA/model/modules/vehicle_joint/initialization.py',
    'starVLA/dataloader/foresight_dataset.py', 'starVLA/model/modules/vlm/QWen3.py',
    'tools/foresight/train_student.py', 'tools/foresight/checkpoints.py', 'tools/foresight/student_state.py',
    'tools/foresight/export_predictions.py', 'tools/foresight/deployment_precision.py',
    'tools/foresight/score_pdms.py', 'tools/foresight/prepare_vehicle_trajectories.py',
    'tools/foresight/train_trajectory_mae.py', 'tools/foresight/export_interaction_targets.py',
    'tools/full_foresight/canonical_scores.py', 'tools/full_foresight/register_priority.py',
    'tools/full_foresight/cache_dino_targets.py',
    'tools/action_video_foresight/run_experiments.py', 'tools/action_video_foresight/run_formal_campaign.py',
    'tools/action_video_foresight/train_frozen_W_probe.py', 'tools/action_video_foresight/build_clip_index.py',
    'tools/action_video_foresight/cache_frame_sequence_targets.py', 'tools/action_video_foresight/cache_video_targets.py',
    'tools/planning_interface_transfer/run_matched_experiments.py',
    'tools/planning_interface_transfer/audit_gradients_and_conditioning.py',
    'tools/s3_optimized_training/labels.py', 'tools/s3_optimized_training/run.py',
}
PREFIXES = ('starVLA/model/modules/foresight/', 'starVLA/model/modules/action_model/flow_matching_head/',
            'starVLA/model/modules/structured_world/', 'starVLA/dataloader/structured_world/',
            'configs/foresight_resolution/', 'configs/action_video_foresight/', 'configs/trajectory_objective/',
            'configs/structured_world/', 'tools/trajectory_objective/')
G_TOOLS = {'train_vla', 'train_geometry', 'build_cache', 'prepare_nuscenes_labels', 'prepare_nuscenes_manifest',
           'build_nuscenes_cache', 'build_navsim_dev_labels', 'build_navsim_current_inputs',
           'infer_checkpoint', 'score_navsim', 'score_nuscenes', 'score_scene_fields',
           'prefetch', 'resume_origin', 'recovery_boundary', 'calibrate_objectives', 'training_assets',
           'stage_timing', 'benchmark_engineering_execution', 'validate_engineering_execution',
           'run_world_scaling_node'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def main():
    assert not OUT.exists(), 'Use a fresh implementation archive'
    (OUT / 'versions').mkdir(parents=True)
    (OUT / 'patches').mkdir()
    seen, manifests, versions = {}, {}, []
    for name, rev, parent, purpose in VERSIONS:
        commit = git('rev-parse', rev + '^{commit}').decode().strip()
        records = git('ls-tree', '-r', commit).decode().splitlines()
        entries = []
        for record in records:
            meta, path = record.split('\t', 1)
            mode, kind, blob = meta.split()
            selected = path in EXACT or path.startswith(PREFIXES)
            selected |= path.startswith('tools/structured_world/') and Path(path).stem in G_TOOLS
            selected |= name == 'FM_STEPS' and 'fm_step' in path and path.endswith('.py')
            if not selected or kind != 'blob' or Path(path).suffix not in ('.py', '.json', '.yaml'):
                continue
            raw = git('cat-file', 'blob', blob)
            sha = hashlib.sha256(raw).hexdigest()
            if sha not in seen:
                dest = Path('snapshots') / name / path
                q = OUT / dest
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(raw)
                seen[sha] = str(dest)
            symbols = []
            if path.endswith('.py'):
                try:
                    tree = ast.parse(raw)
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            symbols.append({'name': node.name, 'line': node.lineno})
                            if isinstance(node, ast.ClassDef):
                                symbols.extend({'name': node.name + '.' + n.name, 'line': n.lineno}
                                               for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                except SyntaxError:
                    raise ValueError('Cannot parse exact archived Python source: ' + path)
            entries.append({'path': path, 'git_blob': blob, 'sha256': sha, 'bytes': len(raw),
                            'archive_path': seen[sha], 'symbols': symbols})
        previous = manifests.get(parent)
        diff_info = None
        if previous:
            paths = sorted({e['path'] for e in entries} | {e['path'] for e in previous['files']})
            patch = git('diff', '--no-ext-diff', '--no-renames', '--full-index', previous['commit'], commit, '--', *paths)
            dest = Path('patches') / (parent + '_to_' + name + '.patch')
            (OUT / dest).write_bytes(patch)
            diff_info = {'comparison_base': parent, 'base_commit': previous['commit'],
                         'path': str(dest), 'sha256': hashlib.sha256(patch).hexdigest(),
                         'meaning': 'Scoped implementation difference; NOT a claim of direct Git ancestry or a single-factor scientific ablation'}
        manifest = {'version': name, 'commit': commit, 'purpose': purpose,
                    'commit_time': git('show', '-s', '--format=%cI', commit).decode().strip(),
                    'files': entries, 'comparison_patch': diff_info,
                    'scope': 'Verbatim selected algorithm/training/data/evaluation files, not a standalone runnable dependency bundle. Use full immutable commit with original registered assets/environment to reproduce.'}
        manifests[name] = manifest
        (OUT / 'versions' / (name + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')
        versions.append({'version': name, 'commit': commit, 'purpose': purpose, 'files': len(entries),
                         'manifest': 'versions/' + name + '.json', 'patch': str(diff_info['path']) if diff_info else None})
        lines = [f'# {name}: implementation at {commit}', '', purpose, '',
                 'This is the exact selected source, including dormant interfaces where present. Formal activation is determined by the archived run configuration, not file existence.', '',
                 f'[Complete immutable source](https://github.com/IDayday/VLA-Drive/tree/{commit}) · [Machine manifest]({name}.json)', '']
        if diff_info:
            lines += [f"[Scoped diff against {parent}](../{diff_info['path']}) (comparison edge, not asserted Git parent).", '']
        lines += ['| Original file | Exact archived source | SHA256 |', '|---|---|---|']
        for e in entries:
            lines.append(f"| `{e['path']}` | [source](../{e['archive_path']}) | `{e['sha256'][:16]}` |")
        (OUT / 'versions' / (name + '.md')).write_text('\n'.join(lines)+'\n')
    (OUT / 'INDEX.json').write_text(json.dumps({'versions': versions, 'unique_files': len(seen),
                                             'source_bytes': sum((OUT/p).stat().st_size for p in seen.values()),
                                             'unchanged_file_reuse': 'Same SHA256 bytes stored once; every version maps original paths explicitly'}, indent=2)+'\n')
    print(json.dumps({'versions': len(versions), 'unique_files': len(seen)}))


if __name__ == '__main__':
    main()
