"""Finite real-RL qualification evidence and explicit controller-only lineage.

This helper never changes actor/math/optimizer code or creates a PASS from a CPU
smoke alone.  Its acceptance requires two real eight-rank train_rl invocations,
the recorded actual sampling terms, and a native mid-inner continuation.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

if __package__ in (None, ''):
    sys.path[:] = [p for p in sys.path if Path(p or os.curdir).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json

SCHEMA = 'qwen_world_compare_RL_engineering_acceptance_v1'
OPS_SCHEMA = 'qwen_world_compare_controller_migration_v1'
OPS_ONLY = frozenset('tools/qwen_world_compare/'+name for name in
                     ('campaign.py', 'qualify_rl.py', 'qualify.py', 'profile.py')) | frozenset({
                         'starVLA/rl/qwen_world_compare/profiling.py'})
UPDATES = 4
MID_UPDATE = 2


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, body):
    path = Path(path);value = dict(body, sha256=fingerprint(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:raise ValueError('Immutable qualification receipt already differs')
    else:atomic_json(path, value)
    return value


def _sealed(value):
    body = dict(value);sha = body.pop('sha256', None)
    if sha != fingerprint(body):raise ValueError('Qualification/lineage receipt checksum changed')
    return value


def _git(source):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip():
        raise ValueError('Execution lineage requires clean immutable source trees')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()


def _ops_diff(before, after):
    changed = {name:dict(before=before.get(name), after=after.get(name))
               for name in sorted(set(before)|set(after)) if before.get(name) != after.get(name)}
    if not changed or not set(changed).issubset(OPS_ONLY):
        raise ValueError('Only explicit campaign/qualifier/profile orchestration may migrate; scientific dependency changed')
    return changed


def create_execution_migration(training_source, controller_source, evidence, output):
    from tools.qwen_world_compare.campaign import expected_source_files
    training_source, controller_source = Path(training_source).resolve(), Path(controller_source).resolve()
    before, after = expected_source_files(training_source), expected_source_files(controller_source)
    changed = _ops_diff(before, after);proofs = []
    for path in evidence:
        value = read(path)
        if value.get('status') != 'PASS' or value.get('source_files') != after:
            raise ValueError('Controller migration requires passed tests bound to this exact new source union')
        proofs.append(dict(path=str(Path(path).resolve()), sha256=file_sha(path)))
    if not proofs:raise ValueError('No actual controller-equivalence evidence was supplied')
    patches = {}
    for name in changed:
        a, b = training_source/name, controller_source/name
        patches[name] = ''.join(difflib.unified_diff(
            a.read_text().splitlines(True) if a.exists() else [],
            b.read_text().splitlines(True) if b.exists() else [], fromfile='training/'+name, tofile='controller/'+name))
    return publish(output, dict(schema=OPS_SCHEMA, status='PASS',
        training_source=str(training_source), controller_source=str(controller_source),
        training_git_sha=_git(training_source), controller_git_sha=_git(controller_source),
        training_source_files=before, controller_source_files=after, changes=changed,
        patches=patches, patch_sha256=fingerprint(patches), evidence=proofs,
        scientific_dependencies_unchanged=True, original_IL_acceptance_preserved=True,
        training_jobs_execute_training_source=True,
        helper_jobs_execute_controller_source=True))


def validate_execution_migration(path, training_source, controller_source, training_files, controller_files):
    value = _sealed(read(path))
    if (value.get('schema') != OPS_SCHEMA or value.get('status') != 'PASS'
            or value.get('training_source') != str(Path(training_source).resolve())
            or value.get('controller_source') != str(Path(controller_source).resolve())
            or value.get('training_source_files') != training_files
            or value.get('controller_source_files') != controller_files
            or value.get('changes') != _ops_diff(training_files, controller_files)
            or value.get('patch_sha256') != fingerprint(value.get('patches'))
            or value.get('scientific_dependencies_unchanged') is not True
            or value.get('original_IL_acceptance_preserved') is not True
            or value.get('training_jobs_execute_training_source') is not True
            or value.get('helper_jobs_execute_controller_source') is not True):
        raise ValueError('Controller migration is not bound to these exact source identities')
    if not value.get('evidence'):raise ValueError('Controller migration has no equivalence evidence')
    for proof in value['evidence']:
        actual = read(proof['path'])
        if (file_sha(proof['path']) != proof['sha256'] or actual.get('status') != 'PASS'
                or actual.get('source_files') != controller_files):
            raise ValueError('Controller-only migration evidence changed')
    return value


def tree_equal(left, right, path='root'):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        if (not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype
                or not torch.equal(left, right)):raise ValueError('Native RL tensor continuation mismatch: '+path)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):raise ValueError('Native RL dictionary mismatch: '+path)
        for key in left:tree_equal(left[key], right[key], path+'.'+str(key))
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):raise ValueError('Native RL sequence mismatch: '+path)
        for i, (a, b) in enumerate(zip(left, right)):tree_equal(a, b, path+'.'+str(i))
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):raise ValueError('Native RL numpy/RNG mismatch: '+path)
    elif left != right:raise ValueError('Native RL scalar continuation mismatch: '+path)


def _torch_load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def sampling_evidence(rollout):
    """Independently read actual sampled means/stds, not a new/current ratio."""
    import torch
    check, terms = rollout.get('qualification_sampling_check'), rollout.get('qualification_sample_terms')
    if (not isinstance(check, dict) or check.get('status') != 'PASS' or not isinstance(terms, list)
            or len(terms) != 3 or rollout['stoch_idx'].tolist() != [7, 8, 9]):
        raise ValueError('Missing actual real-sampler qualification terms or wrong stochastic window')
    old = rollout['old_probability'];rows = []
    for index, (mean, std) in enumerate(terms):
        step = [7, 8, 9][index];mean = mean.float();std = float(std)
        if not math.isfinite(std) or std <= 0:raise ValueError('Illegal actual sampling standard deviation')
        raw = torch.distributions.Normal(mean, torch.tensor(std, dtype=torch.float32)).log_prob(rollout['chain'][:,step+1].float())
        peak = -math.log(std)-.5*math.log(2*math.pi)
        clamped = raw.clamp(min=peak-7., max=peak)
        row = dict(step=step, mean_max_abs=float((mean-old['mean'][:,index]).abs().max()),
            std_abs=float((torch.tensor(std, dtype=torch.float32)-old['std'][index]).abs()),
            raw_gaussian_max_abs=float((raw-old['raw_gaussian_logprob'][:,index]).abs().max()),
            source_clamped_max_abs=float((clamped-old['source_clamped_logprob'][:,index]).abs().max()),
            source_mean_logp_max_abs=float((clamped.mean((-1,-2))-rollout['old_logp'][:,index]).abs().max()))
        # Means/stds were already compared on the actual GPU path exactly. The
        # independent FP32 CPU density readout may differ by transcendental ULPs.
        if row['mean_max_abs'] != 0 or row['std_abs'] != 0 or any(
                not math.isfinite(row[k]) or row[k] > 5e-6 for k in
                ('raw_gaussian_max_abs','source_clamped_max_abs','source_mean_logp_max_abs')):
            raise ValueError('Actual sampler and saved-state probability disagree: '+repr(row))
        rows.append(row)
    return dict(status='PASS', steps=rows, mean_std='actual GPU exact',
        independent_CPU_density_atol=5e-6, raw_density='Gaussian per coordinate',
        used_probability='source relative clamp width7 then H/D mean; empirical transition surrogate')


def checkpoint_at(run, update):
    from starVLA.rl.qwen_world_compare.loading import inspect_RL_checkpoint
    paths = sorted((Path(run)/'checkpoints').glob(f'update{update:06d}_inner*'))
    if len(paths) != 1:raise ValueError('Qualification needs exactly one sealed checkpoint at update '+str(update))
    return inspect_RL_checkpoint(paths[0], verify_files='all')


def prepare_resume(baseline, output):
    """Fork only evidence/history for an independent disposable native resume."""
    baseline, output = Path(baseline), Path(output)
    mid = checkpoint_at(baseline, MID_UPDATE)
    if (mid['seal']['next_inner'] != 2 or mid['seal']['phase'] != 'INNER_COMMITTED'
            or mid['identity']['world'] != 8 or mid['identity']['purpose'] != 'qualification_not_formal_initialization'):
        raise ValueError('Native qualification resume must start at eight-rank inner2')
    receipt = dict(schema='qwen_world_compare_RL_qualification_fork_v1',
        baseline=str(baseline.resolve()), parent_checkpoint=mid['path'],
        parent_files_sha256=file_sha(Path(mid['path'])/'files.json'),
        identity_sha256=mid['identity']['sha256'], preserved_updates=2,
        qualification_only=True, formal_initialization=False)
    if output.exists():
        if read(output/'QUALIFICATION_FORK.json') != receipt:raise ValueError('Resume qualification directory already has another ancestry')
        return receipt
    temporary = output.with_name(output.name+'.incomplete.'+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    for name in ('identity.json', 'resolved_recipe.yaml'):
        shutil.copy2(baseline/name, temporary/name)
    if (baseline/'source').exists():shutil.copytree(baseline/'source', temporary/'source')
    for rank in range(8):
        lines = [json.loads(line) for line in (baseline/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        prefix = [row for row in lines if row['updates'] <= MID_UPDATE]
        if [row['updates'] for row in prefix] != [1,2]:raise ValueError('Missing effective qualification history before inner2')
        (temporary/f'metrics_rank{rank}.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in prefix))
        source = baseline/'behavior/collection000001'/f'rank{rank}.pt'
        target = temporary/'behavior/collection000001'/f'rank{rank}.pt'
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(source, target)
    atomic_json(temporary/'QUALIFICATION_FORK.json', receipt)
    os.rename(temporary, output)
    return receipt


def validate_saved_recipes(baseline_recipe, resumed_recipe, source_config, architecture):
    """The audited train_rl copies common YAML; architecture resolves in runtime."""
    from starVLA.rl.qwen_world_compare.runtime import load_recipe
    expected=load_recipe(source_config,architecture)
    source_sha=file_sha(source_config)
    for path in (baseline_recipe,resumed_recipe):
        if file_sha(path)!=source_sha:raise ValueError('Qualification saved recipe is not the exact audited common source copy')
        if load_recipe(path,architecture)!=expected:raise ValueError('Copied source recipe resolves differently')
    return dict(common_source_config_sha256=source_sha,
        baseline_saved_recipe_sha256=file_sha(baseline_recipe),resumed_saved_recipe_sha256=file_sha(resumed_recipe),
        resolved_recipe_sha256=fingerprint(expected),
        saved_representation='exact_common_source_copy; architecture resolved by audited runtime')


def verify_real_RL(architecture, baseline, resumed, parent_checkpoint, selection_path, training_source,
                   controller_source, config, il_acceptance, output):
    from starVLA.rl.qwen_world_compare.loading import inspect_IL_checkpoint
    from starVLA.rl.qwen_world_compare.parent_selection import validate_il_parent_selection
    from starVLA.rl.qwen_world_compare.recovery_state import validate_boundary, rollout_digest
    from starVLA.rl.qwen_world_compare.runtime import load_recipe
    from tools.qwen_world_compare.campaign import expected_source_files
    parent = inspect_IL_checkpoint(parent_checkpoint, verify_files='model')
    selection = validate_il_parent_selection(selection_path, parent)
    if parent['identity']['architecture_id'] != architecture:raise ValueError('Qualification uses another architecture IL parent')
    baseline, resumed = Path(baseline), Path(resumed)
    end, other, mid = checkpoint_at(baseline,4), checkpoint_at(resumed,4), checkpoint_at(baseline,2)
    identity = end['identity']
    if (identity != other['identity'] or identity != mid['identity'] or identity['world'] != 8
            or identity['purpose'] != 'qualification_not_formal_initialization'
            or identity['architecture_id'] != architecture
            or identity['IL_parent']['model_sha256'] != parent['model_sha256']
            or identity['IL_parent']['identity'] != parent['identity']):
        raise ValueError('Qualification parent/reference or eight-rank source identity changed')
    training_files=expected_source_files(training_source)
    accepted=read(il_acceptance)
    if (accepted.get('status') != 'PASS' or accepted.get('architecture_id') != architecture
            or accepted.get('source_files') != training_files
            or accepted.get('formal_initialized_from_qualification') is not False
            or accepted.get('config_sha256') != file_sha(Path(training_source)/f'configs/qwen_world_compare/arch_{architecture.lower()}.yaml')):
        raise ValueError('Original real IL engineering acceptance does not bind the unchanged training snapshot')
    if (identity['recipe'] != load_recipe(config,architecture) or not identity.get('source_files')
            or any(training_files.get(name) != sha for name,sha in identity['source_files'].items())):
        raise ValueError('Qualification did not execute the unchanged locked training source/recipe')
    config_hash = file_sha(config)
    recipe_check=validate_saved_recipes(baseline/'resolved_recipe.yaml',resumed/'resolved_recipe.yaml',config,architecture)
    common, recovered = _torch_load(Path(end['path'])/'optimizer.pt'), _torch_load(Path(other['path'])/'optimizer.pt')
    tree_equal(common, recovered, 'adapters_Adam_scheduler')
    rank_checks = [];nonzero_PG = 0;changed_names = set()
    for rank in range(8):
        left = _torch_load(Path(end['path'])/f'rank{rank}.pt')
        right = _torch_load(Path(other['path'])/f'rank{rank}.pt')
        pending = _torch_load(Path(mid['path'])/f'rank{rank}.pt')
        for own in (left,right):
            validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'])
            if any(own['state'][k] != v for k,v in dict(updates=4,collections=1,next_inner=0,raw_scene_exposures=8,new_candidates=64,scene_cursor=8).items()):
                raise ValueError('Qualification did not execute four real inner updates on one fresh collection')
        # Time measurements differ between invocations; scientific progress,
        # chain, optimizer and random state must not differ.
        tree_equal({k:v for k,v in left['state'].items() if not k.startswith('seconds_')},
                   {k:v for k,v in right['state'].items() if not k.startswith('seconds_')},'rank.state')
        tree_equal(left['rng'],right['rng'],'rank.RNG');tree_equal(left['rollout'],right['rollout'],'rank.rollout')
        original = _torch_load(baseline/'behavior/collection000001'/f'rank{rank}.pt')
        tree_equal(rollout_digest(original),rollout_digest(pending['rollout']),'mid_inner_chain_old_probability_advantage')
        sample_check = sampling_evidence(original)
        rows = [json.loads(line) for line in (baseline/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        replay = [json.loads(line) for line in (resumed/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
        if [r['updates'] for r in rows] != [1,2,3,4] or [r['updates'] for r in replay] != [1,2,3,4]:
            raise ValueError('Qualification lacks four continuous effective optimizer journal rows')
        if not math.isfinite(rows[0]['ratio_max_abs_from_one']) or rows[0]['ratio_max_abs_from_one'] > 1e-4:
            raise ValueError('Real first-inner fresh-ratio gate failed')
        for a,b in zip(rows,replay):
            if a['collection'] != 1 or a['inner_completed'] != a['updates']:
                raise ValueError('One collection was not reused for exactly four actual updates')
            if not math.isfinite(a['gradient_norm']) or not math.isfinite(a['loss']):
                raise ValueError('Real optimizer update had a nonfinite loss/gradient')
            for key in ('updates','collection','inner_completed','run','chain_identity','raw_scene_exposures','new_candidates',
                        'token','behavior','lr','loss','gradient_norm','ratio_max_abs_from_one','source_clamp_hit_rate','metrics','exploration'):
                tree_equal(a[key],b[key],'journal.'+key)
        result, resumed_result = read(baseline/f'result_rank{rank}.json'), read(resumed/f'result_rank{rank}.json')
        for value in (result,resumed_result):
            if (value.get('qualification') is not True or value.get('updates') != 4
                    or value.get('frozen_SHA_before') != value.get('frozen_SHA_after')
                    or not value.get('frozen_SHA_before')
                    or value.get('frozen_parameters_before') != value.get('frozen_parameters_after')
                    or not value.get('frozen_parameters_before')):
                raise ValueError('Frozen observation/base/permanent parent reference changed')
        tree_equal(result['frozen_SHA_after'],resumed_result['frozen_SHA_after'],'frozen_parent')
        changed_names.update(name for name,changed in result['changed_adapters'].items() if changed)
        nonzero_PG += int(not rows[0]['exploration']['zero_PG'])
        rank_checks.append(dict(rank=rank,sampling=sample_check,first_inner_ratio_error=rows[0]['ratio_max_abs_from_one'],
            frozen_parent_sha256=result['frozen_SHA_after'],four_real_inners=True,native_inner2_resume_exact=True,
            saved_chain_digest=rollout_digest(original)))
    if nonzero_PG == 0 or not changed_names:
        raise ValueError('Finite qualification had no actual PG/adapter learning signal; evidence insufficient, not an algorithm failure')
    return publish(output, dict(schema=SCHEMA,status='PASS',architecture_id=architecture,
        source_files=training_files,controller_source_files=expected_source_files(controller_source),
        training_source=str(Path(training_source).resolve()),controller_source=str(Path(controller_source).resolve()),
        original_IL_acceptance=str(Path(il_acceptance).resolve()),original_IL_acceptance_file_sha256=file_sha(il_acceptance),
        rl_config_sha256=config_hash,IL_parent_checkpoint=parent['path'],IL_parent_model_sha256=parent['model_sha256'],
        saved_recipe_check=recipe_check,
        IL_parent_identity_sha256=fingerprint(parent['identity']),IL_selection_sha256=selection['sha256'],
        IL_selection_file_sha256=file_sha(selection_path),rank_checks=rank_checks,
        baseline_checkpoint=end['path'],resumed_checkpoint=other['path'],mid_inner_checkpoint=mid['path'],
        checkpoint_files_sha256={x['path']:file_sha(Path(x['path'])/'files.json') for x in (end,other,mid)},
        changed_adapter_names=sorted(changed_names),ranks_with_nonzero_PG=nonzero_PG,
        qualification_logical_updates=4,qualification_physical_updates=6,
        qualification_fresh_candidates=64,formal_initialized_from_qualification=False))


def validate_acceptance(path, architecture, parent_selection, selection_path, training_files, controller_files, rl_config, il_acceptance):
    value = _sealed(read(path));parent = parent_selection['selected']
    if (value.get('schema') != SCHEMA or value.get('status') != 'PASS' or value.get('architecture_id') != architecture
            or value.get('source_files') != training_files or value.get('controller_source_files') != controller_files
            or value.get('rl_config_sha256') != file_sha(rl_config)
            or value.get('IL_parent_checkpoint') != str(Path(parent['checkpoint']).resolve())
            or value.get('IL_parent_model_sha256') != parent['model_sha256']
            or value.get('IL_selection_sha256') != parent_selection['sha256']
            or value.get('IL_selection_file_sha256') != file_sha(selection_path)
            or value.get('original_IL_acceptance') != str(Path(il_acceptance).resolve())
            or value.get('original_IL_acceptance_file_sha256') != file_sha(il_acceptance)
            or value.get('formal_initialized_from_qualification') is not False
            or value.get('qualification_logical_updates') != 4 or len(value.get('rank_checks',[])) != 8):
        raise ValueError('Real RL acceptance does not bind the exact own IL parent/source/config')
    for checkpoint,sha in value.get('checkpoint_files_sha256',{}).items():
        if file_sha(Path(checkpoint)/'files.json') != sha:raise ValueError('Qualification native checkpoint inventory changed')
    return value


def main():
    parser = argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare-resume');p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify')
    for name in ('architecture','baseline','resumed','parent-checkpoint','selection-path','training-source','controller-source','config','il-acceptance','output'):
        p.add_argument('--'+name,required=True)
    p=sub.add_parser('migrate')
    for name in ('training-source','controller-source','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--evidence',nargs='+',required=True)
    args=vars(parser.parse_args());mode=args.pop('mode');args={k.replace('-','_'):v for k,v in args.items()}
    if mode=='prepare-resume':value=prepare_resume(**args)
    elif mode=='verify':value=verify_real_RL(**args)
    else:value=create_execution_migration(**args)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__':main()
