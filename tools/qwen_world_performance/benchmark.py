"""Summarize existing full-weight qualification; never start a model or job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path('/mnt/project/qwen-world-performance-rl-20261103')
CASES = ('B64_G16', 'B128_G16_continuous', 'B64_G32', 'B128_G32')
TIMING = ('seconds_rollout', 'seconds_scoring_wait', 'seconds_old_probability', 'seconds_update')


def read(path, inputs):
    path = Path(path);raw = path.read_bytes()
    inputs.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    return json.loads(raw)


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.benchmark.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def case_result(root, name, population, inputs, deadline, now):
    directory = root/'qualification_v3'/name
    cfg = yaml.safe_load((directory/'resolved_recipe.yaml').read_text())
    p = cfg['research']['performance'];b = p['scenes_per_collection'];g = p['group_size'];m = p['inner_updates']
    ranks = [read(directory/f'result_rank{rank}.json', inputs) for rank in range(8)]
    metrics, preparations = [], []
    for rank, result in enumerate(ranks):
        path = directory/f'metrics_rank{rank}.jsonl';raw = path.read_bytes()
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if ([row['updates'] for row in rows] != list(range(1, m+1)) or result['rank'] != rank
                or result.get('qualification') is not True or result['updates'] != m or result['collections'] != 1
                or result['raw_scene_exposures'] != b or result['new_candidates'] != b*g
                or any(row['group_size'] != g or not math.isfinite(row['loss']) or not math.isfinite(row['gradient_norm']) for row in rows)
                or result['frozen_SHA_before'] != result['frozen_SHA_after']):
            raise ValueError('Incomplete/nonfinite/changed qualification prefix: '+str(path))
        if rows[0]['ratio_max_abs_from_one'] > 1e-4:
            raise ValueError('Fresh behavior ratio failed: '+str(path))
        invocation = read(directory/'invocations'/rows[0]['invocation']/f'rank{rank}.json', inputs)
        preparations.append(invocation.get('preparation_seconds'))
        metrics.extend(rows)
    if cfg['world_size'] != 8 or m != 4:
        raise ValueError('This measured benchmark requires actual world8/4back qualification')
    per_rank_steady = [sum(float(row[key]) for key in TIMING) for row in ranks]
    steady = max(per_rank_steady)
    checkpoint = Path(ranks[0]['checkpoint'])
    inventory = read(checkpoint/'files.json', inputs);seal = read(checkpoint/'COMPLETE', inputs)
    if hashlib.sha256((checkpoint/'files.json').read_bytes()).hexdigest() != seal['files_sha256']:
        raise ValueError('Checkpoint inventory seal mismatch')
    optimizer_bytes = (checkpoint/'optimizer.pt').stat().st_size
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.iterdir() if path.is_file())
    exact_windows = population/b;allocated_windows = math.ceil(exact_windows)
    wall_epoch = exact_windows*steady
    epochs = p['evaluation_epochs']
    storage = []
    for horizon in [3., 6., 20.]:
        count = sum(float(epoch) <= horizon for epoch in epochs)
        adam = math.ceil(population/b)*m*horizon
        periodic_upper = math.ceil(adam/cfg['save_every_optimizer_updates'])+count
        storage.append(dict(through_epochs=horizon, official_evaluation_gates=count,
                            optimizer_bytes_times_evaluation_gates=optimizer_bytes*count,
                            native_checkpoint_bytes_times_evaluation_gates=checkpoint_bytes*count,
                            periodic_plus_eval_checkpoint_count_upper=periodic_upper,
                            optimizer_bytes_with_periodic_upper=optimizer_bytes*periodic_upper))
    return dict(id=name, status='COMPLETE_REAL8RANK_4UPDATE_WINDOW', B=b, G=g, world=8, inner_updates=m,
                host=ranks[0]['host'], unique_measured_windows=1, measured_optimizer_rows=len(metrics),
                timing_seconds=dict(component_max_across_ranks={key:max(row[key] for row in ranks) for key in TIMING},
                    steady_window=max(per_rank_steady), steady_definition='max_rank(sampling + scoring_wait + old_probability + four backward/update times)',
                    componentwise_upper=sum(max(row[key] for row in ranks) for key in TIMING),
                    startup_preparation_by_rank=preparations,
                    startup_preparation_max=max(preparations) if all(isinstance(v,(int,float)) for v in preparations) else None,
                    invocation_training_wall_max=max(row['wall_seconds_this_invocation'] for row in ranks),
                    scoring_CPU_sum_max=max(row['seconds_scoring_cpu'] for row in ranks),
                    scoring_CPU_not_added_to_wall=True),
                peak_GPU_bytes=max(row['peak_GPU_bytes'] for row in ranks),
                checks=dict(all8_rank_complete=True, finite_all_rows=True, frozen_model_unchanged=True,
                            first_inner_ratio_max_error=max(row['ratio_max_abs_from_one'] for row in metrics if row['inner_completed']==1)),
                epoch_budget=dict(population=population, exact_valid_windows=exact_windows, padded_windows=allocated_windows,
                    valid_PG_candidates=population*g, allocated_PG_candidates=allocated_windows*b*g,
                    valid_scene_exposures=population, padded_scene_slots=allocated_windows*b-population,
                    estimated_window_only_wall_seconds=wall_epoch, estimated_window_only_wall_hours=wall_epoch/3600,
                    estimated_window_only_GPU_hours=wall_epoch*8/3600,
                    padded_window_wall_seconds=allocated_windows*steady,
                    formula='N/B * measured_window_seconds * world8 /3600 GPUh',
                    six_epochs_window_only_wall_hours=6*wall_epoch/3600,
                    temporary_seconds_remaining=max(0.,deadline-now),
                    projected_epochs_before_deadline=max(0.,deadline-now)/wall_epoch,
                    six_epochs_before_temporary_deadline_estimated=False if deadline-now < 6*wall_epoch else None),
                checkpoint_storage=dict(path=str(checkpoint), mature_optimizer_bytes=optimizer_bytes,
                    one_world_checkpoint_bytes=checkpoint_bytes, optimizer_common_not_times8=True,
                    gates=storage, excluded='behavior caches, exports, growing teacher/history banks, partial attempts and initial IL files'),
                evidence=dict(directory=str(directory), results=[str(directory/f'result_rank{i}.json') for i in range(8)]))


def build_qualification(root=ROOT, *, now=None):
    root = Path(root);now = __import__('time').time() if now is None else now;inputs = []
    audit = read(root/'data/AUDIT.json', inputs);population = audit['checks']['unique_tokens']
    request = read(root/'operations/RESOURCES_UNTIL_20261103T000000Z/REQUEST.json', inputs)
    rows = [case_result(root, name, population, inputs, request['deadline_unix'], now) for name in CASES]
    resume = read(root/'operations/NATIVE_RESUME_QUALIFICATION.json', inputs)
    rank_checks = resume.get('ranks', {})
    resume_pass = resume.get('status')=='PASS' and all(resume.get('checks',{}).get(key) is True for key in ['adapters','optimizer','scheduler']) and set(rank_checks)==set(map(str,range(8))) and all(row.get('RNG_bitexact') and row.get('scene_stream_bitexact') and row.get('updates')==4 for row in rank_checks.values())
    nominal = read(root/'qualification/REAL_NOMINAL_IL70/COMPLETE.json', inputs)
    nominal_pass = nominal.get('status')=='PASS' and nominal.get('real_full_IL70') is True and nominal.get('exact_final_ODE_head_equal') is True and all(float(nominal.get('gradient_norms',{}).get(key,0))>0 for key in ['action_expert.','scene_provider.','predictive_provider.'])
    return dict(schema='world_real_performance_qualification_and_budget_v1',
                status='PASS_EXISTING_QUALIFICATIONS_ONLY' if resume_pass and nominal_pass else 'REVIEW_REQUIRED',
                created_UTC=datetime.fromtimestamp(now, timezone.utc).isoformat(), configurations=rows,
                native_resume=dict(status=resume.get('status'), exact_pass=resume_pass, checks=resume.get('checks'), ranks=rank_checks,
                    split_checkpoint=resume.get('split_checkpoint'), continuous_checkpoint=resume.get('continuous_checkpoint')),
                nominal_qualification={key:nominal.get(key) for key in ['status','real_full_IL70','formal_training','qualification_gate_injected',
                    'actual_scene_exposures','actual_PDM_queries','peak_GPU_bytes','wall_seconds','teacher_gain','exact_final_ODE_head_equal','gradient_norms']},
                nominal_scenes=len(nominal.get('scenes',[])), formal_NAVTEST_scores_created=False,
                deadline=request, input_files=inputs, limitations=[
                    'One completed first window per B/G, not a long-run steady-state or architecture-causal speed experiment.',
                    'Sampling+scoring-wait+old-prob+4back only; startup is separate, scoring CPU is already within wall wait.',
                    'No evaluation/export/checkpoint stalls, resource queues or algorithm-specific extra forwards included in epoch projection.',
                    'S2/S3 have M2 with extra MC4/MC2 work; these S0/M4 timings are not measured S2/S3 throughput.',
                    'The deadline applies to six temporary hosts; no capacity reservation or six-epoch completion is promised.',
                    'Qualification weights and injected nominal gate are never formal initialization or NAVTEST improvement.'])


def write_documents(root, qualification, docs):
    root, docs = Path(root), Path(docs);docs.mkdir(parents=True,exist_ok=True)
    lines = ['# Real qualification and training budget', '', 'Measured full-weight world8, one complete four-update window per configuration. Epoch estimates exclude startup, checkpoint/export/evaluation stalls, queues and new algorithm overhead.', '',
             '| B / G / M | sampling max s | score wait max s | old-prob max s | four updates max s | window s | startup max s | wall h / epoch | GPUh / epoch | peak GiB / GPU |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in qualification['configurations']:
        t=row['timing_seconds'];c=t['component_max_across_ranks'];e=row['epoch_budget']
        lines.append(f"| {row['B']} / {row['G']} / {row['inner_updates']} | {c['seconds_rollout']:.3f} | {c['seconds_scoring_wait']:.3f} | {c['seconds_old_probability']:.3f} | {c['seconds_update']:.3f} | {t['steady_window']:.3f} | {t['startup_preparation_max']:.3f} | {e['estimated_window_only_wall_hours']:.3f} | {e['estimated_window_only_GPU_hours']:.3f} | {row['peak_GPU_bytes']/2**30:.3f} |")
    first=qualification['configurations'][0];opt=first['checkpoint_storage']['mature_optimizer_bytes'];remaining=first['epoch_budget']['temporary_seconds_remaining']/3600
    lines += ['', 'Window = max over ranks of the sum of the four timed phases; it is not the sum of independently maximal phases. CPU scorer work is not added twice. Startup preparation is invocation metadata and is excluded from the window estimate.', '',
              f"For N={first['epoch_budget']['population']:,}, valid candidates are 1,652,608 per G16 epoch and 3,305,216 per G32 epoch. Padding and extra B1/adaptive-G queries are separate. GPUh/epoch=N/B×window_seconds×8/3600.", '',
              f'Temporary time remaining at this snapshot: {remaining:.3f} h, deadline 2026-11-03 00:00 UTC. These measurements imply roughly18–19 h per full epoch on one eight-GPU task. **Do not promise six epochs within the temporary window.** Resume/relocation to separately authorized capacity is an operations decision, not a measured speedup.', '',
              f'Actual mature optimizer.pt: {opt:,} bytes ({opt/2**30:.6f} GiB) per world checkpoint, stored once, not eight copies. Gate-only optimizer storage = this value × retained gate count. Routine native saves add further checkpoints; behavior, exports, teacher banks and failed attempts are excluded.', '',
              '| horizon epochs | evaluation gates | optimizer-only gate GiB | periodic+eval count upper (B128/M4 example) | optimizer GiB upper |', '|---:|---:|---:|---:|---:|']
    ref=next(row for row in qualification['configurations']if row['id']=='B128_G16_continuous')
    for item in ref['checkpoint_storage']['gates']:
        lines.append(f"| {item['through_epochs']:g} | {item['official_evaluation_gates']} | {item['optimizer_bytes_times_evaluation_gates']/2**30:.3f} | {item['periodic_plus_eval_checkpoint_count_upper']} | {item['optimizer_bytes_with_periodic_upper']/2**30:.3f} |")
    lines += ['', 'The split/native-resume qualification passed adapter, Adam, scheduler and all eight RNG/scene-stream equality checks after4 updates. This is exact-recovery evidence, not a new driving score.', '',
              'The separate real IL70 nominal auxiliary qualification used68 official queries on4 TRAIN scenes, injected gate=.5 at actual exposure0, and a strict teacher gain .4167. CFM/ODE/PPO gradients and final ODE-head equality were checked. It is not formal training, a natural gate activation rate, a throughput benchmark, or NAVTEST improvement.', '',
              'Evidence: `performance_qualification.json`, `operations/NATIVE_RESUME_QUALIFICATION.json`, and `qualification/REAL_NOMINAL_IL70/COMPLETE.json`.']
    (docs/'TRAINING_BUDGET.md').write_text('\n'.join(lines)+'\n')
    configs=[]
    for path in sorted((Path(__file__).resolve().parents[2]/'configs/qwen_world_performance_rl').glob('*.yaml')):
        cfg=yaml.safe_load(path.read_text());p=cfg['research']['performance']
        configs.append((path,cfg,p))
    lines=['# Executable performance configurations', '', 'All algorithms independently start from the same complete IL70 and fresh Adam/RNG42. No qualification weights are formal initialization. Qwen is frozen; actor-attention LoRA and scene/predictive memory are trainable. Original W3 mean anchor β50 remains; it is neither token KL nor an extra BC loss.', '',
           '| ID | algorithm | starting B / G / inner M | action / memory peak LR | PG matching MC | planned rescue/history |', '|---|---|---|---|---|---|']
    for path,cfg,p in configs:
        mc=p.get('awfm',p.get('fpo',{})).get('mc_samples','N/A')
        planned='; '.join(f"{stage['name']}@epoch{stage['start_epoch']}:B{stage['B']}, rescue={stage['rescue']}, history={stage['history']}"for stage in p['stages'])
        lines.append(f"| {cfg['research']['arm']} | {p['algorithm']} | {p['scenes_per_collection']} / {p['group_size']} / {p['inner_updates']} | {cfg['learning_rate']:.1e} / {cfg['memory_learning_rate']:.1e} | {mc} | {planned} |")
    lines += ['', 'M is optimizer reuse; it is not MC. S2 uses MC4/inner2 with conservative signed velocity-MSE and rollout regularization. S3 uses MC2/inner2 with per-MC CFM-energy PPOclip.01; no endpoint-density/full-FPO++ claim. Nominal-rescue CFM uses separate auxiliary MC2/sigma[.1,.9]; it is not source PG matching MC. S4 uses the explicit weighted indicator+actual-time cost in ALGORITHM_DESIGNS.md, with strict feasible precedence. The inherited source clip.005 is not S3’s replacement PG clip.', '',
              'Schedule clock is valid scene exposures/N: warmup .08 epoch, hold to .75, cosine to4 epochs then .05 floor; stage LR bases may differ. Unlike the older I WKC schedule, this is an exposure clock, not an assumed fixed Adam clock. Target6/minimum3 epochs and later8/12/20 are configuration plans, subject to actual progress, explicit review and host authorization.', '',
              'Evaluation epochs: .25/.5/1/2/3/4/6/8/12/20; canonical single-B1 ODE, K10/shift5/seed2101, full NAVTEST12146/invalid0. Partial, qualification, TRAIN and oracle outputs are not formal deployment scores. `analyze.py` leaves unobserved gains/zero repairs/long-training holds null.', '',
              'Config files:']
    lines += [f'- `{path}` (SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`)'for path,_,_ in configs]
    (docs/'CONFIGURATIONS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--output',type=Path);ap.add_argument('--docs',type=Path,default=Path(__file__).resolve().parents[2]/'reports/qwen_world_performance_20261103')
    args=ap.parse_args();value=build_qualification(args.root);write_json(args.output or args.root/'performance_qualification.json',value)
    write_documents(args.root,value,args.docs)
    write_json(args.docs/'performance_qualification.json',value)
    print(json.dumps(dict(status=value['status'],output=str(args.output or args.root/'performance_qualification.json'),configurations=len(value['configurations']))))


if __name__=='__main__':main()
