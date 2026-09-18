"""Same-GPU, fixed-real-scene timing of CPU reward/reference overlap.

Six predeclared serial/overlap trials per scene; no reward disk cache. This is
an inference scheduling probe, not optimizer or production acceptance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import torch
from starVLA.rl.flow_grpo.audit import capture_source_environment
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.data import to_device
from starVLA.rl.flow_grpo.loading import load_policy, file_sha
from starVLA.rl.flow_grpo.overlap import score_with_reference
from starVLA.rl.flow_grpo.reward import RewardService
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.rollout import evaluate_transitions


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'bank', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    cfg, sft = resolve_config(args.config)
    configure_numerics()
    out = Path(args.output);out.mkdir(parents=True, exist_ok=False)
    capture_source_environment(out, cfg)
    bank = [torch.load(Path(args.bank)/f'rollout_rank{r}_v0.pt', map_location='cpu', weights_only=False)[0] for r in range(4)]
    policy = load_policy(cfg, sft).cuda().bfloat16().eval()
    policy._inference_qwen_forward_mode = 'optimized'
    tokens = [token for scene in bank for token in scene.observation.tokens]
    service = RewardService(Path('navsim').resolve(), cfg['paths']['metric_cache'], 'train', tokens, workers=1, cache_dir=None)
    executor = ThreadPoolExecutor(1)
    result = {'status': 'RUNNING', 'scope': 'same-GPU scheduling only; no optimizer acceptance',
        'checkpoint_sha256': cfg['checkpoint_contract']['sha256'], 'observer_sha256': file_sha(__file__),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'scenes': []}
    failed = True
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for saved in bank:
                scene = to_device(saved, torch.device('cuda'))
                physical = saved.physical_trajectories
                def score():
                    return service.score(saved.observation.tokens, physical)
                def reference():
                    return evaluate_transitions(policy, scene.observation, scene)
                # Warm both CPU scorer and GPU; fixed, never chosen by outcome.
                expected_records = score()
                expected_stats = reference()
                trials = []
                for mode in ('serial', 'overlap', 'overlap', 'serial', 'serial', 'overlap'):
                    torch.cuda.synchronize();start = time.monotonic()
                    records, stats, phases = score_with_reference(score, reference, 'cuda', executor if mode == 'overlap' else None)
                    torch.cuda.synchronize();seconds = time.monotonic()-start
                    same = records == expected_records and all(torch.equal(stats[k], v) and torch.isfinite(stats[k]).all() for k,v in expected_stats.items())
                    trials.append({'mode':mode, 'seconds':seconds, 'exact_outputs':bool(same), 'phases':phases})
                medians = {mode: statistics.median(r['seconds'] for r in trials if r['mode']==mode) for mode in ('serial','overlap')}
                row = {'tokens': saved.observation.tokens, 'trials':trials, 'median_seconds':medians,
                       'speedup':medians['serial']/medians['overlap']}
                result['scenes'].append(row)
                (out/'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False))
                print(row['tokens'],medians,flush=True)
        result['status'] = 'PASS' if all(t['exact_outputs'] for s in result['scenes'] for t in s['trials']) else 'FAIL'
        (out/'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        if result['status'] != 'PASS':raise RuntimeError('exact scoring/reference outputs changed')
        failed = False
    finally:
        service.close(abort=failed)
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
