"""Reuse actual legacy behavior buffers; group counts never treat equality as failure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha
from starVLA.rl.safe_flow_improvement.diagnostics import candidate_summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--legacy-buffer-dir', action='append', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    pop = json.loads(Path(a.population).read_text())
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    rows, files, seen, duplicates = [], {}, set(), 0
    for root in a.legacy_buffer_dir:
        for path in sorted(Path(root).glob('rollout_rank*_v*.pt')):
            files[str(path)] = file_sha(path)
            buffers = torch.load(path, weights_only=False, map_location='cpu')  # trusted own historical buffers
            for buffer in buffers:
                for i, token in enumerate(buffer.observation.tokens):
                    h = hashlib.sha256(buffer.chain[i].contiguous().numpy().tobytes()).hexdigest()
                    if (token, h) in seen:
                        duplicates += 1; continue
                    seen.add((token, h))
                    records = buffer.score_records[i]
                    if any(r.status != 'valid' for r in records):
                        raise ValueError('Legacy infrastructure error must not become reward zero')
                    # Old advantages consumed the saved FP32 reward tensor.
                    # Preserve that equality definition; do not re-introduce
                    # sub-FP32 differences from Python scorer floats.
                    metrics = [dict(r.metrics, score=float(buffer.rewards[i, j])) for j, r in enumerate(records)]
                    summary = candidate_summary(buffer.physical_trajectories[i], metrics)
                    rows.append({'token': token, 'log': pop['token_logs'][token], 'chain_sha256': h,
                                 'source': str(path), 'policy_version': buffer.policy_version,
                                 'historical_reward_version': buffer.provenance['reward_version'],
                                 'new_split_population': 'RL_TRAIN' if token in pop['train_tokens'] else 'RL_DEV_OLD_MODEL_HAS_SEEN',
                                 **summary})
    if not rows:
        raise ValueError('No real historical buffers found')
    active = [r for r in rows if not r['all_equal']]
    result = {'status': 'COMPLETE_BUFFER_GROUP_DIAGNOSTIC_GRADIENT_STAGE_PENDING',
              'source_sha256': file_sha(__file__), 'source_files': files, 'duplicates_excluded': duplicates,
              'groups': len(rows), 'all_equal_groups': sum(r['all_equal'] for r in rows),
              'all_zero_groups': sum(r['all_zero'] for r in rows),
              'active_groups': len(active), 'progress_only_active_groups': sum(r['progress_only_effective'] for r in active),
              'safety_varying_groups': sum(any(r['safety_component_variation'].values()) for r in rows),
              'mean_pairwise_ADE_m': float(np.mean([r['pairwise_ADE_mean_m'] for r in rows])),
              'mean_XY_effective_rank': float(np.mean([r['XY_covariance_entropy_effective_rank'] for r in rows])),
              'scope': 'Small preserved pilot/resume buffers, descriptive only; full-log 36.204% / 71.461% historical statistics are not replaced by this subset.',
              'rows': rows}
    atomic_json(out/'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_files')}))


if __name__ == '__main__':
    main()
