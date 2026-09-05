#!/usr/bin/env python3
"""Current-only FP32 replay of an old immutable bank; no rescoring/training."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from planreg_audit_runtime import (load_formal_training_agent, build_navtest_samples,
    collate_samples, to_device_non_paths, sha256_file)
from navsim.common.dataloader import MetricCacheLoader


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'checkpoint', 'bank', 'metric-cache', 'logs', 'sensors', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    _, agent, report = load_formal_training_agent(args.config, args.checkpoint, device=device)
    agent.eval()
    with np.load(args.bank, allow_pickle=False) as bank:
        tokens = bank['tokens'][:4].tolist()
        proposals = bank['proposals'][:4].copy()
        indices = bank['selected_indices'][:4].copy()
    cache = MetricCacheLoader(args.metric_cache)
    metadata = {t:{'log_name':Path(cache.metric_cache_paths[t]).relative_to(args.metric_cache).parts[0]} for t in tokens}
    samples = build_navtest_samples(agent, tokens, metadata, navsim_log_path=args.logs, sensor_blobs_path=args.sensors)
    differences = []
    for index, token in enumerate(tokens):
        features, _unused = collate_samples([samples[token]])
        with torch.inference_mode():
            prediction = agent(to_device_non_paths(features, device))
        actual = prediction['proposals'].float().cpu().numpy()[0]
        selected = int(prediction['pdm_score'][0].argmax())
        differences.append({'token':token, 'proposal_max_abs_diff':float(np.abs(actual-proposals[index]).max()),
                            'selected_index_equal': selected == int(indices[index])})
    report.update(kind='new_real_model_legacy_replay_no_new_pdms', bank_sha256=sha256_file(args.bank), scenes=differences)
    report['status'] = 'PASS' if all(x['proposal_max_abs_diff']<=1e-5 and x['selected_index_equal'] for x in differences) else 'FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise RuntimeError('Legacy replay differs; inspect report, do not relabel tolerance')


if __name__ == '__main__':
    main()
