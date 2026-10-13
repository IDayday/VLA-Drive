"""Bounded A/B storage-only transport profile using unchanged canonical scores."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.scoring_queue import ScoringClient


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('local-queue', 'shared-queue', 'population', 'reference-scores', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); host = require_dispatch()
    pop = json.loads(Path(a.population).read_text())
    ref = Path(a.reference_scores); manifest = json.loads((ref/'manifest.json').read_text())
    tokens = pop['calibration_tokens'][:2]
    clients = {name: ScoringClient(getattr(a, name+'_queue'), manifest['protocol_sha256']) for name in ('local', 'shared')}
    rows = []
    for token in tokens:
        path = ref/'scenes'/token[:2]/(token+'.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != manifest['sha256'] or seal['payload_sha256'] != file_sha(path):
            raise ValueError('Reference score identity changed')
        rows.append(torch.load(path, weights_only=True, map_location='cpu'))
    records = []
    for repeat in range(8):
        for i, token in enumerate(tokens):
            row = rows[i]
            for name in (('local', 'shared') if (repeat+i) % 2 == 0 else ('shared', 'local')):
                begin = time.monotonic()
                result, receipt = clients[name].score(token, pop['token_logs'][token], row['sample']['requested_trajectories'].numpy(),
                                                       caller_identity='storage_only_AB_128_calls_v1', request_key=f'{token}_{repeat}')
                elapsed = time.monotonic()-begin
                for actual, expected in zip(result, row['feedback']):
                    if actual['metrics'] != expected['metrics'] or not np.array_equal(actual['simulated_states'], expected['simulated_states'].numpy()):
                        raise ValueError('Storage transport changed official outcomes')
                records.append(dict(storage=name, repeat=repeat, token_index=i, end_to_end_seconds=elapsed,
                                    wait_seconds=receipt['wait_seconds'], candidate_count=receipt['candidate_count'], reused=receipt['reused']))
    medians = {name: float(np.median([x['end_to_end_seconds'] for x in records if x['storage'] == name])) for name in clients}
    report = dict(status='PASS_IDENTICAL_OFFICIAL_OUTCOMES', host=host, source_sha256=file_sha(__file__),
                  protocol_sha256=manifest['protocol_sha256'], scenes=2, total_actual_scoring_calls=128,
                  repeated_candidates_are_profile_overhead_not_discoveries=True, median_roundtrip_seconds=medians,
                  shared_to_local_ratio=medians['shared']/medians['local'], records=records,
                  interpretation='Small-queue latency only; does not prove a full-bank speedup or change GPU topology/numerics.')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
