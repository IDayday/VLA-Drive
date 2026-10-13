"""Small shared-file transport to the existing isolated canonical CPU service.

This transport keeps CPU scoring on the authorized cache host. It adds no model,
reward, normalization, probability, or search logic.
"""
import json
from pathlib import Path
import time

import numpy as np

from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import fingerprint


class ScoringClient:
    def __init__(self, root, protocol_sha256, timeout_seconds=1200):
        self.root = Path(root)
        self.protocol_sha256, self.timeout = protocol_sha256, timeout_seconds
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if manifest['protocol_sha256'] != protocol_sha256:
            raise ValueError('Scoring transport protocol mismatch')
        self.identity = manifest['sha256']
        self.actual_new_scores = 0
        self.reused_scores = 0

    def score(self, token, log, requests, *, caller_identity, request_key):
        requests = np.asarray(requests)
        if requests.ndim != 3 or requests.shape[1:] != (8, 3) or not np.isfinite(requests).all() or not 1 <= len(requests) <= 64:
            raise ValueError('Invalid scoring request; do not convert to zero reward')
        job = {'token': token, 'log': log, 'requests': requests.tolist(),
               'protocol_sha256': self.protocol_sha256, 'server_identity': self.identity,
               'caller_identity': caller_identity, 'request_key': request_key}
        job['sha256'] = fingerprint(job)
        path = self.root / 'requests' / (job['sha256'] + '.json')
        result = self.root / 'results' / path.name
        existed = result.exists()
        if path.exists():
            if json.loads(path.read_text()) != job:
                raise ValueError('Scoring request identity collision')
        else:
            atomic_json(path, job)
        started = time.monotonic()
        while not result.exists():
            status = self.root / 'status.json'
            if status.exists() and json.loads(status.read_text())['status'].startswith(('FAILED', 'STOPPED')):
                raise RuntimeError('Canonical scoring service stopped; affected update must wait for explicit recovery')
            if time.monotonic()-started > self.timeout:
                raise TimeoutError('Canonical scoring timeout; no update allowed')
            time.sleep(.1)
        payload = json.loads(result.read_text())
        if payload['job_sha256'] != job['sha256'] or payload['server_identity'] != self.identity:
            raise ValueError('Scoring result identity mismatch')
        if payload['status'] != 'OK':
            raise RuntimeError('Canonical scoring failed: ' + payload.get('error', 'unknown'))
        if len(payload['feedback']) != len(requests):
            raise ValueError('Scoring changed denominator')
        if existed:
            self.reused_scores += len(requests)
        else:
            self.actual_new_scores += len(requests)
        return payload['feedback'], {'job_sha256': job['sha256'], 'reused': existed,
                                     'candidate_count': len(requests), 'wait_seconds': time.monotonic()-started}
