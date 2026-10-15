"""One canonical simulation, two readouts; no scalar-cost or dual objective."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from starVLA.rl.safe_flow_improvement import v1_reward
from scripts.analysis.constrained_flow_noise.batched_canonical_score import score_scene
from .contracts import CACHE_ROOTS


def readouts(record):
    if record['status']!='OK': raise ValueError('Infrastructure failure is not a zero reward')
    m=record['metrics'];nc=m['no_at_fault_collisions'];dac=m['drivable_area_compliance']
    ep=m['ego_progress'];ttc=m['time_to_collision_within_bound'];comfort=m['comfort']
    canonical=nc*dac*(5*ep+5*ttc+2*comfort)/12
    source=nc*dac*(10*ep+5*ttc+2*comfort)/17
    if not np.isfinite([canonical,source]).all() or abs(canonical-m['score'])>5e-15:
        raise ValueError('Official canonical aggregation differs from registered dual readout')
    return dict(train_reward_source=float(source),canonical_PDMS=float(m['score']))


def score_dual(job):
    records=score_scene(job)
    for r in records:r['readouts']=readouts(r)
    return records


class RewardService:
    def __init__(self, repo, workers=1, *, cache_roots=None):
        self.protocol=v1_reward.protocol_identity(repo)
        spec=dict(repo=str(repo),maps_root='/mnt/navsim/maps',cache_roots=CACHE_ROOTS if cache_roots is None else cache_roots,
                  protocol_sha256=self.protocol['sha256'])
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                      initializer=v1_reward.initialize,initargs=(spec,))

    def submit(self, job):return self.pool.submit(score_dual,job)
    def score(self, job):return self.submit(job).result()
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
