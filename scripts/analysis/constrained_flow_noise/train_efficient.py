"""Qualified execution adapter around the unchanged formal native trainer.

Scientific identity and original checkpoint checks remain intact. A separately
sealed execution identity is saved in every new native checkpoint. All scoring
futures finish before the trainer can reach a save or behavior-refresh boundary.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys

from starVLA.rl.constrained_flow_noise.contracts import file_sha,fingerprint,require_dispatch
from starVLA.rl.constrained_flow_noise.safety_costs import ConstrainedRewardService
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.flow_grpo.transactions import atomic_json
from . import train
from .pipeline_collection import collect_batch
from .vector_geometry import initialize_worker


def source_identity():
    parent=Path(__file__).parent
    sources={name:file_sha(parent/name) for name in ('train_efficient.py','pipeline_collection.py','vector_geometry.py')}
    value=dict(schema='cfn_execution_overlap_geometry_v1',sources=sources,
               model_layout='unchanged rollout B1 and flat saved-chain update',
               scoring='unchanged canonical singleton proposal pool; vector GEOS geometry',
               order='original input scene order and native first16/refill8 boundaries',
               checkpoint='no score future survives collect_batch return')
    value['sha256']=fingerprint(value)
    return value


class EfficientRewardService(ConstrainedRewardService):
    def __init__(self,spec,workers=2):
        self.spec=dict(spec)
        self.spec['protocol_sha256']=v1_reward.protocol_identity(spec['repo'])['sha256']
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                      initializer=initialize_worker,initargs=(self.spec,))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--acceptance',help='Full exact eight-rank learning/checkpoint qualification, required for formal use')
    # torchrun may consume the separator before forwarding module arguments.
    # Parse our one explicit option and preserve all native arguments in order.
    a,args=p.parse_known_args();require_dispatch()
    if args and args[0]=='--':args=args[1:]
    identity=source_identity()
    if '--qualification' not in args:
        if not a.acceptance or '--resume' not in args:raise ValueError('Formal acceleration requires exact-state acceptance and native continuation')
        proof=json.loads(Path(a.acceptance).read_text())
        if (proof.get('status')!='PASS' or proof.get('execution_sha256')!=identity['sha256']
                or not all(proof.get(k) is True for k in ('both_variants_exact_learning','all_rank_optimizer_rng_exact','saved_chains_feedback_exact','resume_exact'))):
            raise ValueError('Execution adapter lacks complete equivalence proof')
    output=Path(args[args.index('--output')+1])
    def publish():
        if int(os.environ.get('RANK','0'))!=0:return
        if not (output/'run_identity.json').exists():return
        record=output/'execution_identity.json'
        if record.exists() and json.loads(record.read_text())!=identity:raise ValueError('Changed execution adapter')
        atomic_json(record,identity)
        snapshot=output/'execution_source_snapshot';snapshot.mkdir(exist_ok=True)
        for name,digest in identity['sources'].items():
            path=snapshot/name;source=Path(__file__).with_name(name)
            if path.exists() and file_sha(path)!=digest:raise ValueError('Execution snapshot changed')
            if not path.exists():path.write_bytes(source.read_bytes())
        if a.acceptance:
            atomic_json(output/'execution_acceptance.json',json.loads(Path(a.acceptance).read_text()))
    if int(os.environ.get('RANK','0'))==0:
        atomic_json(output.with_suffix('.execution_launch.json'),identity)
    # Publish inside the native-created output, never precreate a fresh run.
    class RecordingRewardService(EfficientRewardService):
        def __init__(self,spec,workers=2):
            super().__init__(spec,workers);publish()
    native_save=train.save_checkpoint
    def save(accelerator,engine,root,state,scientific_identity,**kwargs):
        state['execution_adapter']=identity
        return native_save(accelerator,engine,root,state,scientific_identity,**kwargs)
    native_resume=train.resume_checkpoint
    def resume(*pos,**kw):
        state=native_resume(*pos,**kw)
        previous=state.get('execution_adapter')
        if previous is not None and previous!=identity:raise ValueError('Checkpoint execution adapter changed')
        state['execution_adapter']=identity
        return state
    train.collect_batch=collect_batch
    train.ConstrainedRewardService=RecordingRewardService
    train.save_checkpoint=save
    train.resume_checkpoint=resume
    sys.argv=[train.__file__,*args]
    try:train.main()
    finally:publish()


if __name__=='__main__':main()
