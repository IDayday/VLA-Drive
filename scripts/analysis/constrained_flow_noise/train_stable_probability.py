"""Native C continuation with a separately qualified probability-layout repair.

Reuses the qualified batch scorer, collector pipeline, frozen casts and native
trainer/checkpoint. This numerical revision is explicitly NOT a claim of
bitwise equality to the old aborted update. The old guard and data are intact.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

from starVLA.rl.constrained_flow_noise import objective
from starVLA.rl.constrained_flow_noise.contracts import file_sha, fingerprint, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from . import train_efficient as base
from . import train_efficient_batch as batch
from .batched_canonical_score import initialize_worker
from .frozen_weight_cast import install
from .stable_saved_probability import StableSavedProbability


def source_identity():
    result = batch.source_identity()
    result['parent_execution_sha256'] = result.pop('sha256')
    result['schema'] = 'cfn_qualified_sparse_B1_probability_v1'
    result['sources'].update({name:file_sha(Path(__file__).with_name(name)) for name in
                             ('stable_saved_probability.py','train_stable_probability.py')})
    result['model_layout'] = 'Original flat layout; only first-inner transitions above 1e-4 use original B1, frozen through inner2'
    result['scope'] = 'Numerical repair; not bitwise-equivalent to old aborted update; no old-density replacement or guard relaxation'
    result['sha256'] = fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--layout-acceptance')
    parser.add_argument('--qualify-stop-update',type=int)
    known, args = parser.parse_known_args()
    if args and args[0]=='--': args=args[1:]
    require_dispatch()
    identity = source_identity()
    if '--resume' not in args or args[args.index('--variant')+1] != 'C_SDE':
        raise ValueError('This qualified correction continues C_SDE native state only')
    checkpoint = Path(args[args.index('--resume')+1])
    output = Path(args[args.index('--output')+1])
    if not (checkpoint/'COMPLETE').exists(): raise ValueError('Incomplete source checkpoint')
    scientific = json.loads((checkpoint/'identity.json').read_text())
    if scientific['variant'] != 'C_SDE': raise ValueError('Wrong source variant')
    proof = None
    if known.qualify_stop_update:
        boundary = json.loads((checkpoint/'boundary.json').read_text())
        if not (1850 <= boundary['update'] < known.qualify_stop_update <= 1854):
            raise ValueError('Qualification bounded to the actual repaired failure and two collections')
        if 'probability_layout_qualification' not in str(output):
            raise ValueError('Qualification must use a distinct, clearly named output')
    else:
        if not known.layout_acceptance: raise ValueError('Missing numerical repair qualification')
        proof = json.loads(Path(known.layout_acceptance).read_text())
        required = ('failing_chain_probability_pass','transition_values_gradients_pass',
                    'native_update_resume_exact','all_trainable_tensors_updated','saved_old_chain_unchanged')
        if proof.get('status')!='PASS' or proof.get('execution_sha256')!=identity['sha256'] or not all(proof.get(k) is True for k in required):
            raise ValueError('Numerical correction has not passed complete targeted qualification')
        if proof['original_probability_guard'] != 1e-4: raise ValueError('Changed probability guard')
        parent_proof = json.loads(Path(proof['batch_execution_acceptance']).read_text())
        if parent_proof['status']!='PASS' or parent_proof['execution_sha256']!=identity['parent_execution_sha256']:
            raise ValueError('Missing unchanged batching/execution qualification')
    if int(os.environ.get('RANK','0'))==0:
        if not output.exists():
            output.mkdir(parents=True)
            atomic_json(output/'run_identity.json',scientific)
            atomic_json(output/'native_continuation_origin.json',dict(checkpoint=str(checkpoint),
                scientific_identity=scientific['sha256'],qualification_only=bool(known.qualify_stop_update),
                numerical_execution_identity=identity['sha256']))
        old = output/'execution_identity.json'
        if old.exists() and json.loads(old.read_text()) != identity:
            raise ValueError('Never overwrite an existing execution lineage')
    controller = StableSavedProbability()
    native_objective = objective.actor_loss
    objective.actor_loss = lambda actor,**kwargs:controller.objective(native_objective,actor,**kwargs)
    native_micro = base.train.micro_inputs
    def micro(collector,slot,*pos,**kw):
        controller.bind(collector.behavior_identity,slot)
        return native_micro(collector,slot,*pos,**kw)
    base.train.micro_inputs = micro
    native_collector = base.train.EffectiveGroupCollector
    class ExecutionCollector(native_collector):
        def __init__(self,behavior_identity,*pos,**kw):
            version=copy.deepcopy(behavior_identity)
            if version.get('numerical_execution_sha256',identity['sha256']) != identity['sha256']:
                raise ValueError('Saved chain came from another numerical execution')
            version['numerical_execution_sha256']=identity['sha256']
            super().__init__(version,*pos,**kw)
    base.train.EffectiveGroupCollector=ExecutionCollector
    native_save, native_resume = base.train.save_checkpoint, base.train.resume_checkpoint
    def save(accelerator,engine,root,state,scientific_identity,**kw):
        state['execution_adapter']=identity
        state['probability_layout']=controller.state_dict()
        return native_save(accelerator,engine,root,state,scientific_identity,**kw)
    def resume(*pos,**kw):
        state=native_resume(*pos,**kw)
        previous=state.get('execution_adapter')
        if previous is None:
            if state['updates']!=1850 or state['phase']!='ready':
                raise ValueError('Unqualified initial numerical-repair boundary')
        elif previous != identity:
            raise ValueError('Unexpected numerical checkpoint ancestry')
        controller.load_state_dict(state.get('probability_layout'),pending_inner=state['phase']=='updating' and state['next_inner']>0)
        state['execution_adapter']=identity
        return state
    base.train.save_checkpoint,base.train.resume_checkpoint=save,resume
    native_configure=base.train.configure_rollout_graph
    def configure(actor,enabled):
        if enabled:install(actor)
        return native_configure(actor,enabled)
    base.train.configure_rollout_graph=configure
    base.initialize_worker=initialize_worker
    class RewardService(base.EfficientRewardService):
        def __init__(self,spec,workers=2):
            super().__init__(spec,workers)
            if int(os.environ.get('RANK','0'))==0:
                atomic_json(output/'execution_identity.json',identity)
                directory=output/'execution_source_snapshot';directory.mkdir(exist_ok=True)
                for name,digest in identity['sources'].items():
                    target=directory/name;source=Path(__file__).with_name(name)
                    if target.exists() and file_sha(target)!=digest:raise ValueError('Execution snapshot changed')
                    if not target.exists():target.write_bytes(source.read_bytes())
                if proof:atomic_json(output/'execution_acceptance.json',proof)
    base.train.ConstrainedRewardService=RewardService
    base.train.collect_batch=base.collect_batch
    trigger=next(i for i,line in enumerate(Path(base.train.__file__).read_text().splitlines(),1)
                 if 'if rank==0:print(json.dumps(log,allow_nan=False),flush=True)' in line)
    def trace(frame,event,arg):
        if frame.f_code is not base.train.main.__code__:return None
        if event=='line' and frame.f_lineno==trigger and frame.f_locals['state']['updates']>=known.qualify_stop_update:
            base.train.signal_stop()
        return trace
    if known.qualify_stop_update:sys.settrace(trace)
    sys.argv=[base.train.__file__,*args]
    try:base.train.main()
    finally:
        if known.qualify_stop_update:sys.settrace(None)


if __name__=='__main__':main()
