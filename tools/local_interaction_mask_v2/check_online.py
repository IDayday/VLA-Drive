"""Real GPU engineering parity: public checkpoint restore, online/cache, targets, gate zero."""
import argparse
import json
from pathlib import Path
import torch
from tools.local_interaction_mask_v2.extract_current import load_foundation
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,current_example
from tools.local_interaction_mask_v2.planner_runtime import CurrentOnlyCorpus,load_bridge,predict_payload,predict_payloads
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.local_planner import LocalPlanningBridge,PublicLocalPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('foundation','public-qwen','cache','dataset','current-bridge','graph-checkpoint','output'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--visual-cache');p.add_argument('--limit',type=int,default=4);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    corpus=CurrentOnlyCorpus(a.cache);identity=corpus.manifest['identity']
    world,metadata=load_foundation(a.foundation,a.public_qwen,a.visual_cache)
    dim=corpus[0]['native_actions'].shape[-1]
    trained,_=load_bridge(a.current_bridge,corpus.manifest['identity_sha256'],dim)
    graph=torch.load(a.graph_checkpoint,map_location='cpu',weights_only=False)
    spec=json.loads(Path(a.dataset).read_text());rows=[]
    for mode in ('current','all','mask'):
        bridge=LocalPlanningBridge(dim,identity['graph_config'],mode).cuda().eval().requires_grad_(False)
        if bridge.graph is not None:bridge.graph.load_state_dict(graph['model'],strict=True)
        for name in ('current_projection','graph_to_world','adapter'):
            getattr(bridge,name).load_state_dict(getattr(trained,name).state_dict(),strict=True)
        policy=PublicLocalPolicy(world,bridge,identity).eval()
        batch=predict_payloads(world.baseline.action_model,bridge,[corpus[i] for i in range(min(a.limit,len(corpus)))],identity)
        for i in range(min(a.limit,len(corpus))):
            payload=corpus[i];token=payload['token']
            record=current_metadata_from_training_pickle(Path(spec['meta_root'])/(token+'.pkl'),token)
            example,observation=current_example(Path(spec['observations'])/(token+'.npz'),record)
            original_native=world.baseline.native_conditions([example])
            append_tail_delta=float((original_native-payload['native_actions'].cuda()).abs().max())
            cached=predict_payload(world.baseline.action_model,bridge,payload,identity)
            online=policy.predict_action([example],[observation])[0]
            # WorldTargets/action labels are extraneous to the actual deployed API.
            poisoned=dict(example,WorldTargets={'future_xy':float('nan')},actions=float('nan'))
            changed=policy.predict_action([poisoned],[observation])[0]
            visual=world.baseline.current_visual_cache
            world.baseline.current_visual_cache=None
            uncached=policy.predict_action([example],[observation])[0]
            world.baseline.current_visual_cache=visual
            delta=float((cached['normalized_actions']-online['normalized_actions']).abs().max())
            feature_delta=float((cached['action_conditions']-online['action_conditions']).abs().max())
            vision_delta=float((online['normalized_actions']-uncached['normalized_actions']).abs().max())
            target_delta=float((online['normalized_actions']-changed['normalized_actions']).abs().max())
            previous=bridge.adapter.gate.detach().clone();bridge.adapter.gate.zero_()
            zero=predict_payload(world.baseline.action_model,bridge,payload,identity)
            native=predict_payload(world.baseline.action_model,None,payload,identity)
            bridge.adapter.gate.copy_(previous)
            gate_delta=float((zero['normalized_actions']-native['normalized_actions']).abs().max())
            batch_delta=float((cached['normalized_actions'][0]-batch['normalized_actions'][i]).abs().max())
            row={'mode':mode,'token':token,'online_cached_action_max_error':delta,'online_cached_condition_max_error':feature_delta,
                'frozen_visual_cache_action_max_error':vision_delta,'target_poison_action_max_error':target_delta,'gate0_action_max_error':gate_delta,
                'batch_singleton_action_max_error':batch_delta,'trained_append_tail_native_max_error':append_tail_delta}
            rows.append(row)
            if delta>1e-5 or feature_delta>1e-5 or vision_delta>1e-5 or batch_delta>1e-5 or target_delta!=0 or gate_delta!=0 or append_tail_delta!=0:
                atomic_json(out/'failed.json',row);raise AssertionError('Predeclared online/cache or gate/label parity failed')
    atomic_json(out/'result.json',{'status':'PASS','scope':'P1 engineering; ALL/MASK share one diagnostic graph for architecture parity, not comparative research',
        'foundation_step':metadata['step'],'rows':rows,'tolerance_action_and_condition':1e-5,'gate_and_target_tolerance':0.,
        'graphs_have_no_world_targets_input':True,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
    atomic_json(out/'status.json',{'status':'complete'})


if __name__=='__main__':
    with torch.no_grad():main()
