"""Historical CPU-only counterexamples against fixed4ff; no GPU/model weights.

A passed assertion here confirms a defect was reproduced, not correctness PASS.
"""
from copy import deepcopy
import hashlib,json,tempfile
from pathlib import Path
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.action_effect.common import load_config,digest,artifact,write_json
from tools.action_effect import collect_base_trajectories as collector


def run():
    cfg=yaml.safe_load(Path('configs/action_effect/main.yaml').read_text())
    results=[]
    with tempfile.TemporaryDirectory(prefix='effect_cache_review_') as folder:
        root=Path(folder)
        for name,change in [
            ('negative_g0_workers',lambda c:c['g0'].update(workers=-1)),
            ('unknown_nested_training_key',lambda c:c['training'].update(imaginary_loss=99)),
            ('negative_storage_floor',lambda c:c['resources'].update(minimum_free_bytes=-1)),
            ('string_seed',lambda c:c.update(seed='wrong')),
            ('nonfinite_learning_rate',lambda c:c['training'].update(learning_rate=float('nan'))),
        ]:
            wrong=deepcopy(cfg);change(wrong);p=root/(name+'.yaml');p.write_text(yaml.safe_dump(wrong))
            accepted=load_config(p)
            assert accepted is not None
            results.append({'case':name,'defect_reproduced':True,'observed':'invalid_config_accepted'})
        # Explicit fixture replaces prepare ONLY so this isolates the actual
        # collector's completed-manifest fast path before its CUDA guard.
        cohort=root/'COHORT.json';write_json(cohort,{'entries':[{'scene_id':'synthetic_scene'}]})
        binding={'identity':'a'*64,'cohort':artifact(cohort)}
        out=root/'contexts';out.mkdir()
        write_json(out/'MANIFEST_000001.json',{'binding_identity':'a'*64,'shards':[]})
        original=collector.prepare;collector.prepare=lambda _:binding
        try:
            result=collector.collect({'artifact_root':str(root)},max_samples=1,device='cpu')
        finally:collector.prepare=original
        assert result['status']=='PASS' and result['scenes']==1 and result['reused'] is True
        results.append({'case':'empty_manifest_no_head_no_rows','defect_reproduced':True,
                        'observed':result,'cuda_executed':False,'weights_loaded':False})
        # Existing-binding preparation verifies dependencies but does not verify
        # the binding's own identity or its cohort reference.
        fakebase=root/'base.json';faketrain=root/'train.json'
        write_json(fakebase,{'fixture':True});write_json(faketrain,{'fixture':True})
        tiny=deepcopy(cfg);tiny.update(artifact_root=str(root),base_binding=str(fakebase),train_manifest=str(faketrain))
        tiny['resources'].update(minimum_free_bytes=0,artifact_cap_bytes=10_000_000)
        partial={'config_hash':digest(tiny),'base_binding':artifact(fakebase),'train_manifest':artifact(faketrain),
                 'identity':'KNOWN_WRONG_DIGEST','cohort':{'path':str(root/'missing_cohort.json'),'sha256':'x'}}
        write_json(root/'CAMPAIGN_BINDING.json',partial)
        actual=collector.prepare(tiny)
        assert actual['identity']=='KNOWN_WRONG_DIGEST'
        results.append({'case':'existing_binding_wrong_identity_missing_cohort','defect_reproduced':True,
                        'observed':'prepare accepted invalid identity/missing cohort'})
    return {'schema':'action_effect_context_cache_historical_repro_v1',
            'target_commit':'4ff98740b85d4692a55160760e7e53f3faed4af9',
            'interpretation':'Findings reproduced; not production acceptance',
            'source':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                ['tools/action_effect/common.py','tools/action_effect/collect_base_trajectories.py','configs/action_effect/main.yaml']},
            'cases':results,'new_GPU_calls':0,'model_loads':0,'simulator_calls':0}


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,indent=2,allow_nan=False))
