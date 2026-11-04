"""Original single-B1 clean-x ODE and official scorer; common scene schema."""
import argparse
import fcntl
import json
from pathlib import Path
import os
import time
import traceback
import uuid
import torch
from starVLA.rl.d44_native_rl.deployment import load_policy
from starVLA.rl.qwen_world_postcot.data import WorldSceneData,POPULATION
from starVLA.rl.qwen_world_compare.contracts import require_dispatch,boundary_stop_due,fingerprint
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches,preflight_caches,json_feedback
from tools.qwen_world_postcot.evaluate import validate_scene_row,validate_feedback
from tools.qwen_world_compare.evaluate import summarize,EVALUATION_SCHEMA,_numerics


def policy_numerics(training):
    value=_numerics()
    if training.get('recipe',{}).get('augustus',{}).get('text_action_precision')=='float32':
        value['model_forward']='float32_text_action_bfloat16_frozen_visual'
    return value


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--summarize',action='store_true');p.add_argument('--baseline')
    a=p.parse_args();root=Path(a.output)
    if a.summarize:
        result=summarize(root,a.baseline);atomic_json(root.parent/'SUMMARY.json',result);print(json.dumps(result));return
    if not a.checkpoint:p.error('--checkpoint required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    host=require_dispatch();configure_numerics();torch.cuda.set_device(0);torch.set_num_threads(2)
    root.mkdir(parents=True,exist_ok=True);(root/'scenes').mkdir(exist_ok=True);(root/'invocations').mkdir(exist_ok=True)
    lock=(root/f'rank{a.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    invocation=f'rank{a.rank}_{uuid.uuid4().hex}';started=time.monotonic();record=dict(invocation=invocation,rank=a.rank,world=a.world,
        host=host,new=0,reused=0,physical_inference_attempts=0,model_load_seconds=0.,forward_seconds=0.,scoring_seconds=0.,status='STARTING',start_unix=time.time(),shared_GPU_worker_time=True)
    service=None
    try:
        data=WorldSceneData('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST',role='NAVTEST')
        population=json.loads(POPULATION.read_text());tokens=population['navtest_tokens']
        if set(tokens)!=set(data.tokens)or len(tokens)!=12146:raise ValueError('NAVTEST population changed')
        if a.limit:tokens=tokens[:a.limit]
        if not 0<=a.rank<a.world<=len(tokens):raise ValueError('Invalid evaluator rank/world')
        roots=evaluation_caches('NAVTEST');preflight_caches(data,tokens[a.rank::a.world],roots)
        tick=time.monotonic();model,training,seal,model_sha=load_policy(a.checkpoint,'cuda');record['model_load_seconds']=time.monotonic()-tick
        processor=QwenObservationProcessor(str(QWEN_PATH));service=RewardService(Path.cwd(),workers=1,cache_roots=roots)
        identity=dict(schema=EVALUATION_SCHEMA,architecture_id='D',cohort='NAVTEST',tokens=tokens,count=len(tokens),token_sha256=fingerprint(tokens),
            data_sha256=data.sha,seed=2101,inference='single_candidate_cleanx_K10_shift5',qualification=bool(a.limit)or training['qualification'],
            protocol=service.protocol,model_sha256=model_sha,numerics=policy_numerics(training),D44_training_identity=training['sha256'],
            model_identity_scheme='SHA256(native_parent_digest,unmerged_trainable_payload_digest,scope)',
            checkpoint=str(Path(a.checkpoint).resolve()),updates=seal['updates'],test_selected_research=True,
            training_data_is_NAVTRAIN_only=True,no_R16_no_CoT_no_scorer_at_actor_input=True)
        identity['sha256']=fingerprint(identity)
        with(root/'identity.lock').open('a')as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if (root/'identity.json').exists():
                if json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Evaluation identity changed')
            else:atomic_json(root/'identity.json',identity)
        for token in tokens[a.rank::a.world]:
            if boundary_stop_due():record['status']='STOPPED_AT_SCENE_BOUNDARY';break
            log=population['token_logs'][token];path=root/'scenes'/f'{token}.json'
            if path.exists():validate_scene_row(json.loads(path.read_text()),token=token,identity_sha256=identity['sha256'],protocol_sha256=service.protocol['sha256'],expected_log=log);record['reused']+=1;continue
            inputs=processor([data.observation(token)]);ego=torch.tensor([data.records[token]['proprio']],device='cuda',dtype=torch.float32)
            noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token))).cuda()
            tick=time.monotonic();record['physical_inference_attempts']+=1
            prediction=model.predict(inputs,ego,noise);torch.cuda.synchronize();record['forward_seconds']+=time.monotonic()-tick
            requested=prediction['requested_trajectory'].cpu().numpy();tick=time.monotonic();feedback=service.score((token,log,requested));record['scoring_seconds']+=time.monotonic()-tick
            if len(feedback)!=1:raise ValueError('Scorer response count changed')
            validate_feedback(feedback[0],token=token,log=log,requested=requested[0],protocol_sha256=service.protocol['sha256'])
            row=dict(token=token,log=log,identity=identity['sha256'],status='OK',requested_trajectory=requested[0].tolist(),
                normalized_action=prediction['normalized_action'][0].cpu().tolist(),feedback=json_feedback(feedback[0]))
            row['sha256']=fingerprint(row);atomic_json(path,row);record['new']+=1
            record['wall_seconds']=time.monotonic()-started;atomic_json(root/'invocations'/f'{invocation}.json',record)
        else:record['status']='COMPLETE'
    except BaseException:record.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        if service:service.close()
        record.update(wall_seconds=time.monotonic()-started,end_unix=time.time());atomic_json(root/'invocations'/f'{invocation}.json',record)
        cumulative={k:0. for k in ('physical_inference_attempts','new','model_load_seconds','forward_seconds','scoring_seconds','wall_seconds')}
        for path in (root/'invocations').glob(f'rank{a.rank}_*.json'):
            value=json.loads(path.read_text())
            for key in cumulative:cumulative[key]+=value.get(key,0.)
        atomic_json(root/f'rank{a.rank}.json',dict(record,cumulative_worker_cost=cumulative))


if __name__=='__main__':main()
