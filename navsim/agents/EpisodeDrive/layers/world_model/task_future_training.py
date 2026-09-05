"""Training-only glue; never adds physics/teacher/labels to policy scene memory."""
import time
import numpy as np
import torch
from .task_future_loss import sample_training_candidates, task_future_lite_loss

INPUT_SCHEMA = 'task_future_lite_input_v1_logged_pose'


def validate_lite_config(world_model, consequence, scorer_variant):
    expected=dict(train_candidates=8,time_bins=8,inference_use=False)
    if consequence is None or any(getattr(consequence,k,None)!=v for k,v in expected.items()):
        raise ValueError('Lite requires consequence.train_candidates=8/time_bins=8/inference_use=false')
    if list(getattr(consequence,'target_fields',[])) != ['projected_gap','road_margin','route_progress']:
        raise ValueError('Lite supports exactly gap/road/progress')
    if scorer_variant != 'exact_drivor':
        raise ValueError('Lite keeps the exact unchanged scorer')
    if getattr(world_model,'future_mode',None) != 'correct' or getattr(world_model,'predictor_only',True):
        raise ValueError('Lite formal mode is correct future, no predictor-only control')


def validate_lite_targets(targets, batch_size):
    schema=targets.get('task_future_input_schema')
    if isinstance(schema,str):
        schema=[schema]
    if schema != [INPUT_SCHEMA]*batch_size:
        raise RuntimeError('Stale Lite input cache: rebuild in a NEW root with logged future SE(2) poses; V1.1 cache cannot supply them')
    if targets['logged_future_poses'].shape != (batch_size,3,3):
        raise ValueError('Logged future pose must be [B,3,3] from Scene, never candidate pose')
    mask=targets['future_valid_mask'].bool() & targets['logged_future_pose_valid'].bool()
    if not torch.isfinite(targets['logged_future_poses'][mask]).all():
        raise ValueError('Valid logged future poses must be finite')
    if len(targets.get('physical_map_name',[])) != batch_size:
        raise ValueError('Missing sidecar-only raw map identity')
    return mask


def compute_task_future_lite_loss(agent,features,targets,pred):
    from .physical_label_sidecar import score_with_physical_sidecar
    registers=pred['planning_registers']
    b=len(registers)
    future_valid=validate_lite_targets(targets,b).to(registers.device)
    candidates,indices=sample_training_candidates(targets['trajectory'],pred['proposals'])
    if agent.future_register_predictor is not None or agent.physical_query_decoder is None:
        raise RuntimeError('Lite must construct ONLY the shared physical query decoder')
    if agent.ray:
        raise RuntimeError('Lite sidecar uses the audited process pool, not the legacy Ray scorer path')
    tasks=[]
    for row,token in enumerate(targets['token']):
        path=agent.train_metric_cache_paths[token]
        tasks.append((str(path),pred['proposals'][row].detach().float().cpu().numpy(),
                      targets['trajectory'][row].detach().float().cpu().numpy(),
                      indices[row].cpu().numpy(),targets['physical_map_name'][row]))
    started=time.perf_counter()
    futures=[]
    try:
        if agent.score_process_count:
            agent._ensure_score_process_pool()
            futures=[agent._score_process_pool.submit(score_with_physical_sidecar,*task) for task in tasks]
        # One EMA vision call for current + three logged futures, independent of K.
        target_current,target_future,encoded_valid=agent._encode_ema_register_targets(features,targets,batch_size=b)
        future_valid &= encoded_valid.to(future_valid.device)
        results=[future.result() for future in futures] if futures else [score_with_physical_sidecar(*task) for task in tasks]
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    score_rows,labels=zip(*results)
    # Reuse the legacy conversion/loss verbatim, including all 64 candidate labels.
    request=dict(all_res=score_rows,result=None,submitted_at=started,targets=targets,
                 proposals=pred['proposals'].detach(),test=False)
    scores=agent._resolve_score_request(request)
    original=agent.loss(targets,pred,agent.action_head_config,lambda *a,**kw:scores)
    def stack(name):
        return torch.as_tensor(np.stack([row[name] for row in labels]),device=registers.device)
    timer=agent._formal_phase_timer.start('physical_current_future_head_time')
    try:
        aux=task_future_lite_loss(agent.physical_query_decoder,registers,candidates,
            features['status_feature'][:,:8],target_current,target_future,future_valid,
            targets['logged_future_poses'],stack('physical_values'),stack('gap_class'),stack('valid'))
    finally:
        agent._formal_phase_timer.stop(timer)
    weight=registers.new_tensor(agent.current_world_model_weight())
    agent._latest_physical_label_audit=[{key:value for key,value in row.items() if not isinstance(value,np.ndarray)} for row in labels]
    return {**original,**aux,'plan_loss':original['loss'],
        'loss':original['loss']+weight*aux['wm_loss'],'weighted_wm_loss':weight*aux['wm_loss'],
        'wm_weight_current':weight,'ema_momentum_current':registers.new_tensor(float(agent._ema_current_momentum)),
        'physical_label_extract_time':registers.new_tensor(sum(row['timing']['label_extract_sec'] for row in labels)/b)}
