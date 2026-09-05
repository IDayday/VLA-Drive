#!/usr/bin/env python3
"""Real paired InternVL initialization, varied train logs, update and deployment audit."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.shared_planreg_initialization import capture_shared_trainable_state
from navsim.agents.EpisodeDrive.precision import audit_precision,parameter_update_statistics
from navsim.planning.training.dataset import CacheOnlyDataset,drivevla_cached_collate
from planreg_audit_runtime import to_device_non_paths
from export_planreg_student_checkpoint import export_student_checkpoint,sha256_file


def agent_config(variant,shared):
    audit=json.loads((ROOT/'reports/planreg_wm_v1/formal_vlm_initialization_audit.json').read_text())
    info=audit['base' if variant=='base' else 'driving_vqa']
    os.environ.update(PLANREG_BASE_VLM_PATH=audit['base']['checkpoint_path'],
        PLANREG_VQA_VLM_PATH=audit['driving_vqa']['checkpoint_path'],PLANREG_SHARED_INIT=str(shared),
        PLANREG_VLM_CHECKPOINT_SHA256=info['checkpoint_sha256'],PLANREG_VLM_CONFIG_SHA256=info['config_sha256'],
        DRIVEVLA_SCORE_RAY='0',DRIVEVLA_SCORE_PROCESSES='2',PLANREG_FORMAL_TIMING='1',
        PLANREG_PROMPT_VERSION='single_front_v1p1')
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'navsim/planning/script/config/common/agent')):
        return compose(config_name='episode_drive_task_future_lite_'+variant)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('shared-init','cache-root','scene-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--skip-pair',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.manual_seed(0);torch.set_num_threads(4)
    report=dict(status='RUNNING',kind='real_numerical_smoke_not_formal_training',
                shared_init_sha256=sha256_file(args.shared_init),steps=[])
    def save():
        (args.output/'runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    cfg=agent_config('base',args.shared_init)
    OmegaConf.save(cfg,args.output/'architecture.yaml',resolve=True)
    agent=instantiate(cfg);agent.initialize();agent.to('cuda:0')
    state,metadata=capture_shared_trainable_state(agent)
    report['initial_state']=metadata
    report['agent_checkpoint_loaded']=agent._agent_checkpoint_loaded
    report['physical_decoder_parameters']=sum(p.numel() for p in agent.physical_query_decoder.parameters())
    if not args.skip_pair:
        vqa_cfg=agent_config('vqa',args.shared_init)
        vqa=instantiate(vqa_cfg);vqa.initialize()
        other,other_meta=capture_shared_trainable_state(vqa)
        assert state.keys()==other.keys() and all(torch.equal(v,other[k]) for k,v in state.items())
        assert not vqa._agent_checkpoint_loaded
        report['paired_trainable_initial_state_bitwise_equal']=True
        report['vqa_initial_state']=other_meta
        del vqa,other;gc.collect();torch.cuda.empty_cache()
    del state
    manifest=json.loads(args.scene_manifest.read_text())
    # Only pilot-train logs get upstream gradient updates; dev logs stay untouched.
    rows=[row for row in manifest['rows'] if row['partition']=='train']
    assert {row['category'] for row in rows}=={'turn','stop','crowded','boundary'}
    report['train_scenes']=rows;report['development_logs_untouched']=True
    dataset=CacheOnlyDataset(str(args.cache_root),agent.get_feature_builders(),agent.get_target_builders(),
        log_names=[row['log'] for row in rows],preprocess_images=True,preprocess_future_images=True,
        input_only_cache_name='planreg_input_only',reject_dynamic_feature_keys=True)
    indices=[dataset.tokens.index(row['token']) for row in rows]
    samples=[dataset[index] for index in indices]
    total=43578;agent.configure_total_optimizer_steps(total)
    optimizers,schedulers=agent.get_optimizers(total_optimizer_steps=total)
    optimizer,scheduler=optimizers[0],schedulers[0]['scheduler']
    report['optimizer_groups']=agent._planreg_optimizer_group_summary
    before={name:p.detach().clone() for name,p in agent.named_parameters() if p.requires_grad}
    teacher_before=None
    agent.train()
    for step in range(args.steps):
        pair=[samples[(2*step+i)%len(samples)] for i in range(2)]
        features,targets=drivevla_cached_collate(pair)
        features=to_device_non_paths(features,torch.device('cuda:0'))
        targets=to_device_non_paths(targets,torch.device('cuda:0'))
        agent.set_optimizer_step(step);optimizer.zero_grad(set_to_none=True)
        start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=agent(features);losses=agent.compute_loss(features,targets,pred)
        assert torch.isfinite(losses['loss']) and losses['wm_weight_current']>0
        losses['loss'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent.parameters())
        block_norms=[]
        for block in agent.backbone.model.vision_model.encoder.layers:
            norm=sum(float(p.grad.float().square().sum()) for n,p in block.named_parameters() if 'lora_' in n and p.grad is not None)**.5
            assert norm>0;block_norms.append(norm)
        assert agent.backbone.planning_register_adapter.planning_registers.grad.norm()>0
        assert all(p.grad is None for p in agent.backbone.model.language_model.parameters())
        norm=torch.nn.utils.clip_grad_norm_([p for p in agent.parameters() if p.requires_grad],1.)
        optimizer.step();scheduler.step();agent.update_ema_after_optimizer_step(step+1,total)
        torch.cuda.synchronize()
        report['steps'].append(dict(step=step,tokens=targets['token'],seconds=time.perf_counter()-start,
            loss=float(losses['loss'].detach()),wm_loss=float(losses['wm_loss'].detach()),
            wm_weight=float(losses['wm_weight_current']),grad_norm=float(norm),lora_block_grad_norms=block_norms,
            physical_counts={key:float(value) for key,value in losses.items() if key.endswith('_count')},
            label_audit=agent._latest_physical_label_audit))
        report['precision']=audit_precision(agent,optimizer,True)
        save();del pred,losses
    report['actual_updates']=parameter_update_statistics(before,agent)
    unchanged=[name for name,stats in report['actual_updates'].items() if stats['changed_fraction']==0]
    if unchanged:raise RuntimeError(f'Trainable tensors did not update across real steps: {unchanged}')
    del before
    optimizer.zero_grad(set_to_none=True)
    from navsim.agents.EpisodeDrive.gradient_diagnostics import isolated_same_batch_audit
    with torch.autocast('cuda',dtype=torch.bfloat16):
        report['same_batch_gradients']=isolated_same_batch_audit(agent,features,targets)
    assert all(p.grad is None for p in agent.parameters())
    report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    checkpoint=args.output/'smoke_training.ckpt'
    torch.save(dict(state_dict={'agent.'+k:v for k,v in agent.state_dict().items()},
        optimizer_states=[optimizer.state_dict()],lr_schedulers=[scheduler.state_dict()],global_step=args.steps),checkpoint)
    report['export']=export_student_checkpoint(checkpoint,args.output/'smoke_student.ckpt',resolved_config=cfg)
    agent.eval()
    current={key:value for key,value in features.items() if not key.startswith('future')}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        reference=agent(current)['trajectory'].cpu()
    if agent._score_process_pool:agent._score_process_pool.shutdown()
    del optimizer,scheduler,optimizers,schedulers,agent,dataset,samples;gc.collect();torch.cuda.empty_cache()
    deployment=copy.deepcopy(cfg);deployment.initialization=None
    deployment.checkpoint_path=str(args.output/'smoke_student.ckpt')
    OmegaConf.update(deployment,'vlm_config.exact_student_checkpoint',True,force_add=True)
    deployment.world_model.enabled=False;deployment.ema.enabled=False;deployment.vlm_config.gradient_checkpointing=False
    student=instantiate(deployment);student.initialize();student.to('cuda:0').eval()
    assert student.physical_query_decoder is None and student.ema_register_target is None and student.future_register_predictor is None
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        result=student(current)['trajectory'].cpu()
    report['current_only_export_max_abs_diff']=float((result-reference).abs().max())
    assert report['current_only_export_max_abs_diff']==0
    report['status']='PASS';save();print(json.dumps({k:report[k] for k in ('status','physical_decoder_parameters','paired_trainable_initial_state_bitwise_equal','peak_allocated_gib','current_only_export_max_abs_diff')},indent=2))


if __name__=='__main__':main()
