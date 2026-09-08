"""Fixed train-only recorded-drive probe. No optimizer updates or model selection."""
import torch
import hashlib
from .data import v2_collate
from .runtime import rng_state,restore_rng,component_gradient_audit
from .diagnostics import representation_summary,score_summary
from .losses import world_model_loss
from .predictor import state_norm
from .motion import HORIZONS


def fixed_probe(agent,dataset,bank=None,batch_size=2):
    saved=rng_state();was_training=agent.training
    count_context=getattr(agent,'valid_count_context',None)
    agent.valid_count_context=None
    agent.eval();agent.wm_diagnostic_forward=True
    # Read-only gradient diagnostics still need activation checkpointing.
    # This toggles wrapper flags only; dropout/drop-path remain eval/deterministic.
    agent.backbone.activate_gradient_checkpointing_train_mode()
    reports=[];new_bank=[]
    try:
        for offset in range(0,len(dataset),batch_size):
            items=[dataset[i] for i in range(offset,min(offset+batch_size,len(dataset)))]
            features,targets=v2_collate(items)
            query_contract={}
            def check_queries(module,args,kwargs):
                lengths=features['attention_mask'].sum(-1).tolist()
                positions=[]
                for i,length in enumerate(lengths):
                    valid=kwargs['attention_mask'][i].bool()
                    assert valid[:length+16].all() and not valid[length+16:].any()
                    ids=kwargs['position_ids'][i,length:length+16]
                    assert torch.equal(ids,torch.arange(length,length+16,device=ids.device))
                    actual=kwargs['inputs_embeds'][i,length:length+16]
                    torch.testing.assert_close(actual,agent.backbone.task_queries.queries.to(actual),atol=0,rtol=0)
                    positions.append(ids.cpu().tolist())
                query_contract.update(prefix_lengths=lengths,query_positions=positions,padding_checked=True)
            hook=agent.backbone.model.language_model.model.register_forward_pre_hook(check_queries,with_kwargs=True)
            try:predictions=agent(features)
            finally:hook.remove()
            losses=agent.compute_loss(features,targets,predictions)
            if not all(torch.isfinite(v).all() for v in losses.values()):
                raise FloatingPointError('Nonfinite fixed probe loss; tokens='+str(targets['token']))
            scores=agent.last_metric_targets
            report=dict(tokens=targets['token'],losses={k:float(v.detach()) for k,v in losses.items()},
                semantic=representation_summary(predictions['semantic_queries']),
                visual=representation_summary(predictions['visual_content'],predictions['scene_valid_mask']),
                scores=score_summary(predictions,scores),ttc_label_values=scores[...,3].unique().cpu().tolist(),
                memory_lengths=predictions['scene_valid_mask'].sum(-1).cpu().tolist(),
                gradients=component_gradient_audit(losses,agent.named_parameters()),actual_language_query_contract=query_contract)
            if bank is None:
                from .agent import file_sha256
                proposals=predictions['proposals'].detach().cpu()
                new_bank.append(dict(proposals=proposals,scores=scores.cpu(),tokens=targets['token'],
                    candidate_sha256=hashlib.sha256(proposals.contiguous().numpy().tobytes()).hexdigest(),
                    metric_cache_sha256=[file_sha256(p) for p in targets['metric_cache_path']]))
            else:
                entry=bank[offset//batch_size]
                if entry['tokens']!=targets['token']:raise ValueError('Fixed probe candidate provenance mismatch')
                from .agent import file_sha256
                if entry['metric_cache_sha256']!=[file_sha256(p) for p in targets['metric_cache_path']]:raise ValueError('Probe metric cache changed')
                physical=entry['proposals'].to(scores.device)
                status=features['status_feature'].to(scores.device).float()
                ego=agent.action_head.hist_encoding(torch.cat((status.new_zeros(len(status),3),agent.action_head.ego_normalizer(status)),-1))
                with torch.no_grad():
                    logits,log_score=agent.action_head.score(physical,predictions['scene_features'],predictions['scene_valid_mask'],ego)
                    fixed=dict(pred_logit=logits,log_pdm_score=log_score,selected_indices=log_score.argmax(-1))
                    report['fixed_initial_candidate_bank']=score_summary(fixed,entry['scores'].to(scores.device))
            if agent.world_model_enabled:
                with torch.no_grad():
                    teacher=agent.encode_teacher(features,predictions)
                    t=targets['motion_timestamps'].to(scores.device);valid=targets['motion_valid'].to(scores.device)
                    actions,coverage=agent.wm_predictor.motion_encoder(agent.motion_normalizer(targets['motion_sequence'].to(scores.device)),t,valid,HORIZONS)
                    common=(predictions['tile_geometry'],predictions['scene_valid_mask'],predictions['semantic_queries'])
                    controls={}
                    for name,action in [('no_action',torch.zeros_like(actions)),('mismatched_action',actions.roll(1,0))]:
                        tf,ro,_=agent.wm_predictor.branches(predictions['visual_content'],teacher,action,*common)
                        control=world_model_loss(tf,ro,teacher,common[1],targets['future_valid_mask'].to(scores.device),coverage)
                        controls[name]={k:float(v) for k,v in control.items()}
                    copied=state_norm(predictions['visual_content'])[:,None].expand_as(teacher)
                    controls['copy_current']={k:float(v) for k,v in world_model_loss(copied,copied,teacher,common[1],targets['future_valid_mask'].to(scores.device),coverage).items()}
                    report['controls']=controls
            reports.append(report)
        return dict(protocol='training_PDM_fixed_reference_not_official_Navtest',optimizer_updates=int(agent.optimizer_updates),
            updated_on_probe=False,backbone_training_log_exposure='VLM pretraining exposure is not established; these drives are excluded from this smoke optimizer stream',
            manifest=dataset.manifest,records=reports),new_bank if bank is None else bank
    finally:
        agent.wm_diagnostic_forward=False;agent.valid_count_context=count_context
        agent.train(was_training);restore_rng(saved)
        if torch.cuda.is_available():torch.cuda.synchronize()
        agent.backbone.step_timing.collect()
