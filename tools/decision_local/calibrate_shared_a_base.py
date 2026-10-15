"""One A-only shared-gradient calibration using the unchanged full S0 model.

This is not a driving update or a choice using development scores. It retains
the original future/current scale rule and interaction/FM scale rule, refitting
their statistics because neither may use the held-out correction population.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('legacy-source','population-contract','config','data','dino-root','dino-index','clip-root','interaction-root','output','campaign-root','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--samples',type=int,default=4);a=p.parse_args()
    if not 1<=a.samples<=8:raise ValueError('Fixed bounded A-only calibration')
    if Path(a.output).exists():raise FileExistsError('One immutable scale calibration')
    spec=importlib.util.spec_from_file_location('_a_only_runtime',Path(__file__).with_name('train_shared_a_base.py'))
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    helper,contract=runtime.prepare_runtime(Path(a.legacy_source),Path(a.population_contract))
    import torch,numpy as np
    from omegaconf import OmegaConf
    from starVLA.model.framework import build_framework
    from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
    from starVLA.dataloader.foresight_dataset import collate_training
    from tools.ddpolicy_vehicle.run_meter import metered_run
    from tools.foresight.calibrate_auxiliaries import weight_from_gradients
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    cfg=OmegaConf.load(a.config)
    if (cfg.foresight.planner_condition_mode!='action_only' or cfg.foresight.future_action_condition!='none' or
        cfg.foresight.future_target_type!='dino_sequence'):raise ValueError('This is the original S0, not S4/G')
    cfg.foresight.lambda_cur=cfg.foresight.lambda_fut=cfg.foresight.lambda_int=1.
    cfg.foresight.auxiliary_warmup=1
    di,ci,ii=[helper.read(Path(root)/'identity.json') for root in (a.dino_root,a.clip_root,a.interaction_root)]
    data=ActionVideoForesightDataset(a.data,candidate='C1',current=True,future=True,dino_root=a.dino_root,
        dino_index=a.dino_index,expected_dino=di['identity'],interaction_root=a.interaction_root,expected_interaction=ii['identity'],
        clip_root=a.clip_root,expected_clip=ci['identity'],future_type=cfg.foresight.future_target_type,
        expected_teacher=cfg.foresight.video_teacher_identity,allow_partial=False)
    chosen=[]
    for i in range(min(64,len(data))):
        row=data[i]
        if bool(row[1]['interaction_valid']) and bool(row[1]['future_clip_valid'].all()):chosen.append((i,row))
        if len(chosen)==a.samples:break
    if len(chosen)!=a.samples:raise ValueError('Insufficient fixed training-only complete auxiliary examples')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'A_only_S0_scale_calibration','real_optimizer_updates':0}) as (meter,_,save):
        torch.manual_seed(int(cfg.seed));model=build_framework(cfg).cuda().eval()
        language=model.qwen_vl_interface.model.model.language_model
        probes={'W':model.foresight_queries,'first_q':language.layers[0].self_attn.q_proj.weight,
                'last_q':language.layers[-1].self_attn.q_proj.weight}
        rows=[];started=time.monotonic()
        for i,sample in chosen:
            obs,targets=collate_training([sample]);result=model.forward_train(obs,targets,completed_updates=1,
                noise_generator=torch.Generator(device='cuda').manual_seed(9042),
                time_generator=torch.Generator(device='cuda').manual_seed(10042))
            record={'token':data.index[i]['token'],'raw_losses':{k:float(v.detach()) for k,v in result['losses'].items()}}
            if set(result['losses'])!={'ego_fm','current_dino','future_clip','interaction'}:raise ValueError('Full S0 tasks missing')
            for name,loss in result['losses'].items():
                gradients=torch.autograd.grad(loss,tuple(probes.values()),retain_graph=True,allow_unused=True)
                norms={k:float(g.float().norm()) if g is not None else 0. for k,g in zip(probes,gradients)}
                if not all(np.isfinite(v) and v>0 for v in norms.values()):raise ValueError('Missing/nonfinite shared gradients: '+name)
                record[name]={**norms,'language':float(np.hypot(norms['first_q'],norms['last_q'])),'norm':float(np.linalg.norm(list(norms.values())))}
            result['loss'].backward()
            if any(not torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):raise FloatingPointError('Nonfinite full backward')
            model.zero_grad(set_to_none=True);del result,gradients
            rows.append(record);meter['inference_scenes']=len(rows);save()
        means={k:{p:float(np.mean([r[k][p] for r in rows])) for p in ('W','first_q')} for k in ('current_dino','future_clip')}
        future=np.sqrt(np.prod([means['current_dino'][p]/means['future_clip'][p] for p in ('W','first_q')]))
        interaction=weight_from_gradients(rows,'interaction')
        obs,_=collate_training([chosen[0][1]])
        with torch.no_grad():
            noise=torch.zeros(1,8,4,device='cuda');q0=model.predict_action(obs,initial_noise=noise)
            poisoned={**obs[0],'future_GT':None,'repair_target':torch.full((8,4),float('nan')),'GT_map':'label_only'}
            q_poison=model.predict_action([poisoned],initial_noise=noise)
            torch.testing.assert_close(q0,q_poison,rtol=0,atol=0)
        report=dict(passed=True,source_sha=helper.source_sha(ROOT),legacy_source_sha=helper.source_sha(a.legacy_source),
            population_contract_sha256=helper.digest(a.population_contract),teacher_identity=ii['teacher_identity'],
            action_codec_identity=contract['action_codec_identity'],initialization='generic Qwen + random driving/W/heads; no driving checkpoint',
            lambda_cur=1.,lambda_fut=float(future),lambda_int=interaction['weight'],interaction_rule=interaction,
            future_rule='geometric mean of current/future gradients at W and first Q projection',
            interaction_rule_description='original median 0.25*FM/interaction over W + first/last Q projections',
            real_optimizer_updates=0,rows=rows,future_label_invariance=True,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),wall_after_model_load_seconds=time.monotonic()-started)
        helper.atomic_json(a.output,report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
