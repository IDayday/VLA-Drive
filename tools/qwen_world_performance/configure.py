"""Render complete executable recipes from the established actor contract."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import yaml
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY, expected_scope_recipe
from starVLA.rl.qwen_world_performance.contracts import ROOT, SCHEMA, IL70_SHA256, EVALUATION_EPOCHS


def recipes(manifest, data_audit):
    old = Path('/mnt/project/qwen-world-safety-exploration-20261031/integrated_training_20261102/configs/INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC.yaml')
    oldcfg = yaml.safe_load(old.read_text())
    audit = json.loads(Path(data_audit).read_text())
    population = len(json.loads(Path(manifest).read_text())['tokens'])
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    common = expected_scope_recipe(MEMORY)
    common.update(max_steps=64560, learning_rate=5e-5, memory_learning_rate=5e-6,
        lr_scheduler_type='cosine', lr_warmup_steps=1, gradient_accumulation_steps=16,
        save_every_optimizer_updates=400, evaluate_updates=[], data_order='full NAVTRAIN epoch shuffle; valid padding mask; own-run TRAIN-only priorities')
    common['grpo']['sample']['group_size'] = 16
    common['research'] = dict(schema=SCHEMA, campaign_root=str(ROOT), arm='', policy_objective='ppo',
        data_policy='uniform_full', manifest_sha256=digest(manifest),
        observation_data=oldcfg['research']['observation_data'], scoring_cache_roots=oldcfg['research']['scoring_cache_roots'],
        structured_advantage='world_performance_safetyep_v1', NAVTEST_feedback_to_training=False,
        performance=dict(schema=SCHEMA, parent_model_sha256=IL70_SHA256, initialization='same_IL70_fresh_Adam0',
            cross_algorithm_state_transfer=False, population=population, data_audit=str(data_audit), data_audit_sha256=digest(data_audit),
            scenes_per_collection=128, group_size=16, inner_updates=4, algorithm='safetyep', microbatch_scenes=1,
            evaluation_epochs=EVALUATION_EPOCHS, minimum_epochs=3., initial_target_epochs=6., extension_epochs=[8,12,20],
            priority=dict(start_epoch=1., uniform_fraction=.60, nominal_hard_fraction=.25, regression_fraction=.15,
                max_scene_repeat_factor=4., refresh_every_scene_epochs=.5, stale_priority_max_epochs=1., physical_log_balance=True),
            schedule=dict(type='warmup_hold_cosine_floor', clock='valid_scene_exposures', warmup_epochs=.08,
                hold_until_epoch=.75, cosine_decay_end_epoch=4., minimum_lr_ratio=.05),
            nominal_rescue=dict(start_epoch=.5, anchor_budget_share=.05, min_improvement=.02, strict_high_min_pdms=.9,
                sigma_min=.1, sigma_max=.9, mc_samples=2, cfm_share=.75, ode_last_share=.25,
                calibration_slots=100, gradient_diagnostic_slots=500, adaptive_extra_G=True),
            stages=[dict(name='main', start_epoch=0., B=128, action_lr=5e-5, memory_lr=5e-6, rescue=False, history=False)]))
    for name, alg, lr, m in [('s0_large_safetyep','safetyep',5e-5,4), ('s0_low_lr','safetyep',3e-5,4),
            ('s1_nominal_rescue','nominal_rescue',5e-5,4), ('s2_conservative_awfm','awfm',3e-5,2),
            ('s3_large_fpo','fpo',3e-5,2), ('s4_temporal_rescue','temporal_rescue',5e-5,4),
            ('s5_performance_max','performance_max',5e-5,4)]:
        cfg = copy.deepcopy(common); p = cfg['research']['performance']
        cfg['research']['arm'] = name.upper(); p['algorithm'] = alg; p['inner_updates'] = m
        cfg['learning_rate'], cfg['memory_learning_rate'] = lr, lr/10
        cfg['grpo']['train']['num_inner_epochs'] = m
        cfg['max_steps'] = 807*20*m
        p['stages'][0].update(action_lr=lr, memory_lr=lr/10, rescue=alg in ('nominal_rescue','temporal_rescue'))
        if alg == 'awfm':
            p['awfm'] = dict(mc_samples=4, sigma_min=.2, sigma_max=.8, advantage_scale=.25, advantage_cap=5., lambda_floor=1.)
        if alg == 'fpo':
            p['fpo'] = dict(mc_samples=2, sigma_min=.2, sigma_max=.8, clip_lower=.01, clip_upper=.01,
                energy_reduction='mean_xyz_then_sum_8poses')
        if alg == 'performance_max':
            p['scenes_per_collection'] = 64; cfg['gradient_accumulation_steps'] = 8
            p['stages'] = [dict(name='coverage',start_epoch=0.,B=64,action_lr=5e-5,memory_lr=5e-6,rescue=False,history=False),
                dict(name='absorb',start_epoch=1.,B=128,action_lr=3e-5,memory_lr=3e-6,rescue=True,history=False),
                dict(name='retain',start_epoch=4.,B=128,action_lr=1e-5,memory_lr=1e-6,rescue=True,history=True),
                dict(name='refine',start_epoch=8.,B=128,action_lr=5e-6,memory_lr=5e-7,rescue=True,history=True)]
            p['history'] = dict(capacity=512,floor_share=.05,regression_extra_share=.20,min_pdms=.9)
            cfg['max_steps'] = (1614+807*19)*4
        yield name, cfg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--data-audit',required=True)
    ap.add_argument('--output',default='configs/qwen_world_performance_rl');a=ap.parse_args()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    for name,cfg in recipes(a.manifest,a.data_audit):
        (Path(a.output)/(name+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False))


if __name__=='__main__':main()
