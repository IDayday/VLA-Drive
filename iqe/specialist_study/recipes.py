"""Strict stage recipes, resolved once per campaign rather than edited between cells."""
from pathlib import Path
import yaml
from iqe.io import atomic_json,file_hash
from iqe.contracts import require

SCHEMAS={
'pilot_expert_2x2':{'lr','weight_decay','global_batch_size','max_optimizer_steps','warmup_steps','lr_min_ratio','grad_clip_norm','schedule','validation_steps','checkpoint_every_steps','ema_decay','anchor_ratio','cells'},
'specialist_extension':{'max_extension_runs','select_on','minimum_scene_oracle_gain','minimum_mean_oracle_gain','force_three'},
'scorer_2x2':{'lr','weight_decay','global_batch_size','max_optimizer_steps','warmup_steps','lr_min_ratio','grad_clip_norm','schedule','validation_steps','checkpoint_every_steps','phase_switch_step','natural_phase2','comparison_phase2','value_weight','components_weight','ranking_weight','dropout','cells'},
'historical':{'include_Base_in_every_subset','bootstrap_repetitions','bootstrap_seed','use_official_labels_only','reuse_K4_prediction_for_subsets'},
'final_evaluation':{'bootstrap_repetitions','bootstrap_seed','margin_grid','margin_new_NC_budget','margin_new_DAC_budget','margin_high_zero_budget','automatic_ACTIVE'},
}

def load(config,name):
    path=Path(__file__).resolve().parents[2]/'configs/iqe/specialist_scorer'/(name+'.yaml')
    data=yaml.safe_load(path.read_text());require(set(data)==SCHEMAS[name],'unknown/missing stage recipe fields: '+name)
    atomic_json(config.root/'resolved_recipes'/(name+'.json'),{'source':str(path),'checksum':file_hash(path),'resolved':data},immutable=True)
    return data
