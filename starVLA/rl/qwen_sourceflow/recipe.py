"""Two complete source recipes with an explicit native model/observation contract."""
import copy
import json
from pathlib import Path
import yaml
from starVLA.rl.simwam_recipe.contracts import load_recipe as legacy_recipe
from .contracts import fingerprint, file_sha


def expected_recipe(variant):
    if variant not in ('R_SOURCE', 'R_CANONICAL'):
        raise ValueError('Only the two registered reward versions')
    repo = Path(__file__).resolve().parents[3]
    cfg = copy.deepcopy(legacy_recipe(repo/'configs/flow_rl/simwam_source_ddp.yaml'))
    cfg.update(variant=variant, max_steps=50000, action_shape=[8, 3],
        evaluate_updates=[0, 5000, 10000, 20000, 35000, 50000],
        anchor_mapping='source_native_xy_heading',
        model_family='QwenSourceFlowVLA', observation='native_Qwen_full_tokens',
        frozen_prefill_autocast='bfloat16', action_autocast=False,
        data_order='locked public source hard list intersect TRAIN; source epoch order and padding',
        deployment='single_candidate_source_shift5_ODE_K10',
        logical_budget=dict(fresh_global_rollouts=12500, scenes=100000, candidates=800000))
    cfg['grpo']['train']['kl_dim_weights'] = [.2, 1., 1.]
    cfg['grpo']['reward']['progress_weight'] = 10. if variant == 'R_SOURCE' else 5.
    cfg['grpo']['eval']['enabled'] = False
    cfg['grpo']['vis']['enabled'] = False
    return cfg


def load_recipe(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg != expected_recipe(cfg.get('variant')):
        raise ValueError('Native source recipe differs from its entire registered contract')
    return cfg
