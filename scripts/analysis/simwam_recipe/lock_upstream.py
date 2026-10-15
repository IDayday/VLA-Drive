"""Lock the actual Hydra recipe and vendor only the required source algorithms."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT = '1433ee7e06a62bd8b89da74455df0d82bda7b726'
FILES = ['configs/train_grpo.yaml', 'configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml',
         'configs/model/simwam_navsim.yaml', 'configs/data/navsim.yaml',
         'scripts/train_navsim_grpo_zero1_torchrun.sh', 'scripts/train_grpo.py',
         'src/simwam/runtime_grpo.py', 'src/simwam/trainer_grpo.py',
         'src/simwam/models/wan22/simwam_grpo.py', 'src/simwam/models/wan22/lora.py',
         'src/simwam/models/wan22/finetune.py',
         'src/simwam/models/wan22/schedulers/scheduler_continuous.py',
         'src/simwam/datasets/navsim/pdm_reward.py', 'src/simwam/utils/samplers.py',
         'scripts/ds_configs/ds_zero1_config.json', 'README.md', 'LICENSE']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_class(path, name, methods, imports):
    """Keep function source text, decorators and bodies byte-for-byte intact."""
    source = path.read_text(); lines = source.splitlines(keepends=True)
    cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == name)
    selected = []
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            selected.append(''.join(lines[start-1:node.end_lineno]))
    assert len(selected) == len(methods)
    return ('# Copyright (c) 2026 The SimWAM Authors. MIT; see LICENSE.SimWAM.\n'
            '# Source: H-EmbodVis/SimWAM@'+COMMIT+' / '+str(path).split('/SimWAM/')[-1]+'\n'
            '# Extracted methods are unmodified; only imports/class composition are isolated.\n'
            'from __future__ import annotations\n'+imports+'\n\nclass '+name+':\n'+ '\n\n'.join(selected)+'\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', required=True); p.add_argument('--flow-grpo', required=True)
    p.add_argument('--output', default='reports/ddp_simwam_recipe_20261015')
    a = p.parse_args(); root = Path(a.source).resolve(); out = Path(a.output)
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() != COMMIT:
        raise ValueError('SimWAM commit mismatch')
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise ValueError('Upstream checkout must be unmodified')
    out.mkdir(parents=True,exist_ok=True)
    # Only environment path substitutions; no recipe command-line overrides.
    env = dict(NAVSIM_DEVKIT_ROOT=str(root/'navsim'), NAVSIM_LOG_PATH='SOURCE_NAVTRAIN_LOGS',
               NAVSIM_SENSOR_BLOBS_PATH='SOURCE_NAVTRAIN_SENSORS')
    for k,v in env.items(): os.environ[k] = v
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
        cfg = compose(config_name='train_grpo',overrides=['task=navsim_grpo_action_pdm_384x672_flowgrpo_lora'])
        cfg.output_dir = './runs/grpo/RESOLVED_TIME_NOT_A_HYPERPARAMETER'
        resolved = OmegaConf.to_container(cfg,resolve=True)
    OmegaConf.save(OmegaConf.create(resolved),out/'resolved_source_recipe.yaml')
    vendor = Path('starVLA/rl/simwam_recipe/vendor')
    vendor.mkdir(parents=True,exist_ok=True)
    copies = {'lora.py':'src/simwam/models/wan22/lora.py',
              'scheduler_continuous.py':'src/simwam/models/wan22/schedulers/scheduler_continuous.py',
              'samplers.py':'src/simwam/utils/samplers.py','LICENSE.SimWAM':'LICENSE'}
    for dest,src in copies.items(): (vendor/dest).write_bytes((root/src).read_bytes())
    methods = ['configure_grpo','_configure_action_grpo','_resolve_action_lora_target_modules',
               '_stoch_step_indices','_normalize_stoch_idx','_as_weight_list','kl_weight_matrix',
               '_step_std','_sde_mean_std','_merged_span_mean_std','sample_action_chain',
               'action_chain_logprobs','_randn']
    imports = ('import copy\nimport math\nimport logging\nfrom typing import Any, Dict, Optional\n'
               'import torch\nfrom torch.distributions import Normal\nlogger=logging.getLogger(__name__)\n'
               '# Video is deliberately not a supported entry; action math is unchanged.\n'
               'class VideoGRPOMixin: pass\n')
    (vendor/'action_math.py').write_text(extract_class(root/'src/simwam/models/wan22/simwam_grpo.py','ActionGRPOMixin',methods,imports))
    trainer_methods = ['_pg_group_mask','_compute_advantages','_effective_kl_beta','_policy_loss','_build_scheduler']
    (vendor/'trainer_math.py').write_text(extract_class(root/'src/simwam/trainer_grpo.py','SimWAMGRPOTrainer',trainer_methods,
        'import torch\nfrom torch.optim.lr_scheduler import ConstantLR,CosineAnnealingLR,LinearLR,SequentialLR\n'))
    flow = Path(a.flow_grpo)
    lock = dict(schema='ddp_simwam_upstream_lock_v1',simwam=dict(url='https://github.com/H-EmbodVis/SimWAM',commit=COMMIT,
        files={s:sha(root/s) for s in FILES},license='MIT; separate retained third-party headers'),
        flow_grpo=dict(url='https://github.com/yifan123/flow_grpo',commit=subprocess.check_output(['git','-C',str(flow),'rev-parse','HEAD'],text=True).strip(),
                      files={'flow_grpo/diffusers_patch/sd3_sde_with_logprob.py':sha(flow/'flow_grpo/diffusers_patch/sd3_sde_with_logprob.py')},reuse='explanation only; no image training or weights'),
        vendor={str(x):sha(x) for x in vendor.iterdir() if x.is_file()},
        extraction=dict(action_methods=methods,trainer_methods=trainer_methods),hydra_cli=['task=navsim_grpo_action_pdm_384x672_flowgrpo_lora'],
        path_only_environment=env,model_weights_downloaded=False)
    (out/'upstream_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(status='LOCKED',commit=COMMIT,group=resolved['grpo']['sample']['group_size'],max_steps=resolved['max_steps'])))


if __name__ == '__main__': main()
