#!/usr/bin/env python3
"""Resolve the actual paired V1.1 primary configs, without starting training."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scripts.audit_formal_config_pair import audit_formal_config_pair


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shared-init',required=True)
    parser.add_argument('--input-cache',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--vlm-audit',type=Path,default=ROOT/'reports/planreg_wm_v1/formal_vlm_initialization_audit.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit=json.loads(args.vlm_audit.read_text())
    os.environ['PLANREG_SHARED_INIT']=args.shared_init
    os.environ['PLANREG_INPUT_CACHE']=args.input_cache
    configs=[]
    args.output.mkdir(parents=True)
    for variant,suffix in [('base','base'),('driving_vqa','vqa')]:
        vlm=audit[variant]
        os.environ.update(PLANREG_BASE_VLM_PATH=audit['base']['checkpoint_path'],
                          PLANREG_VQA_VLM_PATH=audit['driving_vqa']['checkpoint_path'],
                          PLANREG_VLM_CHECKPOINT_SHA256=vlm['checkpoint_sha256'],
                          PLANREG_VLM_CONFIG_SHA256=vlm['config_sha256'],
                          PLANREG_OUTPUT_DIR=str(args.output/('future_formal_'+variant)),
                          PLANREG_EXPERIMENT_NAME='formal_v1p1_'+variant)
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/'navsim/planning/script/config/training')):
            cfg=compose(config_name='formal_planreg_wm_v1p1_training',overrides=[
                'agent=episode_drive_planreg_wm_v1p1_'+suffix,'experiment_uid=v1p1_pair_audit'])
        resolved=OmegaConf.to_container(cfg,resolve=True)
        configs.append(resolved)
        OmegaConf.save(OmegaConf.create(resolved),args.output/(variant+'.yaml'))
    report=audit_formal_config_pair(*configs)
    report['scope']='Resolved configuration parity only; no layout/convergence promotion'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
