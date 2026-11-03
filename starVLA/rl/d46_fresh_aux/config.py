"""Qualification-only prototype; live v2 policy contract is never amended here."""
import copy
from pathlib import Path
import yaml
from starVLA.rl.d44_native_rl.config import defaults
from starVLA.rl.d44_native_rl.parent import registration,audit_export,D46_ROOT
from starVLA.rl.qwen_world_compare.contracts import file_sha
from .protocol import EXPERIMENT,PROTOCOL,PROTOCOL_SHA256


def build_recipe(*,post_probe=False):
    cfg=defaults('D4');path=D46_ROOT/'PARENT_REGISTRATION_v2.json'
    binding=dict(path=str(path),file_sha256=file_sha(path));value=registration(binding)
    cfg.update(parent='D_IL46K',parent_checkpoint=value['export']['path'],parent_registration=binding,
        initialization='strict_IL46_policy_then_current_only_fresh_R16_reward_heads_seed42',
        permanent_reference='same_verified_IL46_deployment_policy_without_R_or_readouts',
        experiment_id=EXPERIMENT,fresh_auxiliary=copy.deepcopy(PROTOCOL),
        fresh_auxiliary_protocol_sha256=PROTOCOL_SHA256,
        admission_status='PROTOTYPE_QUALIFICATION_ONLY_NO_FORMAL_CONTRACT')
    if post_probe:
        cfg.update(post_start_epoch=0.,qualification_overrides=dict(post_start_epoch=0.,formal_post_start_epoch=1.,
            reason='exercise fresh R/head gradients and active-post pending restore in a finite qualification; never formal initialization'))
    return cfg


def read_qualification(args):
    if not args.qualification:raise ValueError('Fresh auxiliary prototype has no formal registration or GPU qualification')
    cfg=yaml.safe_load(Path(args.config).read_text());expected=build_recipe(post_probe='qualification_overrides'in cfg)
    if cfg!=expected:raise ValueError('Fresh auxiliary prototype recipe changed outside its declared protocol')
    audit_export(cfg['parent_checkpoint'],verify=False,binding=cfg['parent_registration'])
    return cfg
