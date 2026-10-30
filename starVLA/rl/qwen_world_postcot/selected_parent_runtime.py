"""Explicit user-selected IL parent; unchanged qualified memory GRPO engine.

This schema records historically observed Test selection. It neither relaxes
the original IL40k schema nor pretends to be the four-architecture DEV rule.
"""
import json
from pathlib import Path

import yaml

from .contracts import file_sha, fingerprint
from .loading import inspect_checkpoint
from .scope_contracts import MEMORY, expected_scope_recipe
from .scope_runtime import ScopeRuntime


SCHEMA = 'world_postcot_user_best_test_IL_parent_v1'
INSTRUCTION = '再起一个RL训练任务，基于现在最高分的IL模型，以及目前尝试的比较好的RL训练 方法，再跑一版。'


def expected_recipe():
    cfg = expected_scope_recipe(MEMORY)
    cfg['reference_observation'] = 'permanently frozen explicitly selected IL parent Qwen and memory'
    return cfg


def validate_selection(path, parent, config_path, manifest_path):
    receipt = json.loads(Path(path).read_text())
    body = dict(receipt)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed explicit selected-parent receipt')
    if (receipt.get('schema') != SCHEMA or receipt.get('user_instruction') != INSTRUCTION
            or receipt.get('Test_observed') is not True
            or receipt.get('selection_method') != 'user_requested_best_completed_QwenWorld_Test_as_of_registration'
            or receipt.get('training_seed') != 42 or receipt.get('evaluation_seed') != 2101
            or receipt.get('adaptation') != MEMORY):
        raise ValueError('Not the separately authorized selected-parent experiment')
    selected = receipt['selected']
    expected = dict(checkpoint=parent['path'], updates=parent['seal']['updates'],
                    model_sha256=parent['model_sha256'],
                    files_sha256=parent['seal']['files_sha256'],
                    identity_sha256=fingerprint(parent['identity']))
    if any(selected.get(k) != v for k, v in expected.items()):
        raise ValueError('Selected native parent changed')
    if (receipt['training_recipe_sha256'] != file_sha(config_path)
            or receipt['hard_manifest_sha256'] != file_sha(manifest_path)):
        raise ValueError('Selected-parent recipe or TRAIN membership changed')
    proof = receipt['selection_evidence']
    for field in ('summary', 'registration'):
        if file_sha(proof[field]) != proof[field+'_sha256']:
            raise ValueError('Changed completed Test selection evidence')
    summary = json.loads(Path(proof['summary']).read_text())
    reg = json.loads(Path(proof['registration']).read_text())
    for value in (summary, reg):
        check = dict(value)
        if check.pop('sha256', None) != fingerprint(check):
            raise ValueError('Unsealed selection evidence')
    if (summary.get('status') != 'COMPLETE' or summary.get('scenes') != 12146
            or summary.get('invalid_scenes') != 0 or summary.get('seed') != 2101
            or summary.get('PDMS') != selected['PDMS']
            or summary.get('model_sha256') != reg.get('model_sha256')
            or summary.get('protocol_sha256') != reg.get('protocol_sha256')
            or reg.get('IL_parent_model_sha256') != parent['model_sha256']
            or reg.get('checkpoint_files_sha256') != parent['seal']['files_sha256']):
        raise ValueError('Score does not identify the selected complete IL')
    return receipt


class SelectedParentMemoryRuntime(ScopeRuntime):
    @staticmethod
    def source_files():
        files = ScopeRuntime.source_files()
        for name in ('tools/qwen_world_postcot/train_selected_parent_rl.py',
                     'tools/qwen_world_postcot/selected_parent_owner.py',
                     'tools/qwen_world_postcot/job_worker.py'):
            files[name] = file_sha(name)
        return dict(sorted(files.items()))

    def load_recipe(self, args):
        from starVLA.rl.flow_grpo.reproducibility import configure_numerics
        configure_numerics()
        cfg = yaml.safe_load(Path(args.config).read_text())
        if cfg != expected_recipe():
            raise ValueError('Selected-parent memory recipe differs from the registered contract')
        self.cfg = cfg
        self.args = args
        self.parent = inspect_checkpoint(args.il_checkpoint)
        identity = self.parent['identity']
        if (identity.get('uses_rl') is not False or identity.get('uses_metric_supervision') is not True
                or identity.get('qualification') is not False):
            raise ValueError('Selected parent must be a complete formal World IL')
        if not args.parent_selection:
            raise ValueError('Both qualification and formal stages require explicit parent provenance')
        selected = validate_selection(args.parent_selection, self.parent, args.config, args.manifest)
        self.selection = None if args.qualification else selected
        return cfg

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value.update(reference_observation='permanently frozen selected IL parent Qwen, scene and predictive providers',
                     selected_parent_updates=self.parent['seal']['updates'],
                     parent_selection_protocol=SCHEMA,
                     Test_used_for_user_requested_parent_selection=True)
        return value
