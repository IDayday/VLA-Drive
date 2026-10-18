"""Small native actor/observation/codec hooks for the existing audited trainer."""
import json
from pathlib import Path
import torch
from .contracts import ROOT,require_dispatch,source_files,file_sha,configure_native_numerics
from .recipe import load_recipe
from .loading import load_il,inspect_il
from .data import RawSceneData
from .conditions import FrozenConditions
from .actor import NativeRecipeActor
from .collection import collect
from starVLA.rl.fm_transfer.source_process import SourceTokenStream
from scripts.analysis.simwam_recipe.train import parameter_hash


class NativeRuntime:
    require_dispatch=staticmethod(require_dispatch)
    source_files=staticmethod(source_files)
    collect=staticmethod(collect)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--il-checkpoint',required=True)
        parser.add_argument('--parent-selection',help='Sealed DEV-only parent selection receipt')
        parser.add_argument('--data',default=str(ROOT/'data/TRAIN_v3'))

    def load_recipe(self,args):
        configure_native_numerics()
        cfg=load_recipe(args.config)
        self.parent=inspect_il(args.il_checkpoint,qualification=args.qualification)
        if not args.qualification:
            from .selection import validate_selection
            self.selection=validate_selection(args.parent_selection,self.parent)
        else:self.selection=None
        return cfg

    def make_scene_stream(self,cfg,args,world):
        if Path(args.manifest).resolve()!=Path('/mnt/project/simwam-source-process-20261018/source_manifest.json'):
            raise ValueError('The same locked public hard TRAIN list is required')
        return SourceTokenStream(args.manifest,world)

    def make_actor(self,cfg,args,device):
        model,self.parent=load_il(args.il_checkpoint,device=device,qualification=args.qualification)
        data=RawSceneData(args.data,cohort='TRAIN')
        self.conditions=FrozenConditions(model,data,self.parent['files']['model.pt'])
        actor=NativeRecipeActor(model.policy.action_expert,cfg,self.conditions)
        if args.qualification:
            actor.record_sampling_terms=True
            self.observation_before={k:parameter_hash(getattr(self.conditions,k),trainable=False) for k in ['qwen','provider']}
        return actor,None,None

    def make_conditions(self,actor):
        # Loading a complete 13GB IL checkpoint can differ by >90s across
        # ranks on shared storage. Finish startup on the main 20-minute NCCL
        # group before entering the existing short Gloo transaction protocol.
        # No chain/RNG/optimizer operation occurs in this synchronization.
        if torch.distributed.is_initialized():torch.distributed.barrier()
        return self.conditions

    def recipe_identity(self,cfg,args,condition_sha):
        return dict(schema='native_qwen_sourceflow_RL_v1',recipe=cfg,
            upstream_commit='1433ee7e06a62bd8b89da74455df0d82bda7b726',
            SFT_sha256=self.parent['files']['model.pt'],IL_parent=self.parent,parent_selection=self.selection,
            hard_manifest_sha256=file_sha(args.manifest),projected_condition_identity=condition_sha,
            policy_ratio='source relative-clamped dimension-mean stochastic-transition surrogate',
            deployment='native Qwen full tokens, source 8x3, single candidate shift5 ODE10')

    def finish_qualification(self,result):
        after={k:parameter_hash(getattr(self.conditions,k),trainable=False) for k in ['qwen','provider']}
        if after!=self.observation_before:raise ValueError('Frozen native Qwen/provider changed during source RL')
        result.update(frozen_observation_before=self.observation_before,frozen_observation_after=after,
            prefill_calls_this_invocation=self.conditions.prefill_calls,prefill_seconds_this_invocation=self.conditions.prefill_seconds,
            prefill_cost_included_in_rollout=True)

    def validate_collection(self,rollout):
        check=rollout['qualification_sampling_check']
        if check and check['status']!='PASS':
            raise ValueError('Actual sampling/recompute mismatch; full partial chain retained: '+repr(check))
