"""Thin independent S6 runtime; common native Trainer and SafetyEP unchanged."""
import copy
import torch.distributed as dist
from pathlib import Path

from starVLA.rl.qwen_world_postcot.contracts import file_sha
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime,ComponentCapture
from starVLA.rl.qwen_world_performance import nominal_improvement as N
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from .recipe import validate_recipe,PROTOCOL,QUALITY_PROTOCOL_SHA256


class VerifiedQualityLoss(N.NominalLossMixin,ComponentCapture,ResearchLoss):
    nominal_config=N.QUALITY_CONFIG


class VerifiedQualityRuntime(PerformanceRuntime):
    recipe_validator=staticmethod(validate_recipe)

    @staticmethod
    def add_arguments(parser):
        PerformanceRuntime.add_arguments(parser)
        parser.set_defaults(frozen_reference_cache=True)

    @staticmethod
    def source_files():
        files=PerformanceRuntime.source_files()
        for folder in ('starVLA/rl/qwen_world_verified_quality','tools/qwen_world_verified_quality'):
            for path in Path(folder).rglob('*.py'):files[str(path)]=file_sha(path)
        return dict(sorted(files.items()))

    def load_recipe(self,args):
        if getattr(args,'qualification_force_auxiliary',False):
            raise ValueError('S6 starts naturally at epoch0; never inject an artificial gate')
        return super().load_recipe(args)

    def make_actor(self,cfg,args,device):
        actor,a,b=super().make_actor(cfg,args,device)
        actor._loss=VerifiedQualityLoss(cfg)
        return actor,a,b

    def bind_training_state(self,state,rollout,identity):
        super().bind_training_state(state,rollout,identity)
        auxiliary=state['performance_auxiliary']
        if 'nominal_improvement' not in auxiliary:
            if state['updates'] or state['collections']:raise ValueError('S6 native state cannot silently reset calibration')
            auxiliary['nominal_improvement']=N.new_calibration_state()
        N.validate_calibration_state(auxiliary['nominal_improvement'])
        self.actor._loss.auxiliary_state=auxiliary['nominal_improvement']

    def _attach_auxiliary(self,actor,leaf,service,rank):
        N.attach_material(actor,leaf,self.conditions,service,
            exposure_epoch=self.training_state['valid_scene_exposures']/self.spec['population'],rank=rank,
            config=N.QUALITY_CONFIG)

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['performance_implementation_protocols']['nominal']=copy.deepcopy(N.QUALITY_CONFIG)
        value.update(verified_quality_protocol=copy.deepcopy(PROTOCOL),quality_protocol_sha256=QUALITY_PROTOCOL_SHA256)
        return value

    def validate_boundary(self,*args,**kwargs):
        from .recovery import validate_boundary
        return validate_boundary(*args,**kwargs)

    def validate_ranks(self,summaries):
        from .recovery import validate_ranks
        return validate_ranks(summaries)
