"""Explicit shared-engine extension, restricted to qualification until review."""
from pathlib import Path
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.qwen_world_compare.contracts import file_sha
from .actor import FreshAuxActor,optimizer_groups
from .config import read_qualification
from .protocol import EXPERIMENT,PROTOCOL_SHA256


class FreshAuxQualification:
    def read_recipe(self,args):
        self.args=args
        return read_qualification(args)

    @staticmethod
    def load_parent(path,device,*,readouts,binding):
        if not readouts:raise ValueError('Fresh D4 auxiliary qualification must explicitly request its new modules')
        # Parent remains the pure deployed policy; FreshAuxActor adds its OWN
        # modules only AFTER that policy is copied into the permanent reference.
        return load(path,device,readouts=False,binding=binding)

    def make_actor(self,model,backend,cfg):
        return FreshAuxActor(model,backend,cfg,qualification=self.args.qualification)

    optimizer_groups=staticmethod(optimizer_groups)

    @staticmethod
    def identity_fields(actor):
        return dict(experiment_id=EXPERIMENT,original_IL_auxiliary_restored=False,
            fresh_auxiliary_protocol_sha256=PROTOCOL_SHA256,
            fresh_auxiliary_initialization=actor.fresh_auxiliary_initialization,
            formal_admission='not_registered_qualification_only')

    @staticmethod
    def source_files():
        roots=('starVLA/rl/d46_fresh_aux','tools/d46_fresh_aux')
        return {str(p):file_sha(p)for root in roots for p in sorted(Path(root).rglob('*.py'))}
