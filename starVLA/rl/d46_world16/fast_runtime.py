"""Opt-in world16 execution flags, after strict native restore/B1 parity.

No model/parent/conditions code is changed. Every enabled path is bound to the
saved typed upgrade profile; an unavailable optimization is rejected explicitly.
"""
import copy
from .runtime import World16Extension
from .execution_upgrade import FLAG_NAMES,flags,validate_identity


class FastWorld16Extension(World16Extension):
    def execution_profile(self):
        saved=self.saved_identity
        if 'execution_upgrade'not in saved:return dict.fromkeys(FLAG_NAMES,False)
        meta=validate_identity(saved);profile=flags(meta['flags'])
        if profile['no_grad_prefix_cache']:
            raise ValueError('no_grad_prefix_cache is not implemented/qualified in this entry')
        if profile['async_scoring']:
            raise ValueError('async_scoring requires a separate collection hook; not enabled in this entry')
        return profile

    def read_recipe(self,args):
        cfg=super().read_recipe(args)
        self.execution_profile()  # Fail before loading actor or acquiring model memory.
        return cfg

    def identity(self,actual):
        value=copy.deepcopy(actual)
        upgrade=self.saved_identity.get('execution_upgrade')
        if upgrade is not None:
            if 'execution_upgrade'in value and value['execution_upgrade']!=upgrade:
                raise ValueError('Actual execution upgrade differs from saved profile')
            value['execution_upgrade']=copy.deepcopy(upgrade)
        self.execution_profile()
        # Existing strict recipe/parent/scope/current conditions/source check.
        return super().identity(value)

    def configure_actor(self,actor):
        """Called once after restore parity, before any live collection/update.

        Invariant-cache closures must not be installed before the existing B1
        parity helper deep-copies a model. Installation changes no Parameters.
        """
        profile=self.execution_profile()
        if getattr(self,'_configured_actor',None)is not None:
            raise ValueError('Fast execution actor has already been configured')
        caches=[]
        try:
            if profile['rotary_cache']:
                from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache
                current=actor.action_expert.decision;reference=actor.ref_action_expert.decision
                if current is reference:raise ValueError('Current/reference decision graphs must be independent')
                for decision in(current,reference):
                    caches.append(install_invariant_cache(decision,rotary=True,additive=True))
        except BaseException:
            for cache in reversed(caches):cache.close()
            raise
        actor.execution_fast_prefix=profile['fast_prefix']
        actor.execution_fuse_fm_mc=profile['fuse_fm_mc']
        self._configured_actor=actor;self._execution_caches=tuple(caches)
        self.actor_execution_profile=copy.deepcopy(profile)

    def close_execution_caches(self):
        """Optional CPU/test cleanup; worker exit naturally releases the actor."""
        for cache in reversed(getattr(self,'_execution_caches',())):cache.close()
        self._execution_caches=()
