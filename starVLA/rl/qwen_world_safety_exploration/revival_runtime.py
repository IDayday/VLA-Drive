"""Independent revival composition over unchanged original algorithm kernels."""
import copy

from .revival_recipe import validate_revival_recipe,SCHEMA
from .standardized_runtime import StandardizedRuntimeMixin
from .group_runtime import GroupRuntime
from .credit import GroupCredit
from .quality import QualityDiagnosticsMixin
from .dense_credit import DenseCredit,DenseRuntimeMixin,validate_dense_boundary
from .saturated_credit import SaturatedCredit,SaturatedRuntimeMixin,validate_saturated_boundary
from .curriculum import CurriculumRuntimeMixin


class RevivalRuntimeMixin:
 recipe_validator=staticmethod(validate_revival_recipe)
 def make_credit(self,cfg):
  validate_revival_recipe(cfg);role=cfg['research']['standard_role'];g=cfg['grpo']['sample']['group_size']
  if role=='dense':return DenseCredit(g)
  if role=='saturated':return SaturatedCredit(g)
  return GroupCredit(g)
 def recipe_identity(self,cfg,args,condition_sha):
  v=super().recipe_identity(cfg,args,condition_sha)
  v['revival_protocol']=dict(schema=SCHEMA,role=cfg['research']['standard_role'],
   original_method='Same original G16/DenseDAC/Saturated/onlinePriority mathematics and policy; only new independent warm1000 fixedcos50k study',
   scope='Fresh sameIL70/Adam/seed42/update0, W3source PPO/anchor/LRpeak unchanged',
   comparison='External F.SAFETY_G8_CONTROL at same updates; G16 uses2x scoredqueries, do not confuse equalupdates with equalcandidatebudget',
   data='Active retains original rank-local capped onlinepriority stream/privateRNG and full native state; no historicalevaluation feedback')
  return v


def revival_runtime(cfg):
 validate_revival_recipe(cfg);role=cfg['research']['standard_role'];extra=[];attrs={}
 if role=='dense':extra=[DenseRuntimeMixin];attrs['credit_boundary_validator']=staticmethod(validate_dense_boundary)
 elif role=='saturated':extra=[SaturatedRuntimeMixin];attrs['credit_boundary_validator']=staticmethod(validate_saturated_boundary)
 mixins=[RevivalRuntimeMixin,StandardizedRuntimeMixin,QualityDiagnosticsMixin]+extra+[GroupRuntime]
 if role=='active_data':mixins=[CurriculumRuntimeMixin]+mixins
 return type('RevivalFreshRuntime',tuple(mixins),attrs)()
