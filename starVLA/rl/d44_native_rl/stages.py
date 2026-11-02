"""D6 registered stages; modules exist from initialization, Adam is never reset."""
from dataclasses import dataclass,asdict


@dataclass(frozen=True)
class Stage:
    name:str
    global_scenes:int
    group:int
    text_lr:float
    film_lr:float=0.
    positive:bool=False
    post:bool=False
    endpoint:bool=False
    temporal:bool=False
    historical:bool=False
    mixture:bool=False


def stage_at(epoch,boundaries=(1.,4.,8.,12.)):
    if epoch<0 or len(boundaries)!=4 or not all(a<b for a,b in zip((0.,)+tuple(boundaries),boundaries)):
        raise ValueError('Ordered, positive D6 stage boundaries required')
    a,b,c,d=boundaries
    if epoch<a:return Stage('I',32,8,2e-6)
    if epoch<b:return Stage('II',64,16,2e-6,positive=True,mixture=True)
    if epoch<c:return Stage('III',64,16,1e-6,positive=True,post=True,endpoint=True,temporal=True,mixture=True)
    return Stage('IV'if epoch<d else'V',64,16,5e-7,film_lr=2e-6,positive=True,post=True,endpoint=True,temporal=True,historical=True,mixture=True)


def stage_recipe(cfg,epoch):
    stage=stage_at(epoch,tuple(cfg['stage_boundaries']))
    return dict(cfg,global_scenes=stage.global_scenes,group=stage.group,
        rescue_start_epoch=0.,enable_positive=stage.positive,enable_teacher_expansion=stage.positive,
        active_stage=asdict(stage)),stage
