"""Independent coherent packed-kernel recipe; never a v1 execution migration."""
import copy
from starVLA.rl.qwen_world_performance.contracts import validate_recipe as validate_base

SCHEMA='world_coherent_packed_source_W3_execution_v1'


def protocol(scenes):
    if type(scenes)is not int or scenes not in (2,4):raise ValueError('Only explicitly implemented PACK2/PACK4')
    return dict(schema=SCHEMA,micro_scenes=scenes,candidate_group=16,stochastic_steps=[7,8,9],
        kernel='sample,old probability,current probability use identical scene order/padding/S*G BF16 layout',
        credit='original per-scene SafetyEP; all valid B*G population std once; history excluded',
        deployment='original genuine B1 ODE K10 shift5 seed2101 unchanged',
        source_ratio='original W3 dimension-mean/clamped transition proxy, not endpoint/joint density',
        initialization='same IL70 fresh Adam0 only; no native migration from single-scene kernels',
        cache='immutable GPU frozen-Qwen/reference feature cache; CPU backup on misses only')


def base_validation_recipe(cfg):
    value=copy.deepcopy(cfg);value['research'].pop('packed_execution',None)
    value['research']['performance']['microbatch_scenes']=1
    return value


def validate_recipe(cfg):
    p=cfg['research']['performance'];s=p['microbatch_scenes']
    if cfg['research'].get('packed_execution')!=protocol(s):raise ValueError('Packed numerical execution identity changed')
    if (p['algorithm']!='safetyep' or p['group_size']!=16 or p['inner_updates']!=4
            or any(x['rescue']or x['history']or x['B']%(8*s)for x in p['stages'])):
        raise ValueError('First FAST_PACK runtime is SafetyEP G16/M4 without auxiliary coupling')
    validate_base(base_validation_recipe(cfg))
    return cfg


def build_recipe(base,scenes,*,arm=None):
    cfg=copy.deepcopy(base);cfg['research']['arm']=arm or f'S0_FAST_PACK{scenes}'
    cfg['research']['performance']['microbatch_scenes']=scenes
    cfg['research']['packed_execution']=protocol(scenes)
    return validate_recipe(cfg)
