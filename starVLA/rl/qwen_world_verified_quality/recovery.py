"""Keep the strict shared native validator; explicitly select S6 science."""
from starVLA.rl.qwen_world_performance import recovery as base
from starVLA.rl.qwen_world_performance.nominal_improvement import QUALITY_CONFIG
from starVLA.rl.qwen_world_performance.nominal_improvement import validate_material
from .recipe import validate_recipe,SCHEMA,PROTOCOL,QUALITY_PROTOCOL_SHA256
from . import calibration as C


def validate_boundary(state,window,identity,optimizer=None,scheduler=None,*,legacy=False,expected_rank=None):
    validate_recipe(identity['recipe'])
    if (identity.get('verified_quality_protocol')!=PROTOCOL or identity.get('quality_protocol_sha256')!=QUALITY_PROTOCOL_SHA256
            or identity.get('performance_implementation_protocols',{}).get('nominal')!=QUALITY_CONFIG
            or identity.get('performance_implementation_protocols',{}).get('calibration')!=C.PROTOCOL):
        raise ValueError('S6 native identity lacks its exact quality/alpha/start protocol')
    if window is not None:
        epoch=(state['valid_scene_exposures']-window['valid_global_scenes'])/identity['recipe']['research']['performance']['population']
        for leaf in window['samples']:validate_material(leaf,rank=window['rank'],expected_epoch=epoch,config=QUALITY_CONFIG)
    result=base.validate_boundary(state,window,identity,optimizer,scheduler,legacy=legacy,expected_rank=expected_rank,
        recipe_validator=validate_recipe,nominal_config=QUALITY_CONFIG,calibration_validator=C.validate_calibration_state)
    result.update(quality_schema=SCHEMA,quality_protocol_sha256=QUALITY_PROTOCOL_SHA256)
    return result


def validate_ranks(summaries):
    base.validate_ranks(summaries)
    if any(s.get('quality_schema')!=SCHEMA or s.get('quality_protocol_sha256')!=QUALITY_PROTOCOL_SHA256 for s in summaries):
        raise ValueError('S6 ranks disagree on the verified-quality science protocol')
    return summaries
