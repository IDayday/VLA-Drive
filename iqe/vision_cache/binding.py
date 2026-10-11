"""Explicit diagnostic binding of bounded continuation states, never formal S0."""
from ..contracts import require


def diagnostic_binding(state, mode, config):
    require(mode == 'smoke', 'diagnostic checkpoint binding is smoke-only')
    require(config['execution']['scientific_expansion'] is False,
            'diagnostic checkpoint cannot enable scientific expansion')
    require(state.get('mode') == 'performance_v2' and state.get('kind') == 'iqe_query_base',
            'diagnostic binding requires an actual bounded Query continuation')
    require(state.get('training_visual_source') in {'cached_fp32', 'online_fp32'} and
            state.get('inference_visual_source') == 'online_fp32' and
            state.get('visual_numeric_contract') == 'legacy_fp32', 'foreign numerical provenance')
    require(state.get('performance_identity', {}).get('production_cache_runtime') and
            state.get('base_provenance') and state.get('parent_checkpoint'),
            'diagnostic continuation lineage is incomplete')
    require(state.get('learned_driving_weights_loaded') is False,
            'unknown provenance is not a valid framework-only origin')
    return {'diagnostic_only': True, 'source_checkpoint_mode': state['mode'],
            'formal_base_qualified': False, 'permits_deployment': False}
