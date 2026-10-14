"""Execution revision: overlap/vector geometry plus refreshed frozen weight casts."""
from pathlib import Path

from starVLA.rl.constrained_flow_noise.contracts import file_sha,fingerprint
from . import train_efficient as implementation
from .frozen_weight_cast import install

_original_identity=implementation.source_identity


def source_identity():
    value=_original_identity()
    value['parent_execution_sha256']=value.pop('sha256')
    value['schema']='cfn_execution_overlap_geometry_frozen_cast_v2'
    value['sources'].update({name:file_sha(Path(__file__).with_name(name))
                             for name in ('train_efficient_cast.py','frozen_weight_cast.py')})
    value['frozen_casts']='Exact BF16-to-FP32 Linear weight copies only; original parameter dtypes intact; refreshed by every behavior load_state_dict; current actor unchanged'
    value['sha256']=fingerprint(value)
    return value


def main():
    original_configure=implementation.train.configure_rollout_graph
    def configure(actor,enabled):
        if enabled:install(actor)
        return original_configure(actor,enabled)
    implementation.source_identity=source_identity
    implementation.train.configure_rollout_graph=configure
    implementation.main()


if __name__=='__main__':main()
