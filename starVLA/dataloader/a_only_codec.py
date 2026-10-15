"""Explicit action/state affine transform fitted only on the new base's A logs.

The original encoded label files remain read-only. This process-local adapter
does not patch a shared environment or change the historical codec defaults.
"""
import math

OLD_MEAN=(10.172484,.360762)
OLD_STD=(8.805105,2.277741)


def validate_codec(codec):
    if codec['schema']!='a_only_action_affine_v1' or codec['fit_role']!='A_train':
        raise ValueError('Only an identified A-only action affine is accepted')
    values=codec['mean_xy_m']+codec['std_xy_m']
    if len(values)!=4 or not all(math.isfinite(x) for x in values) or min(codec['std_xy_m'])<=0:
        raise ValueError('Invalid action normalization')


def install_legacy_codec(codec):
    """Call before constructing current readers, training, or deployment."""
    validate_codec(codec)
    from starVLA.dataloader import foresight_dataset
    foresight_dataset.X_MEAN,foresight_dataset.Y_MEAN=codec['mean_xy_m']
    foresight_dataset.X_STD,foresight_dataset.Y_STD=codec['std_xy_m']


def recode_historical_targets(encoded,codec):
    validate_codec(codec)
    result=encoded.clone()
    physical=encoded[...,:2]*encoded.new_tensor(OLD_STD)+encoded.new_tensor(OLD_MEAN)
    result[...,:2]=(physical-encoded.new_tensor(codec['mean_xy_m']))/encoded.new_tensor(codec['std_xy_m'])
    return result
