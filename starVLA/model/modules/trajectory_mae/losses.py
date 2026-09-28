from starVLA.model.modules.foresight.losses import masked_regression


def reconstruction_loss(prediction, target, point_valid, xy_scale=20., global_count=None):
    return masked_regression(prediction/xy_scale,target/xy_scale,point_valid[...,None],
                             kind='smooth_l1',global_count=global_count)
