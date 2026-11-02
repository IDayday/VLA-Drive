"""Raw valid scene exposure clock; inner reuse does not advance the clock."""
import math


def scene_lr(epoch, *, warmup=.08, hold=.60, end=4., floor=.05):
    if epoch<0:raise ValueError('Negative scene clock')
    if epoch<warmup:return max(epoch/warmup,1e-4)
    if epoch<=hold:return 1.
    if epoch>=end:return floor
    return floor+(1-floor)*.5*(1+math.cos(math.pi*(epoch-hold)/(end-hold)))
