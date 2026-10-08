"""Explicit FP32-only migration of legacy precision decorators.

The structured geometry wrapper runs with FP32 parameters and autocast disabled.
Legacy auto_fp16/force_fp32 decorators are inactive in that reference mode. Mixed
precision inside these imported geometry classes is deliberately rejected.
"""
from functools import wraps


def _fp32_geometry_decorator(*args, **kwargs):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *positional, **named):
            if getattr(self, 'fp16_enabled', False):
                raise RuntimeError('Standalone geometry port supports explicit FP32 only')
            return function(self, *positional, **named)
        return wrapped
    return decorate


auto_fp16 = _fp32_geometry_decorator
force_fp32 = _fp32_geometry_decorator
