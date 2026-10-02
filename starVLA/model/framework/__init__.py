"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY


try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Import only the selected framework: Action-Only must not import video/depth.

def build_framework(cfg, accelerator=None):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "QwenOFT":
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg, accelerator)
    elif cfg.framework.name == "QwenFast":
        from starVLA.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)
    elif cfg.framework.name == "QWenGROOT":
        from starVLA.model.framework.QWenGROOT import Qwenvl_GROOT
        return Qwenvl_GROOT(cfg)
    elif cfg.framework.name == "QWenGROOT":
        from starVLA.model.framework.QWenPI import Qwenvl_PI
        return Qwenvl_PI(cfg)

    elif cfg.framework.name == "QwenVision":
        from starVLA.model.framework.QWenVision import Qwenvl_Vision
        return Qwenvl_Vision(cfg)

    
    if cfg.framework.name == "DDPForesight":
        from .DDPForesight import DDPForesight
        return DDPForesight(cfg, accelerator)
    if cfg.framework.name == "DDPFullForesight":
        from .ddp_full_foresight import DDPFullForesight
        return DDPFullForesight(cfg, accelerator)
    if cfg.framework.name == "DDPActionVideoForesight":
        from .ddp_action_video_foresight import DDPActionVideoForesight
        return DDPActionVideoForesight(cfg, accelerator)

    # Register a requested custom framework without importing unrelated modules.
    if cfg.framework.name not in FRAMEWORK_REGISTRY._registry:
        importlib.import_module(f"{__name__}.{cfg.framework.name}")
    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
